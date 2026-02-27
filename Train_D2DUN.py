import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from einops import rearrange
import numbers
import numpy as np
from thop import profile
from utils import transform
import random
import torch_dct as dct
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
from time import time

from utils import evaluate, transform

parser = ArgumentParser(description='LUCMT-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0,1', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--data_path', type=str, default='T2', help='Path to the dataset')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list

# w1 = torch.tensor([2.0])
# w2 = torch.tensor([3.0])
# w3 = torch.tensor([2.0])
# w4 = torch.tensor([3.0])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 2
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# train.mat fastMRI_Barin_T1_1680.mat data\Xe\Xe_train.mat
Training_data_Name = 'fastMRI_Barin_T1_1680.mat'
# /code/data/T1/train/ code/data/Xe
Training_data = sio.loadmat('/code/data/T1/train/%s' % (Training_data_Name))
Training_labels = Training_data['reconstruction_esc']

nrtrain = Training_labels.shape[0]  # number of training blocks
print('number of train is', nrtrain)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def zero_filled(x, mask, mod=False, norm=False):
    x_dim_0 = x.shape[0]
    x_dim_1 = x.shape[1]
    x_dim_2 = x.shape[2]
    x_dim_3 = x.shape[3]
    x = x.view(-1, x_dim_2, x_dim_3, 1)

    x_real = x
    x_imag = torch.zeros_like(x_real)
    x_complex = torch.cat([x_real, x_imag], 3)

    x_kspace = torch.fft.fft2(x_complex)
    y_kspace = x_kspace * mask
    xu = torch.fft.ifft2(y_kspace)

    if not mod:
        xu_ret = xu[:, :, :, 0:1]
    else:
        xu_ret = torch.sqrt(xu[..., 0:1] ** 2 + xu[..., 1:2] ** 2)

    xu_ret = xu_ret.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
    xu_ret = xu_ret.float()

    return xu_ret

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sigmoid(input * t)  
        out = (out >= 0.5).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t * 2), 2)) * grad_output 
        return grad_input, None, None, None

class blockNL(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        
        # 修改为处理32通道输入
        self.norm_x = LayerNorm(32, 'WithBias')  # 从1改为32
        self.norm_z = LayerNorm(31, 'WithBias') 

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        # 修改输入通道数从1改为32
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.channels, kernel_size=1, stride=1, bias=True),  # 32->31
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        # 修改输入通道数从1改为32
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.channels, kernel_size=1, stride=1, bias=True),  # 32->31
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        # 修改输入通道数从31+1=32改为31+32=63
        self.v = nn.Conv2d(in_channels=self.channels+32, out_channels=32, kernel_size=1, stride=1, bias=True)  # 63->32
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )
        
        self.w3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w4 = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x, z, w3, w4):
        b, c, h, w = x.shape
        x0 = self.norm_x(x)  
        z0 = self.norm_z(z)  
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1) 
        x1 = self.p(x0)  
        x1 = x1.view(b, c, -1) 
        x2 = self.g1(x0)
        x_v = x2.view(b, c, -1) 
        z2 = self.g2(z0) 
        z_v = z2.view(b, c, -1) 

        num_heads = 4  
        x1_heads = x1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        z1_heads = z1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        z_v_heads = z_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        x_v_heads = x_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  

        x1_heads = torch.nn.functional.normalize(x1_heads, dim=-1)
        z1_heads = torch.nn.functional.normalize(z1_heads, dim=-1)
        x_t_heads = x1_heads.permute(0, 1, 3, 2)  
        att_heads = torch.matmul(z1_heads, x_t_heads) 
        att_heads = self.softmax(att_heads)  

        v_heads = self.w3*z_v_heads+self.w4*x_v_heads

        out_x_heads = torch.matmul(att_heads, v_heads)  
        out_x_heads = out_x_heads.view(b, c, h, w)  

        out_x_heads = self.w(out_x_heads) + self.pos_emb(z2) + z  
        y = self.v(torch.cat([x, out_x_heads], 1))  # x是32通道，out_x_heads是31通道，拼接后是63通道
        return y

class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_qv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
        self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2 = nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, pre, cur, w1, w2):
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q,v1 = self.conv_qv1(cur_ln).chunk(2, dim=1)
        q = q.view(b, c, -1)  
        v1 = v1.view(b, c, -1)
        k, v2 = self.conv_kv(pre_ln).chunk(2, dim=1)  
        k = k.view(b, c, -1)
        v2 = v2.view(b, c, -1)
        
        num_heads = 4  
        q = q.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        k = k.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        v1 = v1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        v2 = v2.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 1, 3, 2))  
        att = self.softmax(att)
        
        v = self.w1*v1+self.w2*v2
        
        out = torch.matmul(att, v)  
        out = out.permute(0, 2, 1, 3).contiguous().view(b, c, h, w)  
        out = self.conv_out(out) + cur

        return out

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31) 
        self.nonlo = blockNL(channels=31) 
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        
        # 通道扩展层 - 将1通道扩展到32通道
        self.channel_expand = nn.Conv2d(1, 32, 3, padding=1)
        
        # 梯度下降模块 (对应论文中的梯度计算)
        self.grad_module = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1)
        )
        
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        
        # 通道压缩层 - 将32通道压缩回1通道
        self.channel_compress = nn.Conv2d(32, 1, 3, padding=1)
        
    def forward(self, x, z_pre, z_cur, mask=None, PhiTb=None):
        # 扩展通道: 1 -> 32
        x_expanded = self.channel_expand(x)
        
        z = self.atten(z_pre, z_cur, w1=1.0, w2=1.0)
        
        # 改进的梯度下降步骤
        if PhiTb is not None:
            # 扩展PhiTb的通道
            PhiTb_expanded = self.channel_expand(PhiTb)
            # 梯度下降: x - η * gradient
            x_grad = x_expanded + self.lambda_step * (PhiTb_expanded - x_expanded)
        else:
            x_grad = x_expanded
            
        # 进一步用卷积细化梯度步骤
        x_grad_refined = self.grad_module(x_grad)
        x_input = x_grad + x_grad_refined

        # 非线性块 (近端映射)
        x_input = self.nonlo(x_input, z, w3=1.0, w4=1.0)

        # 残差卷积
        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred_expanded = x_input + x_backward

        # 压缩通道: 32 -> 1
        x_pred = self.channel_compress(x_pred_expanded)

        # 2. 提取辅助特征传给下一层 (取前 31 通道)
        z_out = x_pred_expanded[:, :31, :, :] 

        return x_pred, z_out # 必须返回两个值

        # return x_pred

# 不会被手动覆盖
# ---------------------------
# 条件滤波 (FiLM 风格)
# ---------------------------
class CondFilterV2(nn.Module):
    def __init__(self, nf=16):
        super().__init__()
        self.nf = nf

        # 类似论文中SS模块的结构
        self.head = nn.Conv2d(1, nf//4, 3, padding=1)
        self.body = nn.Sequential(
            nn.Conv2d(nf//4, nf//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf//4, nf//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf//4, nf//4, 3, padding=1)
        )
        
        # CS ratio 条件缩放
        self.scale = nn.Sequential(
            nn.Conv2d(2, nf//4, 1), 
            nn.ReLU(), 
            nn.Conv2d(nf//4, nf//4, 1)
        )
        
        self.tail = nn.Conv2d(nf//4, 2, 3, padding=1)

    def forward(self, x, cs_ratio):
        # 图像特征提取
        x_head = self.head(x)
        
        # 条件缩放
        scaled = self.scale(cs_ratio) * self.body(x_head)
        
        # 输出两个分支的权重
        weights = self.tail(scaled)
        w_D, w_G = weights[:, 0:1], weights[:, 1:2]
        
        return w_D, w_G

def get_zigzag_ordered_indices(h=8, w=8, q=6):
    x, y = [], []
    x1, x2, y1, y2 = 0, 0, 0, 0
    flag = True
    while x2 < h or y1 < w:
        if flag:
            x = [*x, *range(x1, x2 - 1, -1)]
            y = [*y, *range(y1, y2 + 1)]
        else:
            x = [*x, *range(x2, x1 + 1)]
            y = [*y, *range(y2, y1 - 1, -1)]
        flag = not flag
        x1, y1 = (x1 + 1, 0) if (x1 < h - 1) else (h - 1, y1 + 1)
        x2, y2 = (0, y2 + 1) if (y2 < w - 1) else (x2 + 1, w - 1)
    return x[:q], y[:q]

def get_zigzag_truncated_indices(h=8, w=8, q=6):
    if random.randint(0, 1):
        x, y = get_zigzag_ordered_indices(h, w, q)
    else:
        y, x = get_zigzag_ordered_indices(w, h, q)
    return x, y

class COSO_LUCMT(nn.Module):
    def __init__(self, LayerNo, B=32, nf=16, mode='dct_only'):
        """
        mode: 
          'dual': 原有的双分支 (DCT + Gauss)
          'dct_only': 只有频域分支 (100% 采样预算给 DCT)
          'gauss_only': 只有空间分支 (100% 采样预算给 Gauss)
        """
        super().__init__()
        self.LayerNo = LayerNo
        self.B = B
        self.N = B * B
        
        # 条件滤波网络
        self.cond_filter = CondFilterV2(nf=nf)
        
        # 高斯分支权重 (固定)
        U, S, V = torch.linalg.svd(torch.randn(self.N, self.N))
        self.A_weight_G = nn.Parameter(U.mm(V).reshape(self.N, 1, B, B), requires_grad=False)
        
        # 重建网络 - 修改为处理1通道输入
        self.fe = nn.Conv2d(1, 31, 3, padding=1)  # 输入1通道，输出31通道
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1) # 输入1通道，输出31通道
        self.fcs = nn.ModuleList([BasicBlock() for _ in range(LayerNo)])

        self.mode = mode 

    def define_sampling_operators(self, x, q_G, q_DCT):
        """定义采样和重建操作符，类似论文中的A和AT函数"""
        b, c, h, w = x.shape
        n = h * w
        h_B, w_B = h // self.B, w // self.B
        
        # 随机像素置乱
        perm = torch.randperm(n, device=x.device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(n, device=x.device)
        
        # 高斯分支随机权重
        A_weight_G = self.A_weight_G[torch.randperm(self.N, device=x.device)].to(x.device)
        
        # 创建掩码
        mask_G = (torch.arange(self.N, device=x.device).view(1, self.N).expand(b, self.N) 
                 < q_G.view(b, 1)).view(b, self.N, 1, 1)
        mask_DCT = (torch.arange(self.N, device=x.device).view(1, self.N).expand(b, self.N) 
                   < q_DCT.view(b, 1)).view(b, self.N, 1, 1)
        
        # 获取DCT Zig-Zag索引
        DCT_x, DCT_y = get_zigzag_truncated_indices(h, w, n)
        
        # 定义采样操作
        def A_G(z):
            z_perm = z.reshape(b, c, n)[:, :, perm].reshape(b, c, h, w)
            return F.conv2d(z_perm, A_weight_G, stride=self.B) * mask_G
        
        def A_DCT(z):
            dct_coeff = dct.dct_2d(z, norm='ortho')
            selected = dct_coeff[:, :, DCT_x, DCT_y].reshape(b, self.N, h_B, w_B)
            return selected * mask_DCT
        
        def AT_G(z):
            conv_trans = F.conv_transpose2d(z, A_weight_G, stride=self.B)
            return conv_trans.reshape(b, c, n)[:, :, perm_inv].reshape(b, c, h, w)
        
        def AT_DCT(z):
            z_full = torch.zeros(b, 1, h, w, device=x.device)
            z_full[:, :, DCT_x, DCT_y] = z.reshape(b, 1, -1)
            return dct.idct_2d(z_full, norm='ortho')
        
        A = lambda z: [A_G(z[:, 0:1]), A_DCT(z[:, 1:2])]
        AT = lambda z: torch.cat([AT_G(z[0]), AT_DCT(z[1])], dim=1)
        
        return A, AT, mask_G, mask_DCT

    def forward(self, x, cs_ratio_batch):
        b, c, h, w = x.shape
        
        # 计算双分支测量数 (默认比例: γ_D=0.4γ, γ_G=0.6γ)
        total_m = int(cs_ratio_batch[0].item() * self.N)
        if self.mode == 'dct_only':
            q_G = torch.zeros(b, device=x.device).int()
            q_DCT = torch.full((b,), total_m, device=x.device).int()
        elif self.mode == 'gauss_only':
            q_G = torch.full((b,), total_m, device=x.device).int()
            q_DCT = torch.zeros(b, device=x.device).int()
        else: # dual mode (4:6 比例)
            q_DCT = torch.tensor([int(total_m * 0.4)] * b, device=x.device).int()
            q_G = torch.tensor([total_m - int(total_m * 0.4)] * b, device=x.device).int()
        # q_G = torch.tensor([total_measurements * 0.6] * b, device=x.device).int()
        # q_DCT = torch.tensor([total_measurements * 0.4] * b, device=x.device).int()
        
        # 设置CS比率条件
        cs_ratio_G = (q_G / self.N).view(b, 1, 1, 1)
        cs_ratio_DCT = (q_DCT / self.N).view(b, 1, 1, 1)
        cs_ratio = torch.cat([cs_ratio_G, cs_ratio_DCT], dim=1)
        
        # 深度条件滤波
        w_D, w_G = self.cond_filter(x, cs_ratio)
        # 根据模式准备采样输入 (保持通道逻辑一致)
        x_D = x * w_D if self.mode != 'gauss_only' else torch.zeros_like(x)
        x_G = x * w_G if self.mode != 'dct_only' else torch.zeros_like(x)
        # x_D = x * w_D  # DCT分支输入
        # x_G = x * w_G  # 高斯分支输入
        
        # 定义采样操作符
        A, AT, mask_G, mask_DCT = self.define_sampling_operators(x, q_G, q_DCT)
        
        # 双分支采样
        x_filtered = torch.cat([x_G, x_D], dim=1)  # [B, 2, H, W]
        y = A(x_filtered)
        
        # 初始化重建 (使用AT操作)
        x_init_dual = AT(y)  # [B, 2, H, W]
        
        # 将双通道合并为单通道
        if self.mode == 'dual':
            x_init = torch.mean(x_init_dual, dim=1, keepdim=True)
        elif self.mode == 'dct_only':
            x_init = x_init_dual[:, 1:2, :, :] # 取 DCT 支路结果
        else:
            x_init = x_init_dual[:, 0:1, :, :] # 取 Gauss 支路结果
        # x_init = torch.mean(x_init_dual, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 重建网络
        z_pre = self.fe(x_init)  # [B, 31, H, W]
        z_cur = self.fe2(x_init) # [B, 31, H, W]
        x_recon = x_init         # [B, 1, H, W]
        for i in range(self.LayerNo):
            x_dual, z_next = self.fcs[i](x_recon, z_pre, z_cur, mask=None, PhiTb=x_init)
            x_recon = x_dual  # BasicBlock现在输出[B, 1, H, W]
            z_pre = z_cur
            z_cur = z_next
            
        return x_recon, y, A, q_G, q_DCT, (w_D, w_G)


# class DataConsistency(nn.Module):
#     def __init__(self):
#         super(DataConsistency, self).__init__()

#     def forward(self, x_recon, y_measured, mask):
#         """
#         x_recon: 当前迭代重建的图像 [B, 1, H, W]
#         y_measured: 原始采样到的 K 空间测量值 (复数形式)
#         mask: 采样掩码 [B, 1, H, W]
#         """
#         # 1. 将图像转到频域 (FFT)
#         # 假设 x_recon 是实数图像，fft2 会自动处理
#         x_kspace = torch.fft.fft2(x_recon)
        
#         # 2. 替换采样点：在 mask 为 1 的地方用真实 y，在 0 的地方用预测值
#         # 注意：需要确保 y_measured 和 x_kspace 维度一致
#         # 如果是标准 MRI，DC 公式如下：
#         out_kspace = x_kspace + (y_measured - x_kspace) * mask
        
#         # 3. 转回图像域 (IFFT)
#         x_res = torch.fft.ifft2(out_kspace)
#         return torch.abs(x_res) # 返回实数部分（幅值）

# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')

# def to_4d(x,h,w):
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# def zero_filled(x, mask, mod=False, norm=False):
#     x_dim_0 = x.shape[0]
#     x_dim_1 = x.shape[1]
#     x_dim_2 = x.shape[2]
#     x_dim_3 = x.shape[3]
#     x = x.view(-1, x_dim_2, x_dim_3, 1)

#     x_real = x
#     x_imag = torch.zeros_like(x_real)
#     x_complex = torch.cat([x_real, x_imag], 3)

#     x_kspace = torch.fft.fft2(x_complex)
#     y_kspace = x_kspace * mask
#     xu = torch.fft.ifft2(y_kspace)

#     if not mod:
#         xu_ret = xu[:, :, :, 0:1]
#     else:
#         xu_ret = torch.sqrt(xu[..., 0:1] ** 2 + xu[..., 1:2] ** 2)

#     xu_ret = xu_ret.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
#     xu_ret = xu_ret.float()

#     return xu_ret

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#         assert len(normalized_shape) == 1
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight

# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#         assert len(normalized_shape) == 1
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)

# class BinaryQuantize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, k, t):
#         ctx.save_for_backward(input, k, t)
#         out = torch.sigmoid(input * t)  
#         out = (out >= 0.5).float()
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, k, t = ctx.saved_tensors
#         grad_input = k * t * (1 - torch.pow(torch.tanh(input * t * 2), 2)) * grad_output 
#         return grad_input, None, None, None

# class blockNL(torch.nn.Module):
#     def __init__(self, channels):
#         super(blockNL, self).__init__()
#         self.channels = channels
#         self.softmax = nn.Softmax(dim=-1)
        
#         # 修改为处理32通道输入
#         self.norm_x = LayerNorm(32, 'WithBias')  # 从1改为32
#         self.norm_z = LayerNorm(31, 'WithBias') 

#         self.t = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
#         )
#         # 修改输入通道数从1改为32
#         self.p = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=self.channels, kernel_size=1, stride=1, bias=True),  # 32->31
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
#         )
#         # 修改输入通道数从1改为32
#         self.g1 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=self.channels, kernel_size=1, stride=1, bias=True),  # 32->31
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
#         )
#         self.g2 = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
#         )
#         self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
#         # 修改输入通道数从31+1=32改为31+32=63
#         self.v = nn.Conv2d(in_channels=self.channels+32, out_channels=32, kernel_size=1, stride=1, bias=True)  # 63->32
#         self.pos_emb = nn.Sequential(
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
#             nn.GELU(),
#             nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
#         )
        
#         self.w3 = nn.Parameter(torch.randn(1, requires_grad=True))
#         self.w4 = nn.Parameter(torch.randn(1, requires_grad=True))

#     def forward(self, x, z):
#         b, c, h, w = x.shape
#         x0 = self.norm_x(x)  
#         z0 = self.norm_z(z)  
#         z1 = self.t(z0)
#         b, c, h, w = z1.shape
#         z1 = z1.view(b, c, -1) 
#         x1 = self.p(x0)  
#         x1 = x1.view(b, c, -1) 
#         x2 = self.g1(x0)
#         x_v = x2.view(b, c, -1) 
#         z2 = self.g2(z0) 
#         z_v = z2.view(b, c, -1) 

#         num_heads = 4  
#         x1_heads = x1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
#         z1_heads = z1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
#         z_v_heads = z_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
#         x_v_heads = x_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  

#         x1_heads = torch.nn.functional.normalize(x1_heads, dim=-1)
#         z1_heads = torch.nn.functional.normalize(z1_heads, dim=-1)
#         x_t_heads = x1_heads.permute(0, 1, 3, 2)  
#         att_heads = torch.matmul(z1_heads, x_t_heads) 
#         att_heads = self.softmax(att_heads)  

#         v_heads = self.w3*z_v_heads+self.w4*x_v_heads

#         out_x_heads = torch.matmul(att_heads, v_heads)  
#         out_x_heads = out_x_heads.view(b, c, h, w)  

#         out_x_heads = self.w(out_x_heads) + self.pos_emb(z2) + z  
#         y = self.v(torch.cat([x, out_x_heads], 1))  # x是32通道，out_x_heads是31通道，拼接后是63通道
#         return y

# class Atten(torch.nn.Module):
#     def __init__(self, channels):
#         super(Atten, self).__init__()
               
#         self.channels = channels
#         self.softmax = nn.Softmax(dim=-1)
#         self.norm1 = LayerNorm(self.channels, 'WithBias')
#         self.norm2 = LayerNorm(self.channels, 'WithBias')
#         self.conv_qv1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
#         )
#         self.conv_kv = nn.Sequential(
#             nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
#             nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
#         )
#         self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
#         self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))
#         self.w2 = nn.Parameter(torch.randn(1, requires_grad=True))
    
#     def forward(self, pre, cur, w1, w2):
#         b, c, h, w = pre.shape
#         pre_ln = self.norm1(pre)
#         cur_ln = self.norm2(cur)
#         q,v1 = self.conv_qv1(cur_ln).chunk(2, dim=1)
#         q = q.view(b, c, -1)  
#         v1 = v1.view(b, c, -1)
#         k, v2 = self.conv_kv(pre_ln).chunk(2, dim=1)  
#         k = k.view(b, c, -1)
#         v2 = v2.view(b, c, -1)
        
#         num_heads = 4  
#         q = q.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
#         k = k.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
#         v1 = v1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
#         v2 = v2.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#         att = torch.matmul(q, k.permute(0, 1, 3, 2))  
#         att = self.softmax(att)
        
#         v = self.w1*v1+self.w2*v2
        
#         out = torch.matmul(att, v)  
#         out = out.permute(0, 2, 1, 3).contiguous().view(b, c, h, w)  
#         out = self.conv_out(out) + cur

#         return out

# class BasicBlock(torch.nn.Module):
#     def __init__(self):
#         super(BasicBlock, self).__init__()

#         self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
#         self.atten = Atten(31) 
#         self.nonlo = blockNL(channels=31) 
#         self.norm1 = LayerNorm(32, 'WithBias')
#         self.norm2 = LayerNorm(32, 'WithBias')
        
#         # 通道扩展层 - 将1通道扩展到32通道
#         self.channel_expand = nn.Conv2d(1, 32, 3, padding=1)
        
#         # 梯度下降模块 (对应论文中的梯度计算)
#         self.grad_module = nn.Sequential(
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1)
#         )
        
#         self.conv_forward = nn.Sequential(
#             nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
#             nn.GELU(),
#             nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
#         )
#         self.conv_backward = nn.Sequential(
#             nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
#             nn.GELU(),
#             nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
#             nn.GELU(),
#             nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
#         )

#         # 新增：频域 DC 层
#         self.dc_layer = DataConsistency()
        
#         # 新增：频域特征细化 (可选，增加创新性)
#         self.kspace_conv = nn.Sequential(
#             nn.Conv2d(2, 32, 1), # 2通道因为 K 空间有实部虚部
#             nn.ReLU(),
#             nn.Conv2d(32, 2, 1)
#         )
        
#         # 通道压缩层 - 将32通道压缩回1通道
#         self.channel_compress = nn.Conv2d(32, 1, 3, padding=1)
        
#     def forward(self, x, z_pre, z_cur, mask_G=None, y_G=None, mask_DCT=None, y_DCT=None, PhiTb=None, DCT_indices=None):    
#     # def forward(self, x, z_pre, z_cur, mask=None, PhiTb=None):
#         # --- A. 图像域处理 (你原有的逻辑) ---
#         x_expanded = self.channel_expand(x)     
#         z = self.atten(z_pre, z_cur, w1=1.0, w2=1.0)
        
#         # --- 第二步：高斯分支的双域修正 (基于梯度的 DC) ---
#         # 改进的梯度下降步骤
#         if PhiTb is not None:
#             # 扩展PhiTb的通道 x = x + η * A_T(y - Ax)
#             PhiTb_expanded = self.channel_expand(PhiTb)
#             # 梯度下降: x - η * gradient
#             x_grad = x_expanded + self.lambda_step * (PhiTb_expanded - x_expanded)
#         else:
#             x_grad = x_expanded
            
#         # 进一步用卷积细化梯度步骤
#         x_grad_refined = self.grad_module(x_grad)
#         x_input = x_grad + x_grad_refined

#         # 非线性块 (近端映射)
#         x_input = self.nonlo(x_input, z)

#         # 残差卷积
#         x = self.norm1(x_input)
#         x_forward = self.conv_forward(x) + x_input
#         x = self.norm2(x_forward)
#         x_backward = self.conv_backward(x) + x_forward
#         x_pred_expanded = x_input + x_backward

#         # 压缩通道: 32 -> 1
#         x_pred = self.channel_compress(x_pred_expanded)

#         # 路 B: 提取出辅助特征，作为下一轮迭代的 z_cur
#         # 我们可以取前 31 个通道作为 z，或者再加一个卷积提取 z
#         z_out = x_pred_expanded[:, :31, :, :] 

#          # --- B. 频域处理 (跨域融合关键点) ---
#         if y_DCT is not None and DCT_indices is not None:
#             # 1. 转到 DCT 域
#             b, _, h, w = x_pred.shape
#             x_dct = dct.dct_2d(x_pred, norm='ortho')
#             # 将 y_DCT (测量值) 填回到全图 DCT 系数中
#             # 假设 y_DCT 展平后的形状与索引匹配
#             idx_x, idx_y = DCT_indices
            
#             # 先克隆一份，避免原地操作错误
#             x_dct_new = x_dct.clone()
#             # 只替换采样到的点
#             # y_DCT 需要根据你的 A_DCT 逻辑 flatten
#             x_dct_new[:, :, idx_x, idx_y] = y_DCT.view(b, 1, -1) 
            
#             # 反变换回图像
#             x_final = dct.idct_2d(x_dct_new, norm='ortho')
#         else:
#             x_final = x_pred

#         return x_final, z_out # 返回两个值！

# # 不会被手动覆盖
# # ---------------------------
# # 条件滤波 (FiLM 风格)
# # ---------------------------
# class CondFilterV2(nn.Module):
#     def __init__(self, nf=16):
#         super().__init__()
#         self.nf = nf

#         # 类似论文中SS模块的结构
#         self.head = nn.Conv2d(1, nf//4, 3, padding=1)
#         self.body = nn.Sequential(
#             nn.Conv2d(nf//4, nf//4, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(nf//4, nf//4, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(nf//4, nf//4, 3, padding=1)
#         )
        
#         # CS ratio 条件缩放
#         self.scale = nn.Sequential(
#             nn.Conv2d(2, nf//4, 1), 
#             nn.ReLU(), 
#             nn.Conv2d(nf//4, nf//4, 1)
#         )
        
#         self.tail = nn.Conv2d(nf//4, 2, 3, padding=1)

#     def forward(self, x, cs_ratio):
#         # 图像特征提取
#         x_head = self.head(x)
        
#         # 条件缩放
#         scaled = self.scale(cs_ratio) * self.body(x_head)
        
#         # 输出两个分支的权重
#         weights = self.tail(scaled)
#         w_D, w_G = weights[:, 0:1], weights[:, 1:2]
        
#         return w_D, w_G

# def get_zigzag_ordered_indices(h=8, w=8, q=6):
#     x, y = [], []
#     x1, x2, y1, y2 = 0, 0, 0, 0
#     flag = True
#     while x2 < h or y1 < w:
#         if flag:
#             x = [*x, *range(x1, x2 - 1, -1)]
#             y = [*y, *range(y1, y2 + 1)]
#         else:
#             x = [*x, *range(x2, x1 + 1)]
#             y = [*y, *range(y2, y1 - 1, -1)]
#         flag = not flag
#         x1, y1 = (x1 + 1, 0) if (x1 < h - 1) else (h - 1, y1 + 1)
#         x2, y2 = (0, y2 + 1) if (y2 < w - 1) else (x2 + 1, w - 1)
#     return x[:q], y[:q]

# def get_zigzag_truncated_indices(h=8, w=8, q=6):
#     if random.randint(0, 1):
#         x, y = get_zigzag_ordered_indices(h, w, q)
#     else:
#         y, x = get_zigzag_ordered_indices(w, h, q)
#     return x, y

# class COSO_LUCMT(nn.Module):
#     def __init__(self, LayerNo, B=32, nf=16):
#         super().__init__()
#         self.LayerNo = LayerNo
#         self.B = B
#         self.N = B * B
        
#         # 条件滤波网络
#         self.cond_filter = CondFilterV2(nf=nf)
        
#         # 高斯分支权重 (固定)
#         U, S, V = torch.linalg.svd(torch.randn(self.N, self.N))
#         self.A_weight_G = nn.Parameter(U.mm(V).reshape(self.N, 1, B, B), requires_grad=False)
        
#         # 重建网络 - 修改为处理1通道输入
#         self.fe = nn.Conv2d(1, 31, 3, padding=1)  # 输入1通道，输出31通道
#         self.fe2 = nn.Conv2d(1, 31, 3, padding=1) # 输入1通道，输出31通道
#         self.fcs = nn.ModuleList([BasicBlock() for _ in range(LayerNo)])

#     def define_sampling_operators(self, x, q_G, q_DCT):
#         """定义采样和重建操作符，类似论文中的A和AT函数"""
#         b, c, h, w = x.shape
#         n = h * w
#         h_B, w_B = h // self.B, w // self.B
        
#         # 随机像素置乱
#         perm = torch.randperm(n, device=x.device)
#         perm_inv = torch.empty_like(perm)
#         perm_inv[perm] = torch.arange(n, device=x.device)
        
#         # 高斯分支随机权重
#         A_weight_G = self.A_weight_G[torch.randperm(self.N, device=x.device)].to(x.device)
        
#         # 创建掩码
#         mask_G = (torch.arange(self.N, device=x.device).view(1, self.N).expand(b, self.N) 
#                  < q_G.view(b, 1)).view(b, self.N, 1, 1)
#         mask_DCT = (torch.arange(self.N, device=x.device).view(1, self.N).expand(b, self.N) 
#                    < q_DCT.view(b, 1)).view(b, self.N, 1, 1)
        
#         # 获取DCT Zig-Zag索引
#         DCT_x, DCT_y = get_zigzag_truncated_indices(h, w, n)

#         # 我们只需要前 q_DCT 个索引作为有效采样点
#         # 这里简化处理：直接返回全量索引，Block 内部根据 q_DCT 截取
#         # 或者直接返回 truncated 后的索引
#         def get_mask_indices(b_idx):
#             q = q_DCT[b_idx].item()
#             return DCT_x[:q], DCT_y[:q]
        
#         # 定义采样操作
#         def A_G(z):
#             z_perm = z.reshape(b, c, n)[:, :, perm].reshape(b, c, h, w)
#             return F.conv2d(z_perm, A_weight_G, stride=self.B) * mask_G
        
#         def A_DCT(z):
#             dct_coeff = dct.dct_2d(z, norm='ortho')
#             selected = dct_coeff[:, :, DCT_x, DCT_y].reshape(b, self.N, h_B, w_B)
#             return selected * mask_DCT
        
#         def AT_G(z):
#             conv_trans = F.conv_transpose2d(z, A_weight_G, stride=self.B)
#             return conv_trans.reshape(b, c, n)[:, :, perm_inv].reshape(b, c, h, w)
        
#         def AT_DCT(z):
#             z_full = torch.zeros(b, 1, h, w, device=x.device)
#             z_full[:, :, DCT_x, DCT_y] = z.reshape(b, 1, -1)
#             return dct.idct_2d(z_full, norm='ortho')
        
#         A = lambda z: [A_G(z[:, 0:1]), A_DCT(z[:, 1:2])]
#         AT = lambda z: torch.cat([AT_G(z[0]), AT_DCT(z[1])], dim=1)
        
#         # 但为了方便，我们将 DCT_x, DCT_y 作为 metadata 返回
#         return A, AT, mask_G, mask_DCT, (DCT_x, DCT_y)

#     def forward(self, x, cs_ratio_batch):
#         b, c, h, w = x.shape
        
#         # 计算双分支测量数 (默认比例: γ_D=0.4γ, γ_G=0.6γ)
#         total_measurements = int(cs_ratio_batch[0].item() * self.N)
#         q_G = torch.tensor([total_measurements * 0.6] * b, device=x.device).int()
#         q_DCT = torch.tensor([total_measurements * 0.4] * b, device=x.device).int()
        
#         # 设置CS比率条件
#         cs_ratio_G = (q_G / self.N).view(b, 1, 1, 1)
#         cs_ratio_DCT = (q_DCT / self.N).view(b, 1, 1, 1)
#         cs_ratio = torch.cat([cs_ratio_G, cs_ratio_DCT], dim=1)
        
#         # 深度条件滤波
#         w_D, w_G = self.cond_filter(x, cs_ratio)
#         x_D = x * w_D  # DCT分支输入
#         x_G = x * w_G  # 高斯分支输入
        
#         # 定义采样操作符
#         A, AT, mask_G, mask_DCT, DCT_info = self.define_sampling_operators(x, q_G, q_DCT)
        
#         # 双分支采样
#         x_filtered = torch.cat([x_D, x_G], dim=1)  # [B, 2, H, W]
#         y = A(x_filtered)# y[0] 是高斯测量值, y[1] 是 DCT 测量值

#         # # 🚩 在这里插入第二行：检查采样值量级
#         # if getattr(self, 'print_once', True): # 只打印一次
#         #     print(f"y_DCT (采样值) 范围: {y[1].min().item():.2f} to {y[1].max().item():.2f}")
#         #     self.print_once = False # 打印完设为 False
        
#         # 初始化重建 (使用AT操作)
#         x_init_dual = AT(y)  # [B, 2, H, W]
        
#         # 将双通道合并为单通道
#         x_init = torch.mean(x_init_dual, dim=1, keepdim=True)  # [B, 1, H, W]
        
#         # 重建网络
#         z_pre = self.fe(x_init)  # [B, 31, H, W]
#         z_cur = self.fe2(x_init) # [B, 31, H, W]
#         x_recon = x_init         # [B, 1, H, W]
#         for i in range(self.LayerNo):
#             # x_dual = self.fcs[i](x_recon, z_pre, z_cur, mask=None, PhiTb=x_init)
#             # x_recon = x_dual  # BasicBlock现在输出[B, 1, H, W]
#             # z_pre = z_cur
#             # z_cur = x_dual[:, 1:, :, :] if x_dual.shape[1] > 1 else z_cur

#             x_recon, z_next = self.fcs[i](
#                 x_recon, 
#                 z_pre, 
#                 z_cur, 
#                 mask_G=mask_G,     # 对应 mask_G (高斯掩码)
#                 y_G=y[0],          # 对应 y_G (高斯测量值)
#                 mask_DCT=mask_DCT, # 频率域掩码
#                 y_DCT=y[1],        # 频率域真理 (对应第四个优化)
#                 PhiTb=x_init,       # 空间域/高斯路参考 (对应高斯分支的 DC)
#                 DCT_indices=DCT_info # 传入索引
#             )       
#             # 更新 z 的滑动窗口
#             z_pre = z_cur
#             z_cur = z_next 
            
#         return x_recon

model = COSO_LUCMT(layer_num, mode='dct_only')
model = model.to(device)
# 使用 DataParallel 来分配到多个GPU
# model = nn.DataParallel(model, device_ids=[0, 1]) 

print_flag = 1  

print("Training on device:", next(model.parameters()).device)


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len
dataset=RandomDataset(Training_labels, nrtrain)
# print(dataset[0])

if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=2,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# -------------------------
# 日志和模型保存路径
# -------------------------
chosen_cs = 0.2
log_dir = os.path.join("log")
os.makedirs(log_dir, exist_ok=True)
# all:L1SSIMMeans
log_file_name = os.path.join(log_dir, f"Log_MRI_Dct_only_LUCMT_FastMRI-L1SSIMMeans_layer_{layer_num}_group_{group_num}_{chosen_cs}.txt")

model_dir = os.path.join("model", f"MRI_Dct_only_LUCMT_FastMRI-L1SSIMMeans_layer_{layer_num}_group_{group_num}_{chosen_cs}")
os.makedirs(model_dir, exist_ok=True)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

# -------------------------
# CS ratio 训练模式
# -------------------------
train_mode = "fixed"   # "random" 表示随机 CS ratio，"fixed" 表示固定
fixed_cs_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

# 设置绘图后端，防止在无界面的服务器上报错
plt.switch_backend('agg')

# 加载验证集
Val_data = sio.loadmat(r'.\data\T1\val\fastMRI_val_T1_208.mat')
Val_labels = Val_data['reconstruction_esc'] # 假设 key 是这个
nrval = Val_labels.shape[0]

# 验证集 DataLoader
val_loader = DataLoader(dataset=RandomDataset(Val_labels, nrval), 
                        batch_size=batch_size, shuffle=False, num_workers=0)

# 用于记录历史数据
history = {
    'train_loss': [],
    'val_loss': [],
    'val_psnr': []
}

def plot_convergence(history, save_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss 曲线 (左轴)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(history['train_loss'], label='Train Loss', color='tab:red', linestyle='--')
    ax1.plot(history['val_loss'], label='Val Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 创建右轴绘制 PSNR
    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR(dB)', color='tab:blue')
    ax2.plot(history['val_psnr'], label='Val PSNR', color='tab:blue', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title('Convergence Analysis (Loss and PSNR)')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.savefig(save_path)
    plt.close()

# -------------------------
# 训练循环
# -------------------------
for epoch_i in range(start_epoch + 1, end_epoch + 1):
    model.train()
    for data in rand_loader:
        # 1️⃣ 输入
        epoch_train_losses = []
        batch_x = data.to(device)
        # print(batch_x.shape)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
        batch_x, mean, std = transform.normalize_instance(batch_x, eps=1e-11)
        batch_x = batch_x.clamp(-6, 6)

        # # 🚩 在这里插入第一行：检查图像量级
        # print(f"--- 数值检查 ---")
        # print(f"batch_x (图像) 范围: {batch_x.min().item():.2f} to {batch_x.max().item():.2f}")

        cs_ratio_value = chosen_cs  # scalar

        # 转成 tensor，放到 GPU
        cs_ratio_tensor = torch.tensor([[cs_ratio_value]], device=device).float()
        cs_ratio_batch = cs_ratio_tensor.expand(batch_x.shape[0], -1)  # [B,1]

        # 3️⃣ forward
        # x_recon = model(batch_x, cs_ratio_batch)  # 注意这里传 tensor
        x_recon, y_true, A, q_G, q_DCT, (w_D, w_G) = model(batch_x, cs_ratio_batch)

        # loss_all = torch.mean(torch.pow(x_recon - batch_x, 2))
        # # 4️⃣ Loss (尝试使用 L1 Loss)
        # # loss_all = torch.nn.functional.l1_loss(x_recon, batch_x) 
        # # loss_all = torch.mean((x_recon - batch_x) ** 2)
        # # loss_all = torch.mean(torch.pow(x_recon - batch_x, 2))
        # # 在训练循环中
        l1_loss = torch.nn.functional.l1_loss(x_recon, batch_x)
        d_range = batch_x.max() - batch_x.min()
        # ssim 越接近 1 越好，所以 Loss 用 1 - ssim
        # non_negative=True 确保 ssim 为正值，增加稳定性
        ssim_val = ssim(x_recon, batch_x, data_range=d_range, size_average=True)
        loss_pixel = l1_loss + 0.5 * (1 - ssim_val)

        # 2. 测量一致性 Loss (补偿去掉的双域层)
        # 2. 测量一致性 Loss (Measurement Fidelity) - 负责物理保真
        # 【关键修正】：必须应用采样时的条件权重
        x_recon_D = x_recon * w_D # 对重建图施加同样的 DCT 分支权重
        x_recon_G = x_recon * w_G # 对重建图施加同样的 Gaussian 分支权重
        y_recon = A(torch.cat([x_recon_D, x_recon_G], dim=1)) # 模拟当时的物理采样

        # y_true 是前向传播时采样得到的真实测量值
        loss_meas = torch.nn.functional.mse_loss(y_recon[0], y_true[0]) + \
                    torch.nn.functional.mse_loss(y_recon[1], y_true[1])

        # 3. 频域感知损失
        loss_dct = torch.nn.functional.l1_loss(dct.dct_2d(x_recon, norm='ortho'), 
                                                dct.dct_2d(batch_x, norm='ortho'))

        # 总 Loss 引导
        loss_all = loss_pixel + 0.1 * loss_meas + 0.1 * loss_dct
        # loss_all = l1_loss

        # 5️⃣ backward
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        epoch_train_losses.append(loss_all.item())

        # 6️⃣ 打印训练信息
        msg = (f"[{epoch_i:02d}/{end_epoch:02d}] "
               f"CS={cs_ratio_value:.2f} Loss={loss_all.item():.5f}")
        print(msg)

        with open(log_file_name, "a") as f:
            f.write(msg + "\n")

    # --- Validation Phase (每个 Epoch 结束后跑一次) ---
    model.eval()
    epoch_val_losses = []
    epoch_val_psnr = []
    
    with torch.no_grad():
        for data in val_loader:
            batch_x_val = data.to(device)
            if batch_x_val.dim() == 2: batch_x_val = batch_x_val.view(-1, 1, 256, 256)
            elif batch_x_val.dim() == 3: batch_x_val = batch_x_val.unsqueeze(1)

            batch_x_val, mean, std = transform.normalize_instance(batch_x_val, eps=1e-11)
            batch_x_val = batch_x_val.clamp(-6, 6)

            cs_ratio_v = torch.tensor([[chosen_cs]], device=device).expand(batch_x_val.shape[0], -1)
            
            x_res, _, _, _, _, _  = model(batch_x_val, cs_ratio_v)
            
            # 计算验证 Loss
            v_loss = torch.nn.functional.l1_loss(x_res, batch_x_val)
            epoch_val_losses.append(v_loss.item())
            
            # 计算验证 PSNR (在归一化域计算即可，用于观察收敛性)
            # 或者反归一化计算更准，这里推荐在归一化域算，速度快
            for b in range(x_res.shape[0]):
                cur_psnr = evaluate.psnr(x_res[b,0].cpu().numpy(), batch_x_val[b,0].cpu().numpy())
                epoch_val_psnr.append(cur_psnr)

    # 记录并保存历史
    history['train_loss'].append(np.mean(epoch_train_losses))
    history['val_loss'].append(np.mean(epoch_val_losses))
    history['val_psnr'].append(np.mean(epoch_val_psnr))

    # --- 2. 动态命名文件 (将 0.4 替换为 {chosen_cs}) ---
    # 使用 f-string 自动填充当前的采样率
    plot_filename = f"Convergence_Analysis_Dct_only_LUCMT_FastMRI-L1SSIMMeans-CS{chosen_cs}.png"
    mat_filename = f"training_history_Dct_only_LUCMT_FastMRI-L1SSIMMeans-CS{chosen_cs}.mat"

    # --- 3. 执行绘图与保存 ---
    # 这样如果你跑 CS=0.1，文件名就是 ...-CS0.1.png；跑 0.4 就是 ...-CS0.4.png
    plot_convergence(history, os.path.join(log_dir, plot_filename))
    
    # 同时也保存一份数据，方便以后用 Origin 或 Excel 重新画图
    sio.savemat(os.path.join(log_dir, mat_filename), history)
    
    print(f"Epoch {epoch_i} Summary: Train Loss: {history['train_loss'][-1]:.4f}, "
            f"Val Loss: {history['val_loss'][-1]:.4f}, Val PSNR: {history['val_psnr'][-1]:.2f}")

    # 7️⃣ 保存模型
    torch.save(model.state_dict(), f"{model_dir}/net_params_{epoch_i}.pkl")

# # # 分辨率无关的DISCO卷积层
# # class DISCOConv2d(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
# #                  radius=0.1, basis_functions=9, domain_range=(-1, 1)):
# #         super(DISCOConv2d, self).__init__()
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.kernel_size = kernel_size
# #         self.stride = stride
# #         self.padding = padding
        
# #         # 使用更保守的参数
# #         self.radius = radius
# #         self.basis_functions = basis_functions
        
# #         # 更好的权重初始化 - 使用较小的初始值
# #         self.basis_weights = nn.Parameter(
# #             torch.randn(out_channels, in_channels, basis_functions) * 0.01
# #         )
        
# #         # 创建固定的基函数网格
# #         self.register_buffer('basis_grid', self.create_basis_grid())
        
# #         # 偏置项 - 初始化为0
# #         self.bias = nn.Parameter(torch.zeros(out_channels))
        
# #         # 添加数值稳定性的小常数
# #         self.eps = 1e-8
        
# #     def create_basis_grid(self):
# #         """创建基函数网格"""
# #         grid_size = int(math.sqrt(self.basis_functions))
# #         basis_x = torch.linspace(-1, 1, grid_size)
# #         basis_y = torch.linspace(-1, 1, grid_size)
# #         basis_xx, basis_yy = torch.meshgrid(basis_x, basis_y, indexing='ij')
# #         basis_grid = torch.stack([basis_xx, basis_yy], dim=-1).reshape(-1, 2)
# #         return basis_grid
    
# #     def forward(self, x):
# #         """数值稳定的前向传播"""
# #         batch_size, channels, height_in, width_in = x.shape
        
# #         # 检查输入
# #         if torch.isnan(x).any() or torch.isinf(x).any():
# #             print("DISCO卷积输入包含NaN或Inf")
# #             return torch.zeros(batch_size, self.out_channels, height_in, width_in, device=x.device)
        
# #         # 计算输出尺寸
# #         height_out = (height_in + 2 * self.padding - self.kernel_size) // self.stride + 1
# #         width_out = (width_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
# #         # 对输入进行填充
# #         if self.padding > 0:
# #             x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
# #         else:
# #             x_padded = x
        
# #         # 展开输入为局部块
# #         unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)
# #         x_unfolded = unfold(x_padded)  # [B, C*kernel_size*kernel_size, H_out*W_out]
# #         x_unfolded = x_unfolded.view(batch_size, channels, self.kernel_size * self.kernel_size, -1)
        
# #         # 生成核权重
# #         kernel_weights = self.generate_kernel_weights(x.device)
        
# #         # 修复矩阵乘法维度问题
# #         # x_unfolded: [B, C, K, N] 其中 K = kernel_size^2, N = H_out*W_out
# #         # kernel_weights: [1, O, C, K] 其中 O = out_channels
# #         # 我们需要: [B, O, N]
        
# #         # 使用einsum修复维度问题
# #         output = torch.einsum('bcki,bock->boi', x_unfolded, kernel_weights)
        
# #         # 添加偏置并重塑
# #         output = output + self.bias.unsqueeze(-1)
# #         output = output.reshape(batch_size, self.out_channels, height_out, width_out)
        
# #   import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from einops import rearrange
import numbers
import numpy as np
from utils import transform
import random
import torch_dct as dct
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
from utils import evaluate, transform

parser = ArgumentParser(description='D2DUN')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0,1', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--data_path', type=str, default='T2', help='Path to the dataset')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 2
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Training_data_Name = 'fastMRI_Barin_T1_1680.mat'
Training_data = sio.loadmat('./data/T1/train/%s' % (Training_data_Name))
Training_labels = Training_data['reconstruction_esc']

nrtrain = Training_labels.shape[0]  
print('number of train is', nrtrain)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def zero_filled(x, mask, mod=False, norm=False):
    x_dim_0 = x.shape[0]
    x_dim_1 = x.shape[1]
    x_dim_2 = x.shape[2]
    x_dim_3 = x.shape[3]
    x = x.view(-1, x_dim_2, x_dim_3, 1)

    x_real = x
    x_imag = torch.zeros_like(x_real)
    x_complex = torch.cat([x_real, x_imag], 3)

    x_kspace = torch.fft.fft2(x_complex)
    y_kspace = x_kspace * mask
    xu = torch.fft.ifft2(y_kspace)

    if not mod:
        xu_ret = xu[:, :, :, 0:1]
    else:
        xu_ret = torch.sqrt(xu[..., 0:1] ** 2 + xu[..., 1:2] ** 2)

    xu_ret = xu_ret.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
    xu_ret = xu_ret.float()

    return xu_ret

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sigmoid(input * t)  
        out = (out >= 0.5).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t * 2), 2)) * grad_output 
        return grad_input, None, None, None

class blockNL(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm_x = LayerNorm(32, 'WithBias')  
        self.norm_z = LayerNorm(31, 'WithBias') 

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.channels, kernel_size=1, stride=1, bias=True),  # 32->31
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.channels, kernel_size=1, stride=1, bias=True),  # 32->31
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels+32, out_channels=32, kernel_size=1, stride=1, bias=True)  # 63->32
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )
        
        self.w3 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w4 = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x, z, w3, w4):
        b, c, h, w = x.shape
        x0 = self.norm_x(x)  
        z0 = self.norm_z(z)  
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1) 
        x1 = self.p(x0)  
        x1 = x1.view(b, c, -1) 
        x2 = self.g1(x0)
        x_v = x2.view(b, c, -1) 
        z2 = self.g2(z0) 
        z_v = z2.view(b, c, -1) 

        num_heads = 4  
        x1_heads = x1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        z1_heads = z1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        z_v_heads = z_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        x_v_heads = x_v.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  

        x1_heads = torch.nn.functional.normalize(x1_heads, dim=-1)
        z1_heads = torch.nn.functional.normalize(z1_heads, dim=-1)
        x_t_heads = x1_heads.permute(0, 1, 3, 2)  
        att_heads = torch.matmul(z1_heads, x_t_heads) 
        att_heads = self.softmax(att_heads)  

        v_heads = self.w3*z_v_heads+self.w4*x_v_heads

        out_x_heads = torch.matmul(att_heads, v_heads)  
        out_x_heads = out_x_heads.view(b, c, h, w)  

        out_x_heads = self.w(out_x_heads) + self.pos_emb(z2) + z  
        y = self.v(torch.cat([x, out_x_heads], 1)) 
        return y

class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_qv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
        self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w2 = nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, pre, cur, w1, w2):
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q,v1 = self.conv_qv1(cur_ln).chunk(2, dim=1)
        q = q.view(b, c, -1)  
        v1 = v1.view(b, c, -1)
        k, v2 = self.conv_kv(pre_ln).chunk(2, dim=1)  
        k = k.view(b, c, -1)
        v2 = v2.view(b, c, -1)
        
        num_heads = 4  
        q = q.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        k = k.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        v1 = v1.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  
        v2 = v2.view(b, c, num_heads, -1).permute(0, 2, 1, 3)  

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 1, 3, 2))  
        att = self.softmax(att)
        
        v = self.w1*v1+self.w2*v2
        
        out = torch.matmul(att, v)  
        out = out.permute(0, 2, 1, 3).contiguous().view(b, c, h, w)  
        out = self.conv_out(out) + cur

        return out

class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31) 
        self.nonlo = blockNL(channels=31) 
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.channel_expand = nn.Conv2d(1, 32, 3, padding=1)
        
        self.grad_module = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1)
        )
        
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.channel_compress = nn.Conv2d(32, 1, 3, padding=1)
        
    def forward(self, x, z_pre, z_cur, mask=None, PhiTb=None):
        x_expanded = self.channel_expand(x)
        
        z = self.atten(z_pre, z_cur, w1=1.0, w2=1.0)

        if PhiTb is not None:
            PhiTb_expanded = self.channel_expand(PhiTb)
            x_grad = x_expanded + self.lambda_step * (PhiTb_expanded - x_expanded)
        else:
            x_grad = x_expanded
        x_grad_refined = self.grad_module(x_grad)
        x_input = x_grad + x_grad_refined
        x_input = self.nonlo(x_input, z, w3=1.0, w4=1.0)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred_expanded = x_input + x_backward
        x_pred = self.channel_compress(x_pred_expanded)
        z_out = x_pred_expanded[:, :31, :, :] 

        return x_pred, z_out 

class CondFilterV2(nn.Module):
    def __init__(self, nf=16):
        super().__init__()
        self.nf = nf
        self.head = nn.Conv2d(1, nf//4, 3, padding=1)
        self.body = nn.Sequential(
            nn.Conv2d(nf//4, nf//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf//4, nf//4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nf//4, nf//4, 3, padding=1)
        )
        self.scale = nn.Sequential(
            nn.Conv2d(2, nf//4, 1), 
            nn.ReLU(), 
            nn.Conv2d(nf//4, nf//4, 1)
        )
        
        self.tail = nn.Conv2d(nf//4, 2, 3, padding=1)

    def forward(self, x, cs_ratio):
        x_head = self.head(x)
        scaled = self.scale(cs_ratio) * self.body(x_head)
        weights = self.tail(scaled)
        w_D, w_G = weights[:, 0:1], weights[:, 1:2]
        
        return w_D, w_G

def get_zigzag_ordered_indices(h=8, w=8, q=6):
    x, y = [], []
    x1, x2, y1, y2 = 0, 0, 0, 0
    flag = True
    while x2 < h or y1 < w:
        if flag:
            x = [*x, *range(x1, x2 - 1, -1)]
            y = [*y, *range(y1, y2 + 1)]
        else:
            x = [*x, *range(x2, x1 + 1)]
            y = [*y, *range(y2, y1 - 1, -1)]
        flag = not flag
        x1, y1 = (x1 + 1, 0) if (x1 < h - 1) else (h - 1, y1 + 1)
        x2, y2 = (0, y2 + 1) if (y2 < w - 1) else (x2 + 1, w - 1)
    return x[:q], y[:q]

def get_zigzag_truncated_indices(h=8, w=8, q=6):
    if random.randint(0, 1):
        x, y = get_zigzag_ordered_indices(h, w, q)
    else:
        y, x = get_zigzag_ordered_indices(w, h, q)
    return x, y

class D2DUN(nn.Module):
    def __init__(self, LayerNo, B=32, nf=16, mode='dct_only'):
        super().__init__()
        self.LayerNo = LayerNo
        self.B = B
        self.N = B * B
        self.cond_filter = CondFilterV2(nf=nf)

        U, S, V = torch.linalg.svd(torch.randn(self.N, self.N))
        self.A_weight_G = nn.Parameter(U.mm(V).reshape(self.N, 1, B, B), requires_grad=False)
        self.fe = nn.Conv2d(1, 31, 3, padding=1)  
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1) 
        self.fcs = nn.ModuleList([BasicBlock() for _ in range(LayerNo)])

        self.mode = mode 

    def define_sampling_operators(self, x, q_G, q_DCT):
        b, c, h, w = x.shape
        n = h * w
        h_B, w_B = h // self.B, w // self.B

        perm = torch.randperm(n, device=x.device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(n, device=x.device)
        A_weight_G = self.A_weight_G[torch.randperm(self.N, device=x.device)].to(x.device)
        mask_G = (torch.arange(self.N, device=x.device).view(1, self.N).expand(b, self.N) 
                 < q_G.view(b, 1)).view(b, self.N, 1, 1)
        mask_DCT = (torch.arange(self.N, device=x.device).view(1, self.N).expand(b, self.N) 
                   < q_DCT.view(b, 1)).view(b, self.N, 1, 1)
        DCT_x, DCT_y = get_zigzag_truncated_indices(h, w, n)
        
        def A_G(z):
            z_perm = z.reshape(b, c, n)[:, :, perm].reshape(b, c, h, w)
            return F.conv2d(z_perm, A_weight_G, stride=self.B) * mask_G
        
        def A_DCT(z):
            dct_coeff = dct.dct_2d(z, norm='ortho')
            selected = dct_coeff[:, :, DCT_x, DCT_y].reshape(b, self.N, h_B, w_B)
            return selected * mask_DCT
        
        def AT_G(z):
            conv_trans = F.conv_transpose2d(z, A_weight_G, stride=self.B)
            return conv_trans.reshape(b, c, n)[:, :, perm_inv].reshape(b, c, h, w)
        
        def AT_DCT(z):
            z_full = torch.zeros(b, 1, h, w, device=x.device)
            z_full[:, :, DCT_x, DCT_y] = z.reshape(b, 1, -1)
            return dct.idct_2d(z_full, norm='ortho')
        
        A = lambda z: [A_G(z[:, 0:1]), A_DCT(z[:, 1:2])]
        AT = lambda z: torch.cat([AT_G(z[0]), AT_DCT(z[1])], dim=1)
        
        return A, AT, mask_G, mask_DCT

    def forward(self, x, cs_ratio_batch):
        b, c, h, w = x.shape
        total_m = int(cs_ratio_batch[0].item() * self.N)
        if self.mode == 'dct_only':
            q_G = torch.zeros(b, device=x.device).int()
            q_DCT = torch.full((b,), total_m, device=x.device).int()
        elif self.mode == 'gauss_only':
            q_G = torch.full((b,), total_m, device=x.device).int()
            q_DCT = torch.zeros(b, device=x.device).int()
        else: 
            q_DCT = torch.tensor([int(total_m * 0.4)] * b, device=x.device).int()
            q_G = torch.tensor([total_m - int(total_m * 0.4)] * b, device=x.device).int()

        cs_ratio_G = (q_G / self.N).view(b, 1, 1, 1)
        cs_ratio_DCT = (q_DCT / self.N).view(b, 1, 1, 1)
        cs_ratio = torch.cat([cs_ratio_G, cs_ratio_DCT], dim=1)
        w_D, w_G = self.cond_filter(x, cs_ratio)
        x_D = x * w_D if self.mode != 'gauss_only' else torch.zeros_like(x)
        x_G = x * w_G if self.mode != 'dct_only' else torch.zeros_like(x)

        A, AT, mask_G, mask_DCT = self.define_sampling_operators(x, q_G, q_DCT)
        
        x_filtered = torch.cat([x_G, x_D], dim=1)  
        y = A(x_filtered)
        
        x_init_dual = AT(y)  
        

        if self.mode == 'dual':
            x_init = torch.mean(x_init_dual, dim=1, keepdim=True)
        elif self.mode == 'dct_only':
            x_init = x_init_dual[:, 1:2, :, :] 
        else:
            x_init = x_init_dual[:, 0:1, :, :] 
       
        z_pre = self.fe(x_init) 
        z_cur = self.fe2(x_init) 
        x_recon = x_init         
        for i in range(self.LayerNo):
            x_dual, z_next = self.fcs[i](x_recon, z_pre, z_cur, mask=None, PhiTb=x_init)
            x_recon = x_dual  
            z_pre = z_cur
            z_cur = z_next
            
        return x_recon, y, A, q_G, q_DCT, (w_D, w_G)

model = D2DUN(layer_num, mode='dual')
model = model.to(device)


print_flag = 1  

print("Training on device:", next(model.parameters()).device)


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len
dataset=RandomDataset(Training_labels, nrtrain)

if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=2,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

chosen_cs = 0.2
log_dir = os.path.join("log")
os.makedirs(log_dir, exist_ok=True)

log_file_name = os.path.join(log_dir, f"Log_layer_{layer_num}_group_{group_num}_{chosen_cs}.txt")

model_dir = os.path.join("model", f"MRI_layer_{layer_num}_group_{group_num}_{chosen_cs}")
os.makedirs(model_dir, exist_ok=True)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

train_mode = "fixed"  
fixed_cs_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.switch_backend('agg')

Val_data = sio.loadmat('.\data\T1\val\fastMRI_val_T1_208.mat')
Val_labels = Val_data['reconstruction_esc'] 
nrval = Val_labels.shape[0]

val_loader = DataLoader(dataset=RandomDataset(Val_labels, nrval), 
                        batch_size=batch_size, shuffle=False, num_workers=0)

history = {
    'train_loss': [],
    'val_loss': [],
    'val_psnr': []
}


for epoch_i in range(start_epoch + 1, end_epoch + 1):
    model.train()
    for data in rand_loader:
        epoch_train_losses = []
        batch_x = data.to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
        batch_x, mean, std = transform.normalize_instance(batch_x, eps=1e-11)
        batch_x = batch_x.clamp(-6, 6)

        cs_ratio_value = chosen_cs  

        cs_ratio_tensor = torch.tensor([[cs_ratio_value]], device=device).float()
        cs_ratio_batch = cs_ratio_tensor.expand(batch_x.shape[0], -1)  

        x_recon, y_true, A, q_G, q_DCT, (w_D, w_G) = model(batch_x, cs_ratio_batch)
        l1_loss = torch.nn.functional.l1_loss(x_recon, batch_x)
        d_range = batch_x.max() - batch_x.min()
        ssim_val = ssim(x_recon, batch_x, data_range=d_range, size_average=True)
        loss_pixel = l1_loss + 0.5 * (1 - ssim_val)

        x_recon_D = x_recon * w_D 
        x_recon_G = x_recon * w_G 
        y_recon = A(torch.cat([x_recon_D, x_recon_G], dim=1)) 

        loss_meas = torch.nn.functional.mse_loss(y_recon[0], y_true[0]) + \
                    torch.nn.functional.mse_loss(y_recon[1], y_true[1])

        loss_dct = torch.nn.functional.l1_loss(dct.dct_2d(x_recon, norm='ortho'), 
                                                dct.dct_2d(batch_x, norm='ortho'))

        loss_all = loss_pixel + 0.1 * loss_meas + 0.1 * loss_dct

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        epoch_train_losses.append(loss_all.item())

        msg = (f"[{epoch_i:02d}/{end_epoch:02d}] "
               f"CS={cs_ratio_value:.2f} Loss={loss_all.item():.5f}")
        print(msg)

        with open(log_file_name, "a") as f:
            f.write(msg + "\n")

    model.eval()
    epoch_val_losses = []
    epoch_val_psnr = []
    
    with torch.no_grad():
        for data in val_loader:
            batch_x_val = data.to(device)
            if batch_x_val.dim() == 2: batch_x_val = batch_x_val.view(-1, 1, 256, 256)
            elif batch_x_val.dim() == 3: batch_x_val = batch_x_val.unsqueeze(1)

            batch_x_val, mean, std = transform.normalize_instance(batch_x_val, eps=1e-11)
            batch_x_val = batch_x_val.clamp(-6, 6)

            cs_ratio_v = torch.tensor([[chosen_cs]], device=device).expand(batch_x_val.shape[0], -1)
            
            x_res, _, _, _, _, _  = model(batch_x_val, cs_ratio_v)
            
            v_loss = torch.nn.functional.l1_loss(x_res, batch_x_val)
            epoch_val_losses.append(v_loss.item())
            
            for b in range(x_res.shape[0]):
                cur_psnr = evaluate.psnr(x_res[b,0].cpu().numpy(), batch_x_val[b,0].cpu().numpy())
                epoch_val_psnr.append(cur_psnr)
    history['train_loss'].append(np.mean(epoch_train_losses))
    history['val_loss'].append(np.mean(epoch_val_losses))
    history['val_psnr'].append(np.mean(epoch_val_psnr))
    
    print(f"Epoch {epoch_i} Summary: Train Loss: {history['train_loss'][-1]:.4f}, "
            f"Val Loss: {history['val_loss'][-1]:.4f}, Val PSNR: {history['val_psnr'][-1]:.2f}")

    torch.save(model.state_dict(), f"{model_dir}/net_params_{epoch_i}.pkl")
