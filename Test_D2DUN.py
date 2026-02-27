from sys import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from time import time
from einops import rearrange
import numbers
import random
import torch_dct as dct
from argparse import ArgumentParser
import platform
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils import evaluate, transform

parser = ArgumentParser(description='D2DUN')

parser.add_argument('--epoch_num', type=int, default=200, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of LUCMT')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0,1', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--data_path', type=str, default='T2', help='Path to the dataset')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='D2-DUN', help='name of test set')

args = parser.parse_args()


batch_size = 1
epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list
test_name = args.test_name
cs_ratio = 0.1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Testing_data_Name = 'fastMRI_test_T1_192.mat'
Testing_data = sio.loadmat('./data/T1/test/%s' % (Testing_data_Name))
Testing_labels = Testing_data['reconstruction_esc']

nrtrain = Testing_labels.shape[0] 
print('number of test is',nrtrain)

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

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Testing_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Testing_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

def calculate_nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

model = D2DUN(layer_num, mode='dual')
model = model.to(device)

model_dir = './%s/MRI_layer_%d_group_%d_%.1f' % (args.model_dir, layer_num, group_num, cs_ratio)
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch_num), map_location=device))

result_root = os.path.join(args.result_dir, test_name)
rec_save_dir = os.path.join(result_root, 'D2-DUN')
if not os.path.exists(rec_save_dir):
    os.makedirs(rec_save_dir)
total_test_images = len(rand_loader.dataset)

def calculate_nmse(gt, pred):
    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2

model.eval()
cs_ratio_value = cs_ratio  

total_test_images = len(rand_loader.dataset)
PSNR_All, SSIM_All, NMSE_All = [], [], []

count = 0
with torch.no_grad():
    for img_no, data in enumerate(rand_loader):
        batch_x = data.to(device)
        if batch_x.dim() == 2: batch_x = batch_x.view(-1, 1, 256, 256)
        elif batch_x.dim() == 3: batch_x = batch_x.unsqueeze(1)
        curr_bs = batch_x.shape[0]
        batch_x_norm, mean_tensor, std_tensor = transform.normalize_instance(batch_x, eps=1e-11)
        batch_x_norm = batch_x_norm.clamp(-6, 6)
        cs_ratio_batch = torch.tensor([[cs_ratio_value]], device=device).float().expand(curr_bs, -1)
        x_output, x_init_output, _, _, _, _, _ = model(batch_x_norm, cs_ratio_batch)
        for b in range(curr_bs):
            if count >= total_test_images: break
            m = mean_tensor[b].cpu().item()
            s = std_tensor[b].cpu().item()

            img_gt = batch_x[b, 0].cpu().numpy().astype(np.float64)
            img_zf = x_init_output[b, 0].cpu().numpy().astype(np.float64) * s + m
            img_rec = x_output[b, 0].cpu().numpy().astype(np.float64) * s + m

            rec_psnr = evaluate.psnr(img_rec, img_gt)
            rec_ssim = evaluate.ssim(img_rec, img_gt)
            rec_nmse = calculate_nmse(img_gt, img_rec)

            PSNR_All.append(rec_psnr)
            SSIM_All.append(rec_ssim)
            NMSE_All.append(rec_nmse)
            print(f"[{count+1:03d}/{total_test_images}] PSNR: {rec_psnr:.2f} | SSIM: {rec_ssim:.4f} | NMSE: {rec_nmse:.5f}")
            
            count += 1

avg_psnr = np.mean(PSNR_All)
avg_ssim = np.mean(SSIM_All)
avg_nmse = np.mean(NMSE_All)

report = (f"\n{'='*40}\n"
          f"测试报告 (采样率: {int(cs_ratio_value)}%)\n"
          f"平均 PSNR: {avg_psnr:.2f} dB\n"
          f"平均 SSIM: {avg_ssim:.4f}\n"
          f"平均 NMSE: {avg_nmse:.5f}\n"
          f"{'='*40}")
print(report)

log_name = f"Result_Log_CS{int(cs_ratio_value)}.txt"
with open(os.path.join(args.log_dir, log_name), 'a') as f:
    f.write(report + "\n")
