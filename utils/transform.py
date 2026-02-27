def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


# def normalize_instance(data, eps=0.):
#     """
#         Normalize the given tensor using:
#             (data - mean) / (stddev + eps)
#         where mean and stddev are computed from the data itself.

#         Args:
#             data (torch.Tensor): Input data to be normalized
#             eps (float): Added to stddev to prevent dividing by zero

#         Returns:
#             torch.Tensor: Normalized tensor
#         """
#     mean = data.mean()
#     std = data.std()
#     return normalize(data, mean, std, eps), mean, std

def normalize_instance(data, eps=1e-11):
    # data: (B, C, H, W)
    mean = data.mean(dim=(-2, -1), keepdim=True)
    std = data.std(dim=(-2, -1), keepdim=True)
    return (data - mean) / (std + eps), mean, std

def unnormalize_instance(normalized_data, mean, std, eps=1e-11):
    # normalized_data: (B, C, H, W)
    # mean, std: (B, C, 1, 1)
    return normalized_data * (std + eps) + mean