import torch


def length_to_mask(length, N):
    """
    length: B 
    Return: B x N, with 1s in the first length elements
    """
    assert len(length.shape) == 1, length.shape
    B = length.size(0)
    mask = torch.arange(N, device=length.device).unsqueeze(0).repeat([B, 1])
    mask = (mask < length.unsqueeze(1)).long()
    return mask

