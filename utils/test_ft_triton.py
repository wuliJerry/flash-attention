from functools import partial
import math
import torch
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.flash_attn_triton import flash_attn_func

def test_flash_attn_triton_output(seqlen_q, seqlen_k, d, causal, dtype, bias_shape):
    # if seqlen_q >= 2048 and torch.cuda.get_device_properties('cuda').total_memory <= 16 * 2**30:
    #     pytest.skip()  # Reference implementation OOM
    device = 'cuda'
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k, v = torch.randn(batch_size, seqlen_k, 2, nheads, d, device=device, dtype=dtype).unbind(dim=2)
    if bias_shape == '1h1k':
        bias = torch.randn(1, nheads, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == '1hqk':
        bias = torch.randn(1, nheads, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == 'b11k':
        bias = torch.randn(batch_size, 1, 1, seqlen_k, dtype=torch.float, device=device)
    elif bias_shape == 'b1qk':
        bias = torch.randn(batch_size, 1, seqlen_q, seqlen_k, dtype=torch.float, device=device)
    else:
        bias = None

    q, k, v = [x.detach().requires_grad_() for x in [q, k, v]]
    output = flash_attn_func(q, k, v, bias, causal)

test_flash_attn_triton_output(
    seqlen_q=2048, 
    seqlen_k=2048, 
    d=128, 
    causal=False, 
    dtype=torch.float16, 
    bias_shape='1hqk'
)