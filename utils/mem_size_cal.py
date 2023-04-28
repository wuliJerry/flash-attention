BLOCK_M = 128
BLOCK_N = 128
BLOCK_HEADDIM = 128

batch_size = 32
nheads = 4

Memory_usage = 4 * (3 * BLOCK_M + BLOCK_N + 2 * BLOCK_HEADDIM) + 2 * (2 * BLOCK_M * BLOCK_HEADDIM + BLOCK_M * BLOCK_N + BLOCK_M * BLOCK_HEADDIM) + 4 * (BLOCK_M * BLOCK_N) +  4 * BLOCK_M * BLOCK_N

print(Memory_usage * batch_size * nheads)

'''
Here's the revised memory usage for the tensors:

q, k: 2 bytes (float16) * (BLOCK_M * BLOCK_HEADDIM + BLOCK_N * BLOCK_HEADDIM)
p, acc_o: 2 bytes (float16) * (BLOCK_M * BLOCK_N + BLOCK_M * BLOCK_HEADDIM)
qk, bias: For qk, 4 bytes (float32) * (BLOCK_M * BLOCK_N), and for bias, 4 bytes (float32) * (BLOCK_M * BLOCK_N) if BIAS_TYPE is 'matrix', and 4 bytes (float32) * (1 * BLOCK_N) if BIAS_TYPE is 'vector'.
m_ij, l_ij, lse_i, m_i, l_i_new, acc_o_scale, o_scale: 4 bytes (float32) * (7 * BLOCK_M)
The revised memory usage calculation is:

Memory_usage = 4 * (3 * BLOCK_M + BLOCK_N + 2 * BLOCK_HEADDIM) + 2 * (2 * BLOCK_M * BLOCK_HEADDIM + BLOCK_M * BLOCK_N + BLOCK_M * BLOCK_HEADDIM) + 4 * (BLOCK_M * BLOCK_N)

If BIAS_TYPE is 'matrix', add 4 * BLOCK_M * BLOCK_N to Memory_usage.
If BIAS_TYPE is 'vector', add 4 * BLOCK_N to Memory_usage.

The total memory usage in SRAM depends on the values of BLOCK_M, BLOCK_N, BLOCK_HEADDIM, and BIAS_TYPE.
'''
