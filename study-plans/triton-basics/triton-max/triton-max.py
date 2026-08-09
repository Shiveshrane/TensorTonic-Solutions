import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid=tl.program_id(0)
    offset=BLOCK_SIZE*pid+tl.arange(0, BLOCK_SIZE)
    mask=offset<n
    out=tl.load(x_ptr+offset, mask=mask, other=-float('inf'))
    max_out=tl.max(out, axis=0)
    tl.store(out_ptr, max_out)
    
    


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch max_kernel on the provided tensor with a single-program reduction."""
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)