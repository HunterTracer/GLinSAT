import math
import torch
from typing import Optional


# def _get_dtype_eps(
#     dtype: torch.dtype,
#     eps32: float = 1e-6,
#     eps64: float = 1e-10,
# ) -> float:
#     if dtype == torch.float32:
#         return eps32
#     elif dtype == torch.float64:
#         return eps64
#     else:
#         raise RuntimeError(f"Expected x to be floating-point, got {dtype}")
def _get_dtype_eps(
    dtype: torch.dtype,
    eps32: float = torch.finfo(torch.float32).eps * 10.,
    eps64: float = torch.finfo(torch.float64).eps * 10.,
) -> float:
    if dtype == torch.float32:
        return eps32
    elif dtype == torch.float64:
        return eps64
    else:
        raise RuntimeError(f"Expected x to be floating-point, got {dtype}")


class SparseCsrTensor(torch.autograd.Function):
    """
    This implementation provides a version of torch.sparse_csr_tensor on the backward pass of nonzero values
    """
    @staticmethod
    def forward(ctx, crow_indices, col_indices, values, size):
        ctx.set_materialize_grads(False)
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, size)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None, None, None
        return None, None, grad.values(), None


class SparseCsrMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, At: Optional[torch.Tensor] = None):
        # At is used only for backward
        x = torch.matmul(A, B)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(A, B, At)

        x.requires_grad = A.requires_grad or B.requires_grad
        return x

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor):
        if grad is None:
            return None, None, None

        A, B, At = ctx.saved_tensors

        # The gradient with respect to the matrix A, seen as a dense matrix, would
        # lead to a backprop rule as follows: gradA = grad @ b.T
        # but we are only interested in the gradient with respect to
        # the (non-zero) values of A. To save memory, instead of computing the full
        # dense matrix prev_grad @ b and then subsampling at the nnz locations in A,
        # we can directly only compute the required values:
        # grad_a[i,j] = dotprod(grad[i,:], b[j,:])

        # We start by getting the i and j indices:

        if ctx.needs_input_grad[0]:
            A_col_idx = A.col_indices()
            A_crow_idx = A.crow_indices()
            # Uncompress row indices:
            A_row_idx = torch.repeat_interleave(
                torch.arange(A.size()[0], device=A.device), A_crow_idx[1:] - A_crow_idx[:-1]
            )

            grad_select = grad.index_select(0, A_row_idx)  # grad[i, :]
            B_select = B.index_select(0, A_col_idx)  # B[j, :]

            # Dot product:
            gradB_ewise = grad_select * B_select
            gradA = torch.sum(gradB_ewise, dim=1)
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)
        else:
            gradA = None

        # Now compute the dense gradient with respect to B
        if ctx.needs_input_grad[1]:
            if At is not None:
                gradB = torch.matmul(At, grad)
            else:
                gradB = torch.matmul(A.t(), grad)
        else:
            gradB = None

        return gradA, gradB, None
