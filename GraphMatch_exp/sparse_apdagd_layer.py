import torch
import time
import sys
from typing import List, Optional
# from cg_batch import cg_batch
from cg_batch_new import cg_batch
# from torchsparsegradutils.utils.linear_cg import linear_cg
from sparse_utils import SparseCsrTensor, SparseCsrMatMul, _get_dtype_eps


def sparse_csr_block_diag_from_tensor(
    A_crow_indices: torch.Tensor, A_col_indices: torch.Tensor,
    A_values: torch.Tensor, A_shape: torch.Size
):
    # A_crow_indices: n_c + 1
    # A_col_indices: n_nz
    # A_values: n_b x n_nz
    n_b = A_values.shape[0]
    n_c = A_shape[0]
    n_v = A_shape[1]
    n_nz = A_values.shape[1]

    A_crow_indices = torch.cat((A_crow_indices[0:1], torch.tile(A_crow_indices[1:], (n_b,)) + torch.repeat_interleave(
        torch.arange(0, n_nz * n_b, n_nz, dtype=A_col_indices.dtype, device=A_col_indices.device),
        torch.full((n_b,), fill_value=n_c, dtype=A_crow_indices.dtype, device=A_crow_indices.device)
    )), dim=0)
    A_col_indices = torch.tile(A_col_indices, (n_b,)) + torch.repeat_interleave(
        torch.arange(0, n_v * n_b, n_v, dtype=A_col_indices.dtype, device=A_col_indices.device),
        torch.full((n_b,), fill_value=n_nz, dtype=A_col_indices.dtype, device=A_col_indices.device))
    A_shape = (n_b * n_c, n_b * n_v)

    return SparseCsrTensor.apply(A_crow_indices, A_col_indices, A_values.reshape(-1), A_shape)


@torch.jit.script
def _sparse_apdagd(
    A: torch.Tensor, At: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
    theta: float, y: Optional[torch.Tensor] = None,
    eps: float = 1e-3, max_iter: Optional[int] = None):
    # A: (n_b x n_c) x (n_b x n_v)
    # At: (n_b x n_v) x (n_b x n_c)
    # b: n_b x n_c
    # c: n_b x n_v
    # u: n_b x n_v
    # y: n_b x n_c
    n_b = b.shape[0]
    dtype = A.dtype
    device = A.device
    dtype_eps = _get_dtype_eps(dtype)

    theta_u = theta * u
    btb = torch.sum(b ** 2, dim=-1, keepdim=True)
    zero = torch.zeros((), dtype=dtype, device=device)
    # one = torch.ones((), dtype=dtype, device=device)
    M = torch.full((n_b, 1), fill_value=theta, dtype=dtype, device=device)
    # M = torch.ones((n_b, 1), dtype=dtype, device=device)  # n_b x 1
    # M = theta * torch.linalg.matrix_norm(A, ord=1) ** 2
    beta_old = torch.zeros((n_b, 1), dtype=dtype, device=device)  # n_b x 1
    if y is not None:
        eta_old = y.detach().clone().to(dtype=dtype, device=device)  # n_b x n_c
        zeta_old = y.detach().clone().to(dtype=dtype, device=device)  # n_b x n_c
    else:
        eta_old = torch.zeros_like(b, dtype=dtype, device=device)  # n_b x n_c
        zeta_old = torch.zeros_like(b, dtype=dtype, device=device)  # n_b x n_c
    neg_theta_u_s_eta_new = - (c - SparseCsrMatMul.apply(At, eta_old.reshape(-1, 1), A).reshape(n_b, -1)) * theta_u  # n_b x n_v
    # neg_theta_u_s_eta_new = - (c - torch.matmul(At, eta_old.reshape(-1, 1)).reshape(n_b, -1)) * theta_u  # n_b x n_v
    x_final_pu = torch.sigmoid(neg_theta_u_s_eta_new)  # n_b x n_v
    x_final = u * x_final_pu
    # z_final_pu = 1. - x_final_pu  # n_b x n_v
    # primal_obj = torch.sum(torch.mul(c, x_final), dim=-1, keepdim=True) + (
    #     torch.sum(torch.xlogy(x_final_pu, x_final_pu), dim=-1, keepdim=True)
    #     + torch.sum(torch.xlogy(z_final_pu, z_final_pu), dim=-1, keepdim=True)) / theta  # n_b x 1
    # neg_dual_obj = - torch.sum(torch.mul(b, eta_old), dim=-1, keepdim=True) + torch.sum(
    #     torch.logaddexp(zero, neg_theta_u_s_eta_new), dim=-1, keepdim=True) / theta  # n_b x 1
    # gap = torch.abs(primal_obj + neg_dual_obj) / torch.maximum(torch.abs(neg_dual_obj), one)  # n_b x 1
    primal_inf = torch.linalg.vector_norm(SparseCsrMatMul.apply(
        A, x_final.reshape(-1, 1), At).reshape(n_b, -1) - b, ord=2, dim=-1, keepdim=True)  # n_b x 1
    # primal_inf = torch.linalg.vector_norm(torch.matmul(
    #     A, x_final.reshape(-1, 1)).reshape(n_b, -1) - b, ord=2, dim=-1, keepdim=True)  # n_b x 1
    # primal_inf_eps = torch.maximum(eps * torch.linalg.vector_norm(b, ord=2, dim=-1, keepdim=True),
    #                                torch.tensor(eps, dtype=dtype, device=device))  # n_b x 1
    primal_inf_eps = eps
    # primal_inf = torch.abs(SparseCsrMatMul.apply(A, x_final.reshape(-1, 1), At).reshape(n_b, -1) - b)  # n_b x n_c
    # primal_inf_eps = eps * torch.maximum(torch.abs(b), one)  # n_b x n_c
    last_condition = torch.zeros((n_b, 1), dtype=torch.bool, device=device)
    one_int = torch.ones((), dtype=torch.int32, device=device)
    # M_factor = torch.full((), fill_value=2., dtype=dtype, device=device)

    total_it = 0
    # true_it = 0
    while True:
        # alpha_new = (1. + torch.sqrt(1. + 4. * M * beta_old)) / (2. * M)  # M_k * \alpha_{k+1}^2 - \alpha_{k+1} - \beta_k = 0
        alpha_new = 0.5 / M + torch.sqrt((0.25 / M + beta_old) / M)  # M_k * \alpha_{k+1}^2 - \alpha_{k+1} - \beta_k = 0
        beta_new = beta_old + alpha_new  # \beta_{k+1} = \beta_{k} + \alpha_{k+1}
        tau = alpha_new / beta_new  # \tau_{k} = \alpha_{k+1} / \beta_{k+1}

        lambda_new = eta_old + tau * (zeta_old - eta_old)  # \lambda_{k+1} = \tau_{k} * \zeta_{k} + (1 - \tau_{k}) * \eta_{k}
        neg_theta_u_s_lambda_new = - (c - SparseCsrMatMul.apply(At, lambda_new.reshape(-1, 1), A).reshape(n_b, -1)) * theta_u
        # neg_theta_u_s_lambda_new = - (c - torch.matmul(At, lambda_new.reshape(-1, 1)).reshape(n_b, -1)) * theta_u
        x_lambda_new_pu = torch.sigmoid(neg_theta_u_s_lambda_new)  # calculate x(\lambda_{k+1})
        A_x_lambda_new = SparseCsrMatMul.apply(A, (u * x_lambda_new_pu).reshape(-1, 1), At).reshape(n_b, -1)
        grad_phi_lambda_new = A_x_lambda_new - b  # \nabla \phi(\lambda) = Ax(\lambda) - b
        # grad_phi_lambda_new = torch.matmul(A, (u * x_lambda_new_pu).reshape(-1, 1)).reshape(n_b, -1) - b  # \nabla \phi(\lambda) = Ax(\lambda) - b
        zeta_new = zeta_old - alpha_new * grad_phi_lambda_new  # \zeta_{k+1} = \zeta_{k} - \alpha_{k+1} \nabla \phi(\lambda)
        eta_new = eta_old + tau * (zeta_new - eta_old)  # \eta_{k+1} = \tau_{k} \zeta_{k+1} + (1 - \tau_{k}) \eta_{k}
        neg_theta_u_s_eta_new = - (c - SparseCsrMatMul.apply(At, eta_new.reshape(-1, 1), A).reshape(n_b, -1)) * theta_u
        # neg_theta_u_s_eta_new = - (c - torch.matmul(At, eta_new.reshape(-1, 1)).reshape(n_b, -1)) * theta_u

        # phi_lambda_new = - torch.sum(torch.mul(b, lambda_new), dim=-1, keepdim=True) + torch.sum(
        #     torch.logaddexp(zero, neg_theta_u_s_lambda_new), dim=-1, keepdim=True) / theta  # calculate phi(\lambda_{k+1})
        # phi_eta_new = - torch.sum(torch.mul(b, eta_new), dim=-1, keepdim=True) + torch.sum(
        #     torch.logaddexp(zero, neg_theta_u_s_eta_new), dim=-1, keepdim=True) / theta  # calculate phi(\eta_{k+1})
        # condition = (
        #     phi_eta_new - phi_lambda_new - dtype_eps * torch.abs(phi_lambda_new)
        #     <= - torch.sum(grad_phi_lambda_new ** 2, dim=-1, keepdim=True) * 0.5 / M
        # )
        # phi_eta_estimate = phi_lambda_new + torch.sum(torch.mul(
        #     grad_phi_lambda_new, eta_new - lambda_new), dim=-1, keepdim=True
        # ) + M / 2. * torch.sum((eta_new - lambda_new) ** 2, dim=-1, keepdim=True)
        # condition = (phi_eta_new <= phi_eta_estimate)
        # condition = (torch.sum(torch.mul(A_x_lambda_new + b, grad_phi_lambda_new), dim=-1, keepdim=True) * 0.5 / M
        #              + torch.sum(torch.logaddexp(zero, neg_theta_u_s_eta_new)
        #                          - torch.logaddexp(zero, neg_theta_u_s_lambda_new), dim=-1, keepdim=True) / theta <= dtype_eps)
        condition = ((torch.sum(A_x_lambda_new ** 2, dim=-1, keepdim=True) - btb) * 0.5 / M
                     + torch.sum(torch.logaddexp(zero, neg_theta_u_s_eta_new)
                                 - torch.logaddexp(zero, neg_theta_u_s_lambda_new), dim=-1, keepdim=True) / theta <= dtype_eps)
        # condition = ((torch.sum(A_x_lambda_new.to(torch.float64) ** 2, dim=-1, keepdim=True) - btb.to(torch.float64)) * 0.5 / M.to(torch.float64)
        #              + torch.sum(torch.logaddexp(zero.to(torch.float64), neg_theta_u_s_eta_new.to(torch.float64))
        #                          - torch.logaddexp(zero.to(torch.float64), neg_theta_u_s_lambda_new.to(torch.float64)), dim=-1, keepdim=True
        #                          ) / theta <= dtype_eps)

        M = torch.clamp_min(torch.where(
            condition,
            torch.where(last_condition, torch.ldexp(M, -one_int), M),
            torch.ldexp(M, one_int)
        ), dtype_eps)
        # M = torch.where(condition, torch.where(last_condition, M / M_factor, M), M * M_factor)
        # M = torch.where(condition, M / M_factor, M * M_factor)
        last_condition = condition
        if torch.any(condition):
            beta_old = torch.where(condition, beta_new, beta_old)
            eta_old = torch.where(condition, eta_new, eta_old)
            zeta_old = torch.where(condition, zeta_new, zeta_old)
            x_final_pu = torch.where(condition, x_final_pu + tau * (x_lambda_new_pu - x_final_pu), x_final_pu)
            x_final = x_final_pu * u
            # z_final_pu = 1. - x_final_pu
            # primal_obj = torch.sum(torch.mul(c, x_final), dim=-1, keepdim=True) + (
            #     torch.sum(torch.xlogy(x_final_pu, x_final_pu), dim=-1, keepdim=True)
            #     + torch.sum(torch.xlogy(z_final_pu, z_final_pu), dim=-1, keepdim=True)) / theta
            # neg_dual_obj = torch.where(
            #     condition,
            #     - torch.sum(torch.mul(b, eta_new), dim=-1, keepdim=True) + torch.sum(
            #         torch.logaddexp(zero, neg_theta_u_s_eta_new), dim=-1, keepdim=True) / theta,
            #     neg_dual_obj
            # )
            # gap = torch.abs(primal_obj + neg_dual_obj) / torch.maximum(torch.abs(neg_dual_obj), one)
            primal_inf = torch.linalg.vector_norm(SparseCsrMatMul.apply(
                A, x_final.reshape(-1, 1), At).reshape(n_b, -1) - b, ord=2, dim=-1, keepdim=True)
            # primal_inf = torch.linalg.vector_norm(torch.matmul(
            #     A, x_final.reshape(-1, 1)).reshape(n_b, -1) - b, ord=2, dim=-1, keepdim=True)
            # primal_inf = torch.abs(SparseCsrMatMul.apply(A, x_final.reshape(-1, 1), At).reshape(n_b, -1) - b)
            # true_it += 1
        total_it += 1
        # print(total_it, torch.mean(primal_inf))
        # print(total_it,
        #       condition.item(), phi_eta_new.item(), phi_eta_estimate.item(),
        #       phi_eta_new.item() - phi_eta_estimate.item(),
        #       M.item(), primal_inf.item(), tau.item())

        if torch.all(primal_inf <= primal_inf_eps):
            # print(total_it, torch.mean(primal_inf).item())
            break

        if max_iter is not None and total_it >= max_iter:
            print('Please ensure original problem is feasible or increase max_iter '
                  'or change dtype to float64 when small tolerance or large theta is used.')
            break

    # print(f'it: {it}')
    return x_final, eta_old


@torch.jit.script
def sparse_apdagd(
    A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
    theta: float, y: Optional[torch.Tensor] = None,
    eps: float = 1e-3, max_iter: Optional[int] = None):
    At = A.t().to_sparse_csr()
    x_sol, y_sol = _sparse_apdagd(A, At, b, c, u, theta, y, eps, max_iter)

    return x_sol, y_sol


class SparseAPDAGDFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
        theta: float, y: Optional[torch.Tensor] = None,
        eps: float = 1e-3, max_iter: Optional[int] = None
    ):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        tensors for use in the backward pass using the ctx.save_for_backward method.
        """
        with torch.no_grad():
            At = A.t().to_sparse_csr()
            x_sol, y_sol = _sparse_apdagd(A, At, b, c, u, theta, y, eps, max_iter)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(A, At, b, c, u, x_sol, y_sol)
        ctx.theta = theta
        ctx.n_b = b.shape[0]

        return x_sol, y_sol

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, dldx: torch.Tensor, dldy:torch.Tensor):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output of 'forward', and we need to compute the gradient of the loss
        with respect to the input.

        Here, we use numerator layout in derivative calculation, see https://en.wikipedia.org/wiki/Matrix_calculus

        f = A @ x(y) - b, where x(y) = u * σ(- theta * u * (c - A^T @ y))
        v = A or b or c or u
        ∂y/∂v = - (∂f/∂y)^(-1) @ ∂f/∂v
        ∂l/∂v = ∂l/∂x @ ∂x/∂v + ∂l/∂x @ ∂x/∂y @ ∂y/∂v + ∂l/∂y @ ∂y/∂v
              = ∂l/∂x @ ∂x/∂v + (∂l/∂x @ ∂x/∂y + ∂l/∂y) @ ∂y/∂v
              = ∂l/∂x @ ∂x/∂v + (∂l/∂x @ ∂x/∂y + ∂l/∂y) @ (- (∂f/∂y)^(-1) @ ∂f/∂v)
              = ∂l/∂x @ ∂x/∂v - ((∂l/∂x @ ∂x/∂y + ∂l/∂y) @ (∂f/∂y)^(-1)) @ ∂f/∂v
              = ∂l/∂x @ ∂x/∂v - ∂l/∂f @ ∂f/∂v

        Calculation of ∂l/∂x @ ∂x/∂v:
        ∂l/∂x @ ∂x/∂A = y @ (∂l/∂x * (theta * x * (u - x)))
        ∂l/∂x @ ∂x/∂b = 0
        ∂l/∂x @ ∂x/∂c = - ∂l/∂x * (theta * x * (u - x))
        ∂l/∂x @ ∂x/∂u = ∂l/∂x * ((x - theta * x * (u - x) * s) / u)

        Calculation of ∂l/∂f:
        ∂f/∂y = A @ diag(theta * x * (u - x)) @ A^T (which is a PSD matrix since 0 < x < u)
        ∂x/∂y = diag(theta * x * (u - x)) @ A^T
        ∂l/∂f = (∂l/∂x @ ∂x/∂y + ∂l/∂y) @ (∂f/∂y)^(-1)
        <=> (∂f/∂y)^T @ (∂l/∂f)^T = (∂l/∂x @ ∂x/∂y + ∂l/∂y)^T
        <=> ∂f/∂y @ (∂l/∂f)^T = (∂l/∂x @ ∂x/∂y + ∂l/∂y)^T
        <=> ∂f/∂y @ (∂l/∂f)^T = ((∂l/∂x * (theta * x * (u - x))) @ A^T + ∂l/∂y)^T
        <=> ∂f/∂y @ (∂l/∂f)^T = A @ ((theta * x * (u - x)) * (∂l/∂x)^T) + (∂l/∂y)^T

        Calculation of ∂l/∂f @ ∂f/∂v:
        ∂l/∂f @ ∂f/∂A = (∂l/∂f)^T @ x^T + y @ ((∂l/∂f @ A) * (theta * x * (u - x)))
        ∂l/∂f @ ∂f/∂b = - ∂l/∂f
        ∂l/∂f @ ∂f/∂c = ∂l/∂f @ (- A @ diag(theta * x * (u - x))) = - (∂l/∂f @ A) * (theta * x * (u - x))
        ∂l/∂f @ ∂f/∂u = ∂l/∂f @ (A @ diag((x - theta * x * (u - x) * s) / u))
                      = (∂l/∂f @ A) * ((x - theta * x * (u - x) * s) / u))
        """
        # A: (n_b x n_c) x (n_b x n_v)
        # b: n_b x n_c
        # c: n_b x n_v
        # x_sol: n_b x n_v
        # y_sol: n_b x n_c
        # dldx: n_b x n_v
        # dldy: n_b x n_c
        if dldx is None and dldy is None:
            return None, None, None, None, None, None, None, None

        A, At, b, c, u, x_sol, y_sol = ctx.saved_tensors
        theta = ctx.theta
        n_b = ctx.n_b
        z_sol = theta * x_sol * (u - x_sol)  # n_b x n_v

        if dldy is None:
            dldx_mul_z = dldx * z_sol
            dldy_total = torch.matmul(A, dldx_mul_z.reshape(-1, 1)).reshape(n_b, -1)
        else:
            if dldx is None:
                dldx_mul_z = None
                dldy_total = dldy
            else:
                dldx_mul_z = dldx * z_sol
                dldy_total = torch.matmul(A, dldx_mul_z.reshape(-1, 1)).reshape(n_b, -1) + dldy
        # dldf, _ = cg_batch(
        dldf = cg_batch(
            lambda x: torch.matmul(
                A, (z_sol * torch.matmul(At, x.reshape(-1, 1)).reshape(n_b, -1)).reshape(-1, 1)
            ).reshape(n_b, -1, 1),
            torch.unsqueeze(dldy_total, dim=-1),
            # verbose=True
        )
        # dldf = linear_cg(
        #     lambda x: torch.matmul(
        #         A, (z_sol * torch.matmul(At, x.reshape(-1, 1)).reshape(n_b, -1)).reshape(-1, 1)
        #     ).reshape(n_b, -1, 1),
        #     torch.unsqueeze(dldy_total, dim=-1)
        # )
        dldf = torch.squeeze(dldf, dim=-1)
        dldf_mul_A = torch.matmul(At, dldf.reshape(-1, 1)).reshape(n_b, -1)

        if ctx.needs_input_grad[0]:
            A_crow_indices = A.crow_indices()
            A_col_indices = A.col_indices()
            A_row_indices = torch.repeat_interleave(
                torch.arange(y_sol.numel()), A_crow_indices[1:] - A_crow_indices[:-1])

            y_sol_select_row = torch.index_select(y_sol.reshape(-1), 0, A_row_indices)
            if dldx_mul_z is None:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol_select_col = torch.index_select(
                    (- dldf_mul_A * z_sol).reshape(-1), 0, A_col_indices)
            else:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol_select_col = torch.index_select(
                    (dldx_mul_z - dldf_mul_A * z_sol).reshape(-1), 0, A_col_indices)
            dldf_select_row = torch.index_select(dldf.reshape(-1), 0, A_row_indices)
            x_sol_select_col = torch.index_select(x_sol.reshape(-1), 0, A_col_indices)

            # dldx_mul_dxdA = y_sol_select_row * dldx_mul_z_select_col
            # dldf_mul_dfdA = dldf_select_row * x_sol_select_col + y_sol_select_row * dldf_mul_A_mul_z_sol_select_col
            # dldA = (dldx_mul_dxdA - dldf_mul_dfdA).reshape(n_b, -1)
            dldA = torch.sparse_csr_tensor(
                A_crow_indices, A_col_indices,
                y_sol_select_row * dldx_mul_z_sub_dldf_mul_A_mul_z_sol_select_col
                - dldf_select_row * x_sol_select_col, A.shape
            )
        else:
            dldA = None

        if ctx.needs_input_grad[1]:
            dldb = dldf
        else:
            dldb = None

        if ctx.needs_input_grad[2]:
            if dldx_mul_z is None:
                dldc = dldf_mul_A * z_sol
            else:
                dldc = - dldx_mul_z + dldf_mul_A * z_sol
        else:
            dldc = None

        if ctx.needs_input_grad[3]:
            s_sol = c - torch.matmul(At, y_sol.reshape(-1, 1)).reshape(n_b, -1)
            dxdu = (x_sol - z_sol * s_sol) / u
            if dldx is None:
                dldu = - dldf_mul_A * dxdu
            else:
                dldu = (dldx - dldf_mul_A) * dxdu
        else:
            dldu = None

        return dldA, dldb, dldc, dldu, None, None, None, None


class SparseAPDAGDLayer(torch.nn.Module):
    def __init__(self):
        super(SparseAPDAGDLayer).__init__()

    def forward(
        self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
        theta: float, y: Optional[torch.Tensor] = None, eps: float = 1e-3, max_iter: Optional[int] = None
    ):
        return SparseAPDAGDFunction.apply(A, b, c, u, theta, y, eps, max_iter)

