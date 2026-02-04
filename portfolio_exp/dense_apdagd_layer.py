import torch
import time
import sys
from typing import Optional
# from cg_batch import cg_batch
from cg_batch_new import cg_batch
from sparse_utils import _get_dtype_eps
# from torchsparsegradutils.utils.linear_cg import linear_cg


@torch.jit.script
def dense_apdagd(
    A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
    theta: float, y: Optional[torch.Tensor] = None,
    eps: float = 1e-3, max_iter: Optional[int] = None):
    # A: n_b x n_c x n_v
    # b: n_b x n_c
    # c: n_b x n_v
    # u: n_b x n_v
    # y: n_b x n_c
    n_b = A.shape[0]
    dtype = A.dtype
    device = A.device
    dtype_eps = _get_dtype_eps(dtype)

    theta_u = theta * u
    btb = torch.sum(b ** 2, dim=-1, keepdim=True)
    zero = torch.zeros((), dtype=dtype, device=device)
    M = torch.full((n_b, 1), fill_value=theta, dtype=dtype, device=device)
    beta_old = torch.zeros((n_b, 1), dtype=dtype, device=device)  # n_b x 1
    if y is not None:
        eta_old = y.detach().clone().to(dtype=dtype, device=device)  # n_b x n_c
        zeta_old = y.detach().clone().to(dtype=dtype, device=device)  # n_b x n_c
    else:
        eta_old = torch.zeros_like(b, dtype=dtype, device=device)  # n_b x n_c
        zeta_old = torch.zeros_like(b, dtype=dtype, device=device)  # n_b x n_c
    neg_theta_u_s_eta_new = - (c - torch.squeeze(torch.bmm(torch.unsqueeze(eta_old, dim=-2), A), dim=-2)) * theta_u  # n_b x n_v
    x_final_pu = torch.sigmoid(neg_theta_u_s_eta_new)  # n_b x n_v
    x_final = u * x_final_pu
    primal_inf = torch.linalg.vector_norm(torch.squeeze(
        torch.bmm(A, torch.unsqueeze(x_final, dim=-1)), dim=-1) - b, ord=2, dim=-1, keepdim=True)  # n_b x 1
    primal_inf_eps = eps
    last_condition = torch.zeros((n_b, 1), dtype=torch.bool, device=device)
    one_int = torch.ones((), dtype=torch.int32, device=device)

    total_it = 0
    while True:
        alpha_new = 0.5 / M + torch.sqrt((0.25 / M + beta_old) / M)  # M_k * \alpha_{k+1}^2 - \alpha_{k+1} - \beta_k = 0
        beta_new = beta_old + alpha_new  # \beta_{k+1} = \beta_{k} + \alpha_{k+1}
        tau = alpha_new / beta_new  # \tau_{k} = \alpha_{k+1} / \beta_{k+1}

        lambda_new = eta_old + tau * (zeta_old - eta_old)  # \lambda_{k+1} = \tau_{k} * \zeta_{k} + (1 - \tau_{k}) * \eta_{k}
        neg_theta_u_s_lambda_new = - (c - torch.squeeze(torch.bmm(torch.unsqueeze(lambda_new, dim=-2), A), dim=-2)) * theta_u
        x_lambda_new_pu = torch.sigmoid(neg_theta_u_s_lambda_new)  # calculate x(\lambda_{k+1})
        A_x_lambda_new = torch.squeeze(torch.bmm(A, torch.unsqueeze(u * x_lambda_new_pu, dim=-1)), dim=-1)
        grad_phi_lambda_new = A_x_lambda_new - b  # \nabla \phi(\lambda) = Ax(\lambda) - b
        zeta_new = zeta_old - alpha_new * grad_phi_lambda_new  # \zeta_{k+1} = \zeta_{k} - \alpha_{k+1} \nabla \phi(\lambda)
        eta_new = eta_old + tau * (zeta_new - eta_old)  # \eta_{k+1} = \tau_{k} \zeta_{k+1} + (1 - \tau_{k}) \eta_{k}
        neg_theta_u_s_eta_new = - (c - torch.squeeze(torch.bmm(torch.unsqueeze(eta_new, dim=-2), A), dim=-2)) * theta_u

        condition = ((torch.sum(A_x_lambda_new ** 2, dim=-1, keepdim=True) - btb) * 0.5 / M
                     + torch.sum(torch.logaddexp(zero, neg_theta_u_s_eta_new)
                                 - torch.logaddexp(zero, neg_theta_u_s_lambda_new), dim=-1, keepdim=True) / theta <= dtype_eps)

        M = torch.clamp_min(torch.where(
            condition,
            torch.where(last_condition, torch.ldexp(M, -one_int), M),
            torch.ldexp(M, one_int)
        ), dtype_eps)
        last_condition = condition
        if torch.any(condition):
            beta_old = torch.where(condition, beta_new, beta_old)
            eta_old = torch.where(condition, eta_new, eta_old)
            zeta_old = torch.where(condition, zeta_new, zeta_old)
            x_final_pu = torch.where(
                condition, x_final_pu + tau * (x_lambda_new_pu - x_final_pu), x_final_pu)
            x_final = x_final_pu * u
            primal_inf = torch.linalg.vector_norm(torch.squeeze(
                torch.bmm(A, torch.unsqueeze(x_final, dim=-1)), dim=-1) - b, ord=2, dim=-1, keepdim=True)
        total_it += 1

        if torch.all(primal_inf <= primal_inf_eps):
            break

        if max_iter is not None and total_it >= max_iter:
            print('Please ensure original problem is feasible or increase max_iter '
                  'or change dtype to float64 when small tolerance or large theta is used.')
            break

    return x_final, eta_old


class DenseAPDAGDFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
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
            x_sol, y_sol = dense_apdagd(A, b, c, u, theta, y, eps, max_iter)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(A, b, c, u, x_sol, y_sol)
        ctx.theta = theta
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
        # A: n_b x n_c x n_v
        # b: n_b x n_c
        # c: n_b x n_v
        # x_sol: n_b x n_v
        # y_sol: n_b x n_c
        # dldx: n_b x n_v
        # dldy: n_b x n_c
        if dldx is None and dldy is None:
            return None, None, None, None, None, None, None, None

        A, b, c, u, x_sol, y_sol = ctx.saved_tensors
        theta = ctx.theta
        z_sol = theta * x_sol * (u - x_sol)  # n_b x n_v

        if dldy is None:
            dldx_mul_z = dldx * z_sol
            dldy_total = torch.squeeze(torch.bmm(A, torch.unsqueeze(dldx_mul_z, dim=-1)), dim=-1)
        else:
            if dldx is None:
                dldx_mul_z = None
                dldy_total = dldy
            else:
                dldx_mul_z = dldx * z_sol
                dldy_total = torch.squeeze(torch.bmm(A, torch.unsqueeze(dldx_mul_z, dim=-1)), dim=-1) + dldy
        dldf = cg_batch(
            lambda x: torch.bmm(A, torch.unsqueeze(z_sol * torch.squeeze(
                torch.bmm(torch.unsqueeze(torch.squeeze(x, dim=-1), dim=-2), A), dim=-2), dim=-1)),
            torch.unsqueeze(dldy_total, dim=-1),
        )
        dldf = torch.squeeze(dldf, dim=-1)
        dldf_mul_A = torch.squeeze(torch.bmm(torch.unsqueeze(dldf, dim=-2), A), dim=-2)

        if ctx.needs_input_grad[0]:
            if dldx_mul_z is None:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol = - dldf_mul_A * z_sol
            else:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol = dldx_mul_z - dldf_mul_A * z_sol
            dldA = torch.unsqueeze(y_sol, dim=-1) * torch.unsqueeze(dldx_mul_z_sub_dldf_mul_A_mul_z_sol, dim=-2) \
                   - torch.unsqueeze(dldf, dim=-1) * torch.unsqueeze(x_sol, dim=-2)
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
            s_sol = c - torch.squeeze(torch.bmm(torch.unsqueeze(y_sol, dim=-2), A), dim=-2)
            dxdu = (x_sol - z_sol * s_sol) / u
            if dldx is None:
                dldu = - dldf_mul_A * dxdu
            else:
                dldu = (dldx - dldf_mul_A) * dxdu
        else:
            dldu = None

        return dldA, dldb, dldc, dldu, None, None, None, None


class DenseAPDAGDLayer(torch.nn.Module):
    def __init__(self):
        super(DenseAPDAGDLayer).__init__()

    def forward(self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor, theta: float,
                y: Optional[torch.Tensor] = None, eps: float = 1e-3, max_iter: Optional[int] = None):
        return DenseAPDAGDFunction.apply(A, b, c, u, theta, y, eps, max_iter)
