import torch
import time
import sys
from typing import Optional, List
# from cg_batch import cg_batch
from cg_batch_new import cg_batch
from cg_block import cg_block
from torch.masked import masked_tensor
from sparse_utils import _get_dtype_eps
# from torchsparsegradutils.utils.linear_cg import linear_cg


@torch.jit.script
def dense_block_diag_apdagd(
    A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
    n_c: torch.Tensor, n_v: torch.Tensor,
    theta: float, y: Optional[torch.Tensor] = None,
    eps: float = 1e-3, max_iter: Optional[int] = None):
    # A: sum(n_c) x sum(n_v)
    # b: sum(n_c)
    # c: sum(n_v)
    # u: sum(n_v)
    # y: sum(n_c)
    # n_c: n_k, where n_c[i] represents number of constraints in instance i
    # n_v: n_k, where n_v[i] represents number of variables in instance i
    n_k = len(n_c)
    n_c_index = torch.repeat_interleave(n_c)
    n_v_index = torch.repeat_interleave(n_v)
    dtype = A.dtype
    device = A.device
    dtype_eps = _get_dtype_eps(dtype)

    theta_u = theta * u
    btb = torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(-1, n_c_index, b ** 2)
    zero = torch.zeros((), dtype=dtype, device=device)
    # one = torch.ones((), dtype=dtype, device=device)
    M = torch.full((n_k, ), fill_value=theta, dtype=dtype, device=device)
    # M = torch.ones((n_k, 1), dtype=dtype, device=device)  # n_k
    beta_old = torch.zeros((n_k, ), dtype=dtype, device=device)  # n_k
    if y is not None:
        eta_old = y.detach().clone().to(dtype=dtype, device=device)  # sum(n_c)
        zeta_old = y.detach().clone().to(dtype=dtype, device=device)  # sum(n_c)
    else:
        eta_old = torch.zeros_like(b, dtype=dtype, device=device)  # sum(n_c)
        zeta_old = torch.zeros_like(b, dtype=dtype, device=device)  # sum(n_c)
    neg_theta_u_s_eta_new = - (c - torch.matmul(eta_old, A)) * theta_u  # sum(n_v)
    x_final_pu = torch.sigmoid(neg_theta_u_s_eta_new)  # sum(n_v)
    x_final = u * x_final_pu
    # z_final_pu = 1. - x_final_pu  # sum(n_v)
    # primal_obj = torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_v_index,
    #     torch.mul(c, x_final) + (torch.xlogy(x_final_pu, x_final_pu) + torch.xlogy(z_final_pu, z_final_pu)) / theta
    # )  # n_k
    # neg_dual_obj = torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_c_index, - torch.mul(b, eta_old)
    # ) + torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_eta_new) / theta
    # )  # n_k
    # gap = torch.abs(primal_obj + neg_dual_obj) / torch.maximum(torch.abs(neg_dual_obj), one)  # n_k
    primal_inf = torch.sqrt(torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        -1, n_c_index, (torch.matmul(A, x_final) - b) ** 2
    ))  # n_k
    # primal_inf_eps = torch.maximum(eps * torch.sqrt(torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_c_index, b ** 2
    # )), torch.tensor(eps, dtype=dtype, device=device))  # n_k
    primal_inf_eps = eps
    # primal_inf = torch.abs(torch.matmul(A, x_final) - b)  # sum(n_c)
    # primal_inf_eps = eps * torch.maximum(torch.abs(b), one)  # sum(n_c)
    last_condition = torch.zeros((n_k, ), dtype=torch.bool, device=device)
    one_int = torch.ones((), dtype=torch.int32, device=device)
    # M_factor = torch.full((), fill_value=2., dtype=dtype, device=device)

    total_it = 0
    # true_it = 0
    while True:
        # alpha_new = (1. + torch.sqrt(1. + 4. * M * beta_old)) / (2. * M)  # M_k * \alpha_{k+1}^2 - \alpha_{k+1} - \beta_k = 0
        alpha_new = 0.5 / M + torch.sqrt((0.25 / M + beta_old) / M)  # M_k * \alpha_{k+1}^2 - \alpha_{k+1} - \beta_k = 0
        beta_new = beta_old + alpha_new  # \beta_{k+1} = \beta_{k} + \alpha_{k+1}
        tau = alpha_new / beta_new  # \tau_{k} = \alpha_{k+1} / \beta_{k+1}
        tau_gather_n_c = torch.gather(tau, -1, n_c_index)

        lambda_new = eta_old + tau_gather_n_c * (zeta_old - eta_old)  # \lambda_{k+1} = \tau_{k} * \zeta_{k} + (1 - \tau_{k}) * \eta_{k}
        neg_theta_u_s_lambda_new = - (c - torch.matmul(lambda_new, A)) * theta_u
        x_lambda_new_pu = torch.sigmoid(neg_theta_u_s_lambda_new)  # calculate x(\lambda_{k+1})
        A_x_lambda_new = torch.matmul(A, u * x_lambda_new_pu)
        grad_phi_lambda_new = A_x_lambda_new - b  # \nabla \phi(\lambda) = Ax(\lambda) - b
        alpha_new_gather_n_c = torch.gather(alpha_new, -1, n_c_index)
        zeta_new = zeta_old - alpha_new_gather_n_c * grad_phi_lambda_new  # \zeta_{k+1} = \zeta_{k} - \alpha_{k+1} \nabla \phi(\lambda)
        eta_new = eta_old + tau_gather_n_c * (zeta_new - eta_old)  # \eta_{k+1} = \tau_{k} \zeta_{k+1} + (1 - \tau_{k}) \eta_{k}
        neg_theta_u_s_eta_new = - (c - torch.matmul(eta_new, A)) * theta_u

        # phi_lambda_new = torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, - torch.mul(b, lambda_new)
        # ) + torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_lambda_new) / theta
        # )  # calculate phi(\lambda_{k+1})
        # phi_eta_new = torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, - torch.mul(b, eta_new)
        # ) + torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_eta_new) / theta
        # )  # calculate phi(\eta_{k+1})
        # condition = (
        #     phi_eta_new - phi_lambda_new - dtype_eps * torch.abs(phi_lambda_new)
        #     <= - (torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #         -1, n_c_index, grad_phi_lambda_new ** 2
        #     )) * 0.5 / M
        # )
        # phi_eta_estimate = phi_lambda_new + (torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, torch.mul(grad_phi_lambda_new, eta_new - lambda_new)
        # )) + M / 2. * (torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, (eta_new - lambda_new) ** 2
        # ))
        # condition = (phi_eta_new <= phi_eta_estimate + dtype_eps)
        # condition = (torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, torch.mul(A_x_lambda_new + b, grad_phi_lambda_new)
        # ) * 0.5 / M + torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_v_index, (torch.logaddexp(zero, neg_theta_u_s_eta_new)
        #                     - torch.logaddexp(zero, neg_theta_u_s_lambda_new)) / theta
        # ) <= dtype_eps)
        condition = ((torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
            -1, n_c_index, A_x_lambda_new ** 2
        ) - btb) * 0.5 / M + torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
            -1, n_v_index, (torch.logaddexp(zero, neg_theta_u_s_eta_new)
                            - torch.logaddexp(zero, neg_theta_u_s_lambda_new)) / theta
        ) <= dtype_eps)

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
            condition_gather_n_c = torch.gather(condition, -1, n_c_index)
            eta_old = torch.where(condition_gather_n_c, eta_new, eta_old)
            zeta_old = torch.where(condition_gather_n_c, zeta_new, zeta_old)
            x_final_pu = torch.where(
                torch.gather(condition, -1, n_v_index),
                x_final_pu + torch.gather(tau, -1, n_v_index) * (x_lambda_new_pu - x_final_pu),
                x_final_pu
            )
            x_final = x_final_pu * u
            # z_final_pu = 1. - x_final_pu  # sum(n_v)
            # primal_obj = torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
            #     -1, n_v_index,
            #     torch.mul(c, x_final) + (torch.xlogy(x_final_pu, x_final_pu) + torch.xlogy(z_final_pu, z_final_pu)) / theta
            # )  # n_k
            # neg_dual_obj = torch.where(
            #     condition_gather_n_c,
            #     torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
            #         -1, n_c_index, - torch.mul(b, eta_new)
            #     ) + torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
            #         -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_eta_new) / theta
            #     ),
            #     neg_dual_obj
            # )
            # gap = torch.abs(primal_obj + neg_dual_obj) / torch.maximum(torch.abs(neg_dual_obj), one)  # n_k
            primal_inf = torch.sqrt(torch.zeros((n_k, ), dtype=dtype, device=device).scatter_add_(
                -1, n_c_index,
                (torch.matmul(A, x_final) - b) ** 2
            ))  # n_k
            # primal_inf = torch.abs(torch.matmul(A, x_final) - b)  # sum(n_c)
            # true_it += 1
        total_it += 1
        # print(total_it, torch.mean(primal_inf).item(), torch.mean(M).item(), torch.min(M).item(),
        #       torch.mean(alpha_new).item(), torch.mean(beta_new).item(), torch.mean(tau).item(), torch.mean(lambda_new).item())
        # print(total_it, primal_inf, primal_inf_eps)

        if torch.all(primal_inf <= primal_inf_eps):
            # print(total_it, torch.mean(primal_inf).item())
            break

        if max_iter is not None and total_it >= max_iter:
            print('Please ensure original problem is feasible or increase max_iter '
                  'or change dtype to float64 when small tolerance or large theta is used.')
            break

    # print(f'it: {it}')
    return x_final, eta_old


class DenseBlockDiagAPDAGDFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
        n_c: torch.Tensor, n_v: torch.Tensor,
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
            x_sol, y_sol = dense_block_diag_apdagd(A, b, c, u, n_c, n_v, theta, y, eps, max_iter)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(A, b, c, u, n_c, n_v, x_sol, y_sol)
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
        # A: sum(n_c) x sum(n_v)
        # b: sum(n_c)
        # c: sum(n_v)
        # u: sum(n_v)
        # dldx: sum(n_v)
        # dldy: sum(n_c)
        if dldx is None and dldy is None:
            return None, None, None, None, None, None, None, None, None, None

        A, b, c, u, n_c, n_v, x_sol, y_sol = ctx.saved_tensors
        theta = ctx.theta
        z_sol = theta * x_sol * (u - x_sol)  # sum(n_v)

        if dldy is None:
            dldx_mul_z = dldx * z_sol
            dldy_total = torch.matmul(A, dldx_mul_z)
        else:
            if dldx is None:
                dldx_mul_z = None
                dldy_total = dldy
            else:
                dldx_mul_z = dldx * z_sol
                dldy_total = torch.matmul(A, dldx_mul_z) + dldy
        # dldf, _ = cg_batch(
        dldf = cg_batch(
            lambda x: (torch.matmul(A, z_sol * torch.matmul(torch.squeeze(x), A))).view(1, -1, 1),
            dldy_total.view(1, -1, 1),
            # verbose=False
        )
        # dldf = linear_cg(
        #     lambda x: (torch.matmul(A, z_sol * torch.matmul(torch.squeeze(x), A))).view(1, -1, 1),
        #     dldy_total.view(1, -1, 1)
        # )
        # dldf = cg_block(
        #     lambda x: torch.unsqueeze(torch.matmul(A, z_sol * torch.matmul(torch.squeeze(x), A)), dim=-1),
        #     torch.unsqueeze(dldy_total, dim=-1),
        #     n_c,
        #     # verbose=True
        # )
        dldf = torch.squeeze(dldf)
        dldf_mul_A = torch.matmul(dldf, A)

        if ctx.needs_input_grad[0]:
            # here, we only calculate gradients of diagonal blocks
            mask = torch.block_diag(*[
                torch.ones((n_c_i, n_v_i), dtype=torch.bool, device=A.device)
                for n_c_i, n_v_i in zip(n_c, n_v)
            ])
            if dldx_mul_z is None:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol = - dldf_mul_A * z_sol
            else:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol = dldx_mul_z - dldf_mul_A * z_sol
            dldA = masked_tensor(torch.unsqueeze(y_sol, dim=-1).expand(-1, A.shape[1]).detach(), mask) * masked_tensor(
                torch.unsqueeze(dldx_mul_z_sub_dldf_mul_A_mul_z_sol, dim=-2).expand(A.shape[0], -1).detach(), mask
            ) - masked_tensor(torch.unsqueeze(dldf, dim=-1).expand(-1, A.shape[1]).detach(), mask) * masked_tensor(
                torch.unsqueeze(x_sol, dim=-2).expand(A.shape[0], -1).detach(), mask
            )
            dldA = dldA.to_tensor(0.)
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
            s_sol = c - torch.matmul(y_sol, A)
            dxdu = (x_sol - z_sol * s_sol) / u
            if dldx is None:
                dldu = - dldf_mul_A * dxdu
            else:
                dldu = (dldx - dldf_mul_A) * dxdu
        else:
            dldu = None

        return dldA, dldb, dldc, dldu, None, None, None, None, None, None


class DenseBlockDiagAPDAGDLayer(torch.nn.Module):
    def __init__(self):
        super(DenseBlockDiagAPDAGDLayer).__init__()

    def forward(
        self,  A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
        n_c: torch.Tensor, n_v: torch.Tensor,
        theta: float, y: Optional[torch.Tensor] = None,
        eps: float = 1e-3, max_iter: Optional[int] = None
    ):
        return DenseBlockDiagAPDAGDFunction.apply(A, b, c, u, n_c, n_v, theta, y, eps, max_iter)

