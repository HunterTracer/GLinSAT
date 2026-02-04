import torch
import time
import sys
from typing import Optional, Union, Tuple, List
# from cg_batch import cg_batch
from cg_batch_new import cg_batch
from cg_block import cg_block
# from torchsparsegradutils.utils.linear_cg import linear_cg
from sparse_utils import SparseCsrTensor, SparseCsrMatMul, _get_dtype_eps


def sparse_csr_block_diag_from_tuple_list(
    A_crow_indices: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    A_col_indices: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    A_values: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    A_shape: Union[Tuple[torch.Size, ...], List[torch.Size]],
):
    # A_crow_indices[i]: n_c[i] + 1
    # A_col_indices[i]: n_nz[i]
    # A_values[i]: n_nz[i]
    # A_shape[i]: 2
    n_c_sum = 0
    n_v_sum = 0
    n_nz_sum = 0
    crow_indices_list = []
    col_indices_list = []
    for i, (crow_indices, col_indices, values, shape) in enumerate(zip(
        A_crow_indices, A_col_indices, A_values, A_shape
    )):
        if i == 0:
            crow_indices_list.append(crow_indices)
            col_indices_list.append(col_indices)
        else:
            crow_indices_list.append(crow_indices[1:] + n_nz_sum)
            col_indices_list.append(col_indices + n_v_sum)
        n_c_sum += shape[0]
        n_v_sum += shape[1]
        n_nz_sum += len(values)

    return SparseCsrTensor.apply(torch.cat(crow_indices_list, dim=0), torch.cat(col_indices_list, dim=0),
                                 torch.cat(A_values, dim=0), (n_c_sum, n_v_sum))


@torch.jit.script
def _sparse_block_diag_apdagd(
    A: torch.Tensor, At: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
    n_c: torch.Tensor, n_v: torch.Tensor,
    theta: float, y: Optional[torch.Tensor] = None,
    eps: float = 1e-3, max_iter: Optional[int] = None):
    # A: sum(n_c) x sum(n_v)
    # At: sum(n_v) x sum(n_c)
    # b: sum(n_c)
    # c: sum(n_v)
    # u: sum(n_v)
    # y: sum(n_c)
    n_b = len(n_c)
    n_c_index = torch.repeat_interleave(n_c)
    n_v_index = torch.repeat_interleave(n_v)
    dtype = A.dtype
    device = A.device
    dtype_eps = _get_dtype_eps(dtype)

    theta_u = theta * u
    btb = torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(-1, n_c_index, b ** 2)
    zero = torch.zeros((), dtype=dtype, device=device)
    # one = torch.ones((), dtype=dtype, device=device)
    M = torch.full((n_b, ), fill_value=theta, dtype=dtype, device=device)
    # M = torch.ones((n_b, ), dtype=dtype, device=device)  # n_b
    beta_old = torch.zeros((n_b, ), dtype=dtype, device=device)  # n_b
    if y is not None:
        eta_old = y.detach().clone().to(dtype=dtype, device=device)  # sum(n_c)
        zeta_old = y.detach().clone().to(dtype=dtype, device=device)  # sum(n_c)
    else:
        eta_old = torch.zeros_like(b, dtype=dtype, device=device)  # sum(n_c)
        zeta_old = torch.zeros_like(b, dtype=dtype, device=device)  # sum(n_c)
    neg_theta_u_s_eta_new = - (
        c - torch.squeeze(SparseCsrMatMul.apply(At, torch.unsqueeze(eta_old, dim=-1), A), dim=-1)) * theta_u  # sum(n_v)
    # neg_theta_u_s_eta_new = - (
    #     c - torch.squeeze(torch.matmul(At, torch.unsqueeze(eta_old, dim=-1)), dim=-1)) * theta_u  # sum(n_v)
    x_final_pu = torch.sigmoid(neg_theta_u_s_eta_new)  # sum(n_v)
    x_final = u * x_final_pu  # sum(n_v)
    # z_final_pu = 1. - x_final_pu  # sum(n_v)
    # primal_obj = torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_v_index,
    #     torch.mul(c, x_final) + (torch.xlogy(x_final_pu, x_final_pu) + torch.xlogy(z_final_pu, z_final_pu)) / theta
    # )  # n_b
    # neg_dual_obj = torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_c_index, - torch.mul(b, eta_old)
    # ) + torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
    #     -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_eta_new) / theta
    # )  # n_b
    # gap = torch.abs(primal_obj + neg_dual_obj) / torch.maximum(torch.abs(neg_dual_obj), one)  # n_b
    primal_inf = torch.sqrt(torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        -1, n_c_index,
        (torch.squeeze(SparseCsrMatMul.apply(A, torch.unsqueeze(x_final, dim=-1), At), dim=-1) - b) ** 2
    ))  # n_b
    # primal_inf = torch.sqrt(torch.zeros((n_b,), dtype=dtype, device=device).scatter_add_(
    #     -1, n_c_index,
    #     (torch.squeeze(torch.matmul(A, torch.unsqueeze(x_final, dim=-1)), dim=-1) - b) ** 2
    # ))  # n_b
    # primal_inf_eps = torch.maximum(eps * torch.sqrt(torch.zeros((n_b,), dtype=dtype, device=device).scatter_add_(
    #     -1, n_c_index, b ** 2
    # )), torch.tensor(eps, dtype=dtype, device=device))  # n_b
    primal_inf_eps = eps
    # primal_inf = torch.abs(torch.matmul(A, x_final) - b)  # sum(n_c)
    # primal_inf_eps = eps * torch.maximum(torch.abs(b), one)  # sum(n_c)
    last_condition = torch.zeros((n_b, ), dtype=torch.bool, device=device)
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
        neg_theta_u_s_lambda_new = - (
            c - torch.squeeze(SparseCsrMatMul.apply(At, torch.unsqueeze(lambda_new, dim=-1), A), dim=-1)) * theta_u  # sum(n_v)
        # neg_theta_u_s_lambda_new = - (
        #     c - torch.squeeze(torch.matmul(At, torch.unsqueeze(lambda_new, dim=-1)), dim=-1)) * theta_u  # sum(n_v)
        x_lambda_new_pu = torch.sigmoid(neg_theta_u_s_lambda_new)  # calculate x(\lambda_{k+1})
        A_x_lambda_new = torch.squeeze(
            SparseCsrMatMul.apply(A, torch.unsqueeze(u * x_lambda_new_pu, dim=-1), At), dim=-1)
        grad_phi_lambda_new = A_x_lambda_new - b  # \nabla \phi(\lambda) = Ax(\lambda) - b
        # grad_phi_lambda_new = torch.squeeze(
        #     torch.matmul(A, torch.unsqueeze(u * x_lambda_new_pu, dim=-1)), dim=-1) - b  # \nabla \phi(\lambda) = Ax(\lambda) - b
        alpha_new_gather_n_c = torch.gather(alpha_new, -1, n_c_index)
        zeta_new = zeta_old - alpha_new_gather_n_c * grad_phi_lambda_new  # \zeta_{k+1} = \zeta_{k} - \alpha_{k+1} \nabla \phi(\lambda)
        eta_new = eta_old + tau_gather_n_c * (zeta_new - eta_old)  # \eta_{k+1} = \tau_{k} \zeta_{k+1} + (1 - \tau_{k}) \eta_{k}
        neg_theta_u_s_eta_new = - (
            c - torch.squeeze(SparseCsrMatMul.apply(At, torch.unsqueeze(eta_new, dim=-1), A), dim=-1)) * theta_u
        # neg_theta_u_s_eta_new = - (
        #     c - torch.squeeze(torch.matmul(At, torch.unsqueeze(eta_new, dim=-1)), dim=-1)) * theta_u

        # phi_lambda_new = torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, - torch.mul(b, lambda_new)
        # ) + torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_lambda_new) / theta
        # )  # calculate phi(\lambda_{k+1})
        # phi_eta_new = torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, - torch.mul(b, eta_new)
        # ) + torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_eta_new) / theta
        # )  # calculate phi(\eta_{k+1})
        # condition = (
        #     phi_eta_new - phi_lambda_new - dtype_eps * torch.abs(phi_lambda_new)
        #     <= - (torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #         -1, n_c_index, grad_phi_lambda_new ** 2
        #     )) * 0.5 / M
        # )
        # print('phi_lambda_new', torch.abs(phi_lambda_new).reshape(1, -1))
        # phi_eta_estimate = phi_lambda_new + (torch.zeros((n_b,), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, torch.mul(grad_phi_lambda_new, eta_new - lambda_new)
        # )) + M / 2. * (torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, (eta_new - lambda_new) ** 2
        # ))
        # condition = (phi_eta_new <= phi_eta_estimate)
        # condition = (torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_c_index, torch.mul(A_x_lambda_new + b, grad_phi_lambda_new)
        # ) * 0.5 / M + torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
        #     -1, n_v_index, (torch.logaddexp(zero, neg_theta_u_s_eta_new)
        #                     - torch.logaddexp(zero, neg_theta_u_s_lambda_new)) / theta
        # ) <= dtype_eps)
        condition = ((torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
            -1, n_c_index, A_x_lambda_new ** 2
        ) - btb) * 0.5 / M + torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
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
            # primal_obj = torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
            #     -1, n_v_index,
            #     torch.mul(c, x_final) + (torch.xlogy(x_final_pu, x_final_pu) + torch.xlogy(z_final_pu, z_final_pu)) / theta
            # )  # n_b
            # neg_dual_obj = torch.where(
            #     condition_gather_n_c,
            #     torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
            #         -1, n_c_index, - torch.mul(b, eta_new)
            #     ) + torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
            #         -1, n_v_index, torch.logaddexp(zero, neg_theta_u_s_eta_new) / theta
            #     ),
            #     neg_dual_obj
            # )
            # gap = torch.abs(primal_obj + neg_dual_obj) / torch.maximum(torch.abs(neg_dual_obj), one)  # n_b
            primal_inf = torch.sqrt(torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
                -1, n_c_index,
                (torch.squeeze(SparseCsrMatMul.apply(A, torch.unsqueeze(x_final, dim=-1), At), dim=-1) - b) ** 2
            ))  # n_b
            # primal_inf = torch.sqrt(torch.zeros((n_b, ), dtype=dtype, device=device).scatter_add_(
            #     -1, n_c_index,
            #     (torch.squeeze(torch.matmul(A, torch.unsqueeze(x_final, dim=-1)), dim=-1) - b) ** 2
            # ))  # n_b
            # primal_inf = torch.abs(torch.matmul(A, x_final) - b)
            # true_it += 1
        total_it += 1
        # print(total_it, torch.mean(primal_inf).item(),
        #       torch.mean(M).item(), torch.min(M).item(), torch.max(M).item(),
        #       torch.mean(alpha_new).item(), torch.min(alpha_new).item(), torch.max(alpha_new).item(),
        #       torch.mean(beta_new).item(), torch.min(beta_new).item(), torch.max(beta_new).item(),
        #       torch.mean(tau).item(), torch.min(tau).item(), torch.max(tau).item(),
        #       torch.mean(x_final_pu).item(), torch.min(x_final_pu).item(), torch.max(x_final_pu).item())
        # print(total_it, primal_inf.reshape(1, -1), M.reshape(1, -1))
        # print(total_it, torch.mean(primal_inf).item(), torch.mean(M).item())
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
def sparse_block_diag_apdagd(
    A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
    n_c: torch.Tensor, n_v: torch.Tensor,
    theta: float, y: Optional[torch.Tensor] = None,
    eps: float = 1e-3, max_iter: Optional[int] = None):
    At = A.t().to_sparse_csr()
    x_sol, y_sol = _sparse_block_diag_apdagd(A, At, b, c, u, n_c, n_v, theta, y, eps, max_iter)

    return x_sol, y_sol


class SparseBlockDiagAPDAGDFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
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
            At = A.t().to_sparse_csr()
            x_sol, y_sol = _sparse_block_diag_apdagd(A, At, b, c, u, n_c, n_v, theta, y, eps, max_iter)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(A, At, b, c, u, n_c, n_v, x_sol, y_sol)
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

        A, At, b, c, u, n_c, n_v, x_sol, y_sol = ctx.saved_tensors
        theta = ctx.theta
        z_sol = theta * x_sol * (u - x_sol)  # sum(n_v)

        if dldy is None:
            dldx_mul_z = dldx * z_sol
            dldy_total = torch.squeeze(torch.matmul(A, torch.unsqueeze(dldx_mul_z, dim=-1)), dim=-1)
        else:
            if dldx is None:
                dldx_mul_z = None
                dldy_total = dldy
            else:
                dldx_mul_z = dldx * z_sol
                dldy_total = torch.squeeze(torch.matmul(A, torch.unsqueeze(dldx_mul_z, dim=-1)), dim=-1) + dldy
        # dldf, _ = cg_batch(
        dldf = cg_batch(
            lambda x: torch.unsqueeze(torch.matmul(
                A, torch.unsqueeze(z_sol, dim=-1) * torch.matmul(At, torch.squeeze(x, dim=0))
            ), dim=0),
            dldy_total.view(1, -1, 1),
            # verbose=True
        )
        # dldf = linear_cg(
        #     lambda x: torch.unsqueeze(torch.matmul(
        #         A, torch.unsqueeze(z_sol, dim=-1) * torch.matmul(At, torch.squeeze(x, dim=0))
        #     ), dim=0),
        #     dldy_total.view(1, -1, 1)
        # )
        # dldf = cg_block(
        #     lambda x: torch.matmul(A, torch.unsqueeze(z_sol, dim=-1) * torch.matmul(At, x)),
        #     torch.unsqueeze(dldy_total, dim=-1),
        #     n_c,
        #     verbose=True
        # )
        dldf = torch.squeeze(dldf)
        dldf_mul_A = torch.squeeze(torch.matmul(At, torch.unsqueeze(dldf, dim=-1)), dim=-1)

        if ctx.needs_input_grad[0]:
            A_crow_indices = A.crow_indices()
            A_col_indices = A.col_indices()
            A_row_indices = torch.repeat_interleave(
                torch.arange(y_sol.numel()), A_crow_indices[1:] - A_crow_indices[:-1])

            y_sol_select_row = torch.index_select(y_sol, 0, A_row_indices)
            if dldx_mul_z is None:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol_select_col = torch.index_select(
                    - dldf_mul_A * z_sol, 0, A_col_indices)
            else:
                dldx_mul_z_sub_dldf_mul_A_mul_z_sol_select_col = torch.index_select(
                    dldx_mul_z - dldf_mul_A * z_sol, 0, A_col_indices)
            dldf_select_row = torch.index_select(dldf, 0, A_row_indices)
            x_sol_select_col = torch.index_select(x_sol, 0, A_col_indices)

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
            s_sol = c - torch.squeeze(torch.matmul(At, torch.unsqueeze(y_sol, dim=-1)), dim=-1)
            dxdu = (x_sol - z_sol * s_sol) / u
            if dldx is None:
                dldu = - dldf_mul_A * dxdu
            else:
                dldu = (dldx - dldf_mul_A) * dxdu
        else:
            dldu = None

        return dldA, dldb, dldc, dldu, None, None, None, None, None, None


class SparseBlockDiagAPDAGDLayer(torch.nn.Module):
    def __init__(self):
        super(SparseBlockDiagAPDAGDLayer).__init__()

    def forward(
        self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, u: torch.Tensor,
        n_c: torch.Tensor, n_v: torch.Tensor,
        theta: float, y: Optional[torch.Tensor] = None,
        eps: float = 1e-3, max_iter: Optional[int] = None
    ):
        return SparseBlockDiagAPDAGDFunction.apply(A, b, c, u, n_c, n_v, theta, y, eps, max_iter)


