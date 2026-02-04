import time
import torch
import numpy as np
from LinSATNet import linsat_layer
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import qpth
from dense_apdagd_layer import dense_apdagd, DenseAPDAGDFunction
from sparse_apdagd_layer import sparse_csr_block_diag_from_tensor, sparse_apdagd, SparseAPDAGDFunction
# from sparse_apdagd_shared_layer import sparse_apdagd_shared, SparseAPDAGDSharedFunction


def StartEnd_constrain(node_num):
    #transfer the constrain Sum_i X_ij = 1, Sum_j Xij = 1, X_s1 = 1, X_en = 1, to Ax = b, x is the flattened X
    #this function return the required A and b
    b = torch.ones(2 * node_num + 2, dtype=torch.float32)
    A = torch.zeros([2 * node_num + 2, node_num * node_num], dtype=torch.float32)
    
    #the row constrain
    for i in range(node_num):
        A[i, node_num * i : node_num * (i + 1)] = 1

    #the column constrain
    column_gap = node_num * torch.arange(node_num)
    for i in range(node_num):
        A[node_num + i, i + column_gap] = 1

    #start and end node constrain, set to 0.999 to avoid numerical issue
    A[2 * node_num, 0] = 1; b[2 * node_num] = 0.999
    A[2 * node_num + 1, -1] = 1; b[2 * node_num + 1] = 0.999
    
    return A, b

def Priority_constrain(node_num, priority_level = 6):
    #Start-End constrain with priority customer, the second node should be visited within priority_level steps

    b = torch.ones(2 * node_num + 3, dtype=torch.float32)
    A = torch.zeros([2 * node_num + 3, node_num * node_num], dtype=torch.float32)
    
    #the row constrain
    for i in range(node_num):
        A[i, node_num * i : node_num * (i + 1)] = 1

    #the column constrain
    column_gap = node_num * torch.arange(node_num)
    for i in range(node_num):
        A[node_num + i, i + column_gap] = 1

    #start and end node constrain
    A[2 * node_num, 0] = 1; b[2 * node_num] = 0.999
    A[2 * node_num + 1, -1] = 1; b[2 * node_num + 1] = 0.999

    #second node should be visited within priority_level steps
    A[2 * node_num + 2, node_num : node_num + priority_level + 1] = 1; b[2 * node_num + 2] = 0.999
    
    return A, b


def project_one_batch(
    pre_project_logits, project_way, device, temp = 1e0, max_iter = 100, constrain_left = None, constrain_right = None
):
    torch.cuda.reset_peak_memory_stats(device=device)
    max_memory_before_project = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024

    batch_size, node_num, _ = pre_project_logits.shape
    pre_project_logits_flatten = pre_project_logits.reshape(-1, node_num * node_num)
    if project_way == 'linsat':
        constrain_left = constrain_left.to(device)
        constrain_right = constrain_right.to(device)
        st = time.time_ns()
        post_project_exp_flatten = linsat_layer(
            x=pre_project_logits_flatten, E=constrain_left, f=constrain_right,
            tau=temp, max_iter=max_iter, no_warning=False)
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'sparse_linsat':
        constrain_left = constrain_left.to_sparse_coo().to(device)
        constrain_right = constrain_right.to(device)
        st = time.time_ns()
        post_project_exp_flatten = linsat_layer(
            x=pre_project_logits_flatten, E=constrain_left, f=constrain_right,
            tau=temp, max_iter=max_iter, no_warning=False)
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'qpth':
        constrain_left = constrain_left.to(device)
        constrain_right = constrain_right.to(device)
        Q = temp * torch.eye(constrain_left.shape[1], dtype=constrain_left.dtype, device=device)
        G = torch.cat([
            torch.eye(constrain_left.shape[1], dtype=constrain_left.dtype, device=device),
            - torch.eye(constrain_left.shape[1], dtype=constrain_left.dtype, device=device),
        ], dim=0)
        h = torch.cat([
            torch.ones(constrain_left.shape[1], dtype=constrain_left.dtype, device=device),
            torch.zeros(constrain_left.shape[1], dtype=constrain_left.dtype, device=device),
        ], dim=0)
        st = time.time_ns()
        post_project_exp_flatten = qpth.qp.QPFunction(eps=1e-3, verbose=0, maxIter=100000)(
            Q,
            pre_project_logits_flatten,
            G,
            h,
            constrain_left,
            constrain_right,
        )
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'dense_apdagd_direct':
        constrain_left = constrain_left.to(device).expand(batch_size, -1, -1)
        constrain_right = constrain_right.to(device).expand(batch_size, -1)
        st = time.time_ns()
        post_project_exp_flatten, _ = dense_apdagd(
            A=constrain_left, b=constrain_right,
            c=pre_project_logits_flatten, u=torch.ones_like(pre_project_logits_flatten), theta=1. / temp
        )
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'dense_apdagd_kkt':
        constrain_left = constrain_left.to(device).expand(batch_size, -1, -1)
        constrain_right = constrain_right.to(device).expand(batch_size, -1)
        st = time.time_ns()
        post_project_exp_flatten, _ = DenseAPDAGDFunction.apply(
            constrain_left, constrain_right,
            pre_project_logits_flatten, torch.ones_like(pre_project_logits_flatten), 1. / temp
        )
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'sparse_apdagd_direct':
        constrain_left_sparse_csr = constrain_left.to_sparse_csr()
        A = sparse_csr_block_diag_from_tensor(
            constrain_left_sparse_csr.crow_indices(),
            constrain_left_sparse_csr.col_indices(),
            constrain_left_sparse_csr.values().expand(batch_size, -1),
            constrain_left.shape
        ).to(device)
        constrain_right = constrain_right.to(device).expand(batch_size, -1)
        st = time.time_ns()
        post_project_exp_flatten, _ = sparse_apdagd(
            A=A, b=constrain_right,
            c=pre_project_logits_flatten, u=torch.ones_like(pre_project_logits_flatten), theta=1. / temp
        )
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'sparse_apdagd_kkt':
        constrain_left_sparse_csr = constrain_left.to_sparse_csr()
        A = sparse_csr_block_diag_from_tensor(
            constrain_left_sparse_csr.crow_indices(),
            constrain_left_sparse_csr.col_indices(),
            constrain_left_sparse_csr.values().expand(batch_size, -1),
            constrain_left.shape
        ).to(device)
        constrain_right = constrain_right.to(device).expand(batch_size, -1)
        st = time.time_ns()
        post_project_exp_flatten, _ = SparseAPDAGDFunction.apply(
            A, constrain_right,
            pre_project_logits_flatten, torch.ones_like(pre_project_logits_flatten), 1. / temp
        )
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    # elif project_way == 'sparse_apdagd_shared_direct':
    #     constrain_left_sparse_csr = constrain_left.to_sparse_csr()
    #     crow_indices = constrain_left_sparse_csr.crow_indices()
    #     col_indices = constrain_left_sparse_csr.col_indices()
    #     values = constrain_left_sparse_csr.values()
    #     shape = torch.tensor(constrain_left_sparse_csr.shape, device=values.device)
    #     post_project_exp_flatten, _ = sparse_apdagd_shared(
    #         A_crow_indices=crow_indices, A_col_indices=col_indices, A_values=values, A_shape=shape,
    #         b=constrain_right.expand(batch_size, -1),
    #         c=pre_project_logits_flatten, u=torch.ones_like(pre_project_logits_flatten), theta=1. / temp
    #     )
    #     post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    # elif project_way == 'sparse_apdagd_shared_kkt':
    #     constrain_left_sparse_csr = constrain_left.to_sparse_csr()
    #     crow_indices = constrain_left_sparse_csr.crow_indices()
    #     col_indices = constrain_left_sparse_csr.col_indices()
    #     values = constrain_left_sparse_csr.values()
    #     shape = torch.tensor(constrain_left_sparse_csr.shape, device=values.device)
    #     post_project_exp_flatten, _ = SparseAPDAGDSharedFunction.apply(
    #         crow_indices, col_indices, values, shape,
    #         constrain_right.expand(batch_size, -1),
    #         pre_project_logits_flatten, torch.ones_like(pre_project_logits_flatten), 1. / temp
    #     )
    #     post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    elif project_way == 'cvxpylayers':
        x = cp.Variable(constrain_left.shape[1], nonneg=True)
        c = cp.Parameter(constrain_left.shape[1])
        objective = cp.Minimize(cp.sum(cp.multiply(c, x) - temp * cp.entr(x) - temp * cp.entr(1. - x)))
        constraints = [constrain_left @ x == constrain_right,
                       x >= 0, x <= 1]
        prob = cp.Problem(objective, constraints)
        st = time.time_ns()
        opt_layer = CvxpyLayer(prob, parameters=[c], variables=[x])
        post_project_exp_flatten, = opt_layer(pre_project_logits_flatten, solver_args={
            'n_jobs_forward': 24, 'n_jobs_backward': 24
        })
        ed = time.time_ns()
        print('project_time/s:', (ed - st) / 1e9)
        post_project_exp = post_project_exp_flatten.reshape(-1, node_num, node_num)
    else:
        raise ValueError(f"Undefined project_way: {project_way}")

    max_memory_after_project = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024
    # max_memory_reserved_after_project = torch.cuda.max_memory_reserved(device=s.device) / 1024 / 1024
    print('max_memory_allocated before project/MB:', max_memory_before_project)
    print('max_memory_allocated after project/MB:', max_memory_after_project)
    print('max_memory_allocated during project/MB:', max_memory_after_project - max_memory_before_project)

    return post_project_exp



