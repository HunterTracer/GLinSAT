import json
from datetime import datetime
from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import time
import os
from matplotlib import pyplot as plt
from typing import Optional

from utils.math_utils import evaluate, compute_measures, compute_measures_batch, \
    compute_sharpe_ratio, compute_sharpe_ratio_batch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import qpth
from LinSATNet import linsat_layer
from dense_apdagd_layer import dense_apdagd, DenseAPDAGDFunction
from sparse_apdagd_layer import sparse_csr_block_diag_from_tensor, sparse_apdagd, SparseAPDAGDFunction


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, device, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f, map_location=device)
    return model


def project(weight, args):
    project_dtype = args.project_dtype
    device = weight.device
    torch.cuda.reset_peak_memory_stats(device=weight.device)
    max_memory_before_project = torch.cuda.max_memory_allocated(device=weight.device) / 1024 / 1024

    if args.project_way == "linsat" or args.project_way == 'sparse_linsat' or args.project_way == 'qpth':
        C = torch.zeros((1, 493), dtype=project_dtype, device=device)
        C[0, 0:5] = 1.
        d = torch.tensor([0.5], dtype=project_dtype, device=device)
        E = torch.ones((1, 493), dtype=project_dtype, device=device)
        f = torch.tensor([1.0], dtype=project_dtype, device=device)

        if args.project_way == 'linsat':
            st = time.time_ns()
            probs = linsat_layer(weight.to(dtype=project_dtype), C=C, d=d, E=E, f=f, tau=args.temp, max_iter=args.max_iter)
            probs = probs.to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        elif args.project_way == 'sparse_linsat':
            Ccoo = C.to_sparse_coo()
            Ecoo = E.to_sparse_coo()

            st = time.time_ns()
            probs = linsat_layer(weight.to(dtype=project_dtype), C=Ccoo, d=d, E=Ecoo, f=f, tau=args.temp, max_iter=args.max_iter)
            probs = probs.to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        else:
            Q = args.temp * torch.eye(493, dtype=project_dtype, device=device)
            p = weight.to(dtype=project_dtype)
            G = torch.cat([
                - C,
                torch.eye(493, dtype=project_dtype, device=device),
                - torch.eye(493, dtype=project_dtype, device=device),
            ], dim=0)
            h = torch.cat([
                - d,
                torch.ones(493, dtype=project_dtype, device=device),
                torch.zeros(493, dtype=project_dtype, device=device),
            ], dim=0)
            A = E
            b = f

            st = time.time_ns()
            probs = qpth.qp.QPFunction(eps=1e-3, verbose=0, maxIter=100000)(Q, p, G, h, A, b)
            probs = probs.to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
    elif args.project_way == "none":
        probs = weight
    else:
        A = torch.zeros((2, 493 + 1), dtype=project_dtype, device=device)
        A[0, 0:5] = 1.
        A[0, -1] = -1.
        A[1, :-1] = 1.
        b = torch.tensor([0.5, 1.], dtype=project_dtype, device=device)
        c = torch.cat([
            weight.to(dtype=project_dtype), torch.zeros((weight.shape[0], 1), dtype=project_dtype, device=device)
        ], dim=1)
        u = torch.ones(493 + 1, dtype=project_dtype, device=device)
        u[-1] = 4.5

        if args.project_way == 'cvxpylayers':
            x_cp = cp.Variable(A.shape[1], nonneg=True)
            c_cp = cp.Parameter(A.shape[1])
            u_cpu = u.cpu().detach().numpy()
            objective = cp.Minimize(
                cp.sum(cp.multiply(c_cp, x_cp)
                       - args.temp * cp.entr(cp.multiply(1. / u_cpu, x_cp))
                       - args.temp * cp.entr(1. - cp.multiply(1. / u_cpu, x_cp))))
            constraints = [A.detach().cpu().numpy() @ x_cp == b.detach().cpu().numpy(),
                           x_cp >= 0, x_cp <= u.detach().cpu().numpy()]
            prob = cp.Problem(objective, constraints)

            st = time.time_ns()
            opt_layer = CvxpyLayer(prob, parameters=[c_cp], variables=[x_cp])
            probs_slack, = opt_layer(c, solver_args={
                # "solve_method": "ECOS", "abstol": 1e-3,
                "solve_method": "SCS", "eps_abs": 1e-3,
                'n_jobs_forward': 24, 'n_jobs_backward': 24
            })
            probs = probs_slack[:, :-1].to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        elif args.project_way == 'dense_apdagd_direct':
            st = time.time_ns()
            probs_slack, _ = dense_apdagd(
                A=A.expand(weight.shape[0], -1, -1),
                b=b.expand(weight.shape[0], -1),
                c=c,
                u=u.expand(weight.shape[0], -1), theta=1. / args.temp)
            probs = probs_slack[:, :-1].to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        elif args.project_way == 'dense_apdagd_kkt':
            st = time.time_ns()
            probs_slack, _ = DenseAPDAGDFunction.apply(
                A.expand(weight.shape[0], -1, -1), b.expand(weight.shape[0], -1),
                c, u.expand(weight.shape[0], -1), 1. / args.temp
            )
            probs = probs_slack[:, :-1].to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        elif args.project_way == 'sparse_apdagd_direct':
            A_csr = A.to_sparse_csr()
            A = sparse_csr_block_diag_from_tensor(
                A_csr.crow_indices(),
                A_csr.col_indices(),
                A_csr.values().expand(weight.shape[0], -1),
                A_csr.shape
            )

            st = time.time_ns()
            probs_slack, _ = sparse_apdagd(
                A=A,
                b=b.expand(weight.shape[0], -1),
                c=c,
                u=u.expand(weight.shape[0], -1),
                theta=1. / args.temp
            )
            probs = probs_slack[:, :-1].to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        elif args.project_way == 'sparse_apdagd_kkt':
            A_csr = A.to_sparse_csr()
            A = sparse_csr_block_diag_from_tensor(
                A_csr.crow_indices(),
                A_csr.col_indices(),
                A_csr.values().expand(weight.shape[0], -1),
                A_csr.shape
            )

            st = time.time_ns()
            probs_slack, _ = SparseAPDAGDFunction.apply(
                A,
                b.expand(weight.shape[0], -1),
                c,
                u.expand(weight.shape[0], -1),
                1. / args.temp
            )
            probs = probs_slack[:, :-1].to(dtype=torch.float32)
            ed = time.time_ns()
            print('project_time/s:', (ed - st) / 1e9)
        else:
            raise ValueError(f"Undefined project_way: {args.project_way}")

    max_memory_after_project = torch.cuda.max_memory_allocated(device=weight.device) / 1024 / 1024
    # max_memory_reserved_after_project = torch.cuda.max_memory_reserved(device=s.device) / 1024 / 1024
    print('max_memory_allocated before project/MB:', max_memory_before_project)
    print('max_memory_allocated after project/MB:', max_memory_after_project)
    print('max_memory_allocated during project/MB:', max_memory_after_project - max_memory_before_project)

    return probs


def train_and_test_loop(train_data, valid_data, args, result_file):
    node_cnt = train_data.shape[1]
    model = Model(units=node_cnt, stack_cnt=2, time_step=args.window_size, multi_layer=args.multi_layer,
                  horizon=args.horizon, dropout_rate=args.dropout_rate, leaky_rate=args.leakyrelu_rate)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    
    train_loader = torch_data.DataLoader(
        train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
    valid_loader = torch_data.DataLoader(
        valid_set, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=0)

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}", flush=True)

    best_validate_sharpe_ratio = - np.inf
    validate_score_non_decrease_count = 0
    to_plot = []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_f_total = 0.
        loss_s_total = 0.
        cons_1_inf_total = 0.
        cons_2_inf_total = 0.
        infeasible_num = 0
        cnt = 0
        # with torch.autograd.set_detect_anomaly(True):
        # with torch.autograd.detect_anomaly():
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            my_optim.zero_grad()
            forecast, _, weight = model(inputs)

            pred_weight = args.pred_weight
            loss_f = pred_weight * forecast_loss(forecast, target)

            sharpe_weight = args.sharpe_weight
            mus, covs = compute_measures_batch(target)
            # sharpes = compute_sharpe_ratio_batch(mus, covs, weight, 0.03)
            # loss_s = torch.sum(- sharpe_weight * sharpes)
            probs = project(weight, args)
            sharpes = compute_sharpe_ratio_batch(mus, covs, probs, 0.03)
            infeasible_num += torch.sum(torch.logical_or(
                (torch.sum(probs[:, 0:5], dim=1) + 1e-3 < 0.5),
                torch.abs(torch.sum(probs, dim=1) - 1.) > 1e-3
            )).item()
            cons_1_inf = torch.sum(torch.relu(0.5 - torch.sum(probs[:, 0:5], dim=1)))
            cons_2_inf = torch.sum(torch.abs(torch.sum(probs, dim=1) - 1.))
            loss_s = torch.sum(- sharpe_weight * sharpes)
            cnt += target.shape[0]
            loss = loss_f + loss_s
            if loss != loss:
                raise ValueError("NaN in loss")

            torch.cuda.reset_max_memory_allocated(device=args.device)
            max_memory_before_backward = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
            st_backward = time.time_ns()
            loss.backward()
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print('encounter nan gradient')
                        valid_gradients = False
                        break
                    if torch.isinf(param.grad).any():
                        print('encounter inf gradient')
                        valid_gradients = False
                        break
            if not valid_gradients:
                # raise ValueError("Invalid gradient!")
                my_optim.zero_grad()
            else:
                my_optim.step()
            ed_backward = time.time_ns()
            max_memory_after_backward = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
            print('backward time/s:', (ed_backward - st_backward) / 1e9)
            print('max_memory_allocated before backward/MB:', max_memory_before_backward)
            print('max_memory_allocated after backward/MB:', max_memory_after_backward)
            print('max_memory_allocated during backward/MB:', max_memory_after_backward - max_memory_before_backward)

            loss_f_total += float(loss_f)
            loss_s_total += float(loss_s)
            cons_1_inf_total += float(cons_1_inf)
            cons_2_inf_total += float(cons_2_inf)
        # loss_total = loss_f_total + loss_s_total
        epoch_train_end_time = time.time()
        print(f'train inf: {cons_1_inf_total / cnt}, {cons_2_inf_total / cnt}', flush=True)
        print(f'train infeasible num: {infeasible_num}', flush=True)

        print('Start evaluation...')
        model.eval()
        sharpe_test = 0.
        cons_1_inf_total = 0.
        cons_2_inf_total = 0.
        infeasible_num = 0
        count = 0
        for j, (inputs, target) in enumerate(valid_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            forecast, _, weight = model(inputs)

            count += target.shape[0]
            mus, covs = compute_measures_batch(target)
            # mus_pre, covs_pre = compute_measures(forecast)
            probs = project(weight, args)
            sharpes = compute_sharpe_ratio_batch(mus, covs, probs, 0.03)
            cons_1_inf = torch.sum(torch.relu(0.5 - torch.sum(probs[:, 0:5], dim=1)))
            cons_2_inf = torch.sum(torch.abs(torch.sum(probs, dim=1) - 1.))
            infeasible_num += torch.sum(torch.logical_or(
                (torch.sum(probs[:, 0:5], dim=1) + 1e-3 < 0.5),
                torch.abs(torch.sum(probs, dim=1) - 1.) > 1e-3
            )).item()
            sharpe_test += torch.sum(sharpes)
            cons_1_inf_total += float(cons_1_inf)
            cons_2_inf_total += float(cons_2_inf)
        print(f'eval inf: {cons_1_inf_total / count}, {cons_2_inf_total / count}', flush=True)
        print(f'eval infeasible num: {infeasible_num}', flush=True)
        average_sharpe_test = sharpe_test / count
        epoch_eval_end_time = time.time()
        print("The average sharpe ratio is:", average_sharpe_test.item(), flush=True)
        
        to_plot.append((sharpe_test / count).cpu().detach().numpy().item())
        if (epoch + 1) % 2 == 0:
            save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if average_sharpe_test > best_validate_sharpe_ratio:
            save_model(model, result_file)
            best_validate_sharpe_ratio = average_sharpe_test
            validate_score_non_decrease_count = 0
        else:
            validate_score_non_decrease_count += 1
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
        print('| end of epoch {:3d} | time: {:5.2f}s | train time: {:5.2f}s | eval time: {:5.2f}s '
              '| train_total_loss_f {:5.4f} | train_total_loss_s {:5.4f} | average_sharpe_test {:5.4f}'.format(
            epoch, time.time() - epoch_start_time,
            epoch_train_end_time - epoch_start_time, epoch_eval_end_time - epoch_train_end_time,
            loss_f_total / cnt, loss_s_total / cnt, average_sharpe_test.item()), flush=True)
            
    plt.plot(to_plot)
    plt.savefig(os.path.join(result_file, "sharpe_ratio.png"))
    np.save(os.path.join(result_file, "average_sharpe_test.npy"), np.array(to_plot))


def validation(valid_data, args):
    # node_cnt = train_data.shape[1]
    # model = Model(units=node_cnt, stack_cnt=2, time_step=args.window_size, multi_layer=args.multi_layer,
    #               horizon=args.horizon, dropout_rate=args.dropout_rate, leaky_rate=args.leakyrelu_rate)
    model = load_model(model_dir=args.model_dir, device=args.device, epoch=args.epoch)
    model.to(args.device)
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if os.path.exists(os.path.join(args.model_dir, 'norm_stat.json')):
        with open(os.path.join(args.model_dir, 'norm_stat.json'), 'r') as f:
            normalize_statistic = json.load(f)
        if 'mean' in normalize_statistic and 'std' in normalize_statistic:
            args.norm_method = 'z_score'
        elif 'min' in normalize_statistic and 'max' in normalize_statistic:
            args.norm_method = 'min_max'
        else:
            args.norm_method = None
            normalize_statistic = None
    else:
        args.norm_method = None
        normalize_statistic = None

    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)

    valid_loader = torch_data.DataLoader(
        valid_set, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=0)

    eval_start_time = time.time_ns()
    model.eval()
    sharpe_test = 0.
    cons_1_inf_total = 0.
    cons_2_inf_total = 0.
    infeasible_num = 0
    count = 0
    project_time = 0.
    for j, (inputs, target) in enumerate(valid_loader):
        inputs = inputs.to(args.device)
        target = target.to(args.device)
        forecast, _, weight = model(inputs)

        count += target.shape[0]
        mus, covs = compute_measures_batch(target)
        # mus_pre, covs_pre = compute_measures(forecast)
        project_start_time = time.time_ns()
        probs = project(weight, args, is_train=False)
        project_end_time = time.time_ns()
        sharpes = compute_sharpe_ratio_batch(mus, covs, probs, 0.03)
        cons_1_inf = torch.sum(torch.relu(0.5 - torch.sum(probs[:, 0:5], dim=1)))
        cons_2_inf = torch.sum(torch.abs(torch.sum(probs, dim=1) - 1.))
        infeasible_num += torch.sum(torch.logical_or(
            (torch.sum(probs[:, 0:5], dim=1) + 1e-3 < 0.5),
            torch.abs(torch.sum(probs, dim=1) - 1.) > 1e-3
        )).item()
        sharpe_test += torch.sum(sharpes)
        cons_1_inf_total += float(cons_1_inf)
        cons_2_inf_total += float(cons_2_inf)
        project_time += project_end_time - project_start_time
    print(f'eval inf: {cons_1_inf_total / count}, {cons_2_inf_total / count}', flush=True)
    print(f'eval infeasible num: {infeasible_num}', flush=True)
    average_sharpe_test = sharpe_test / count
    eval_end_time = time.time_ns()
    print(f"The average sharpe ratio is: {average_sharpe_test.item()}, "
          f"eval time: {(eval_end_time - eval_start_time) / 1e9}, "
          f"project time: {project_time / 1e9}", flush=True)
