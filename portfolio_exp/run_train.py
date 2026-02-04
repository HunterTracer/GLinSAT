import os
import torch
import time
from models.handler import train_and_test_loop
import argparse
import pandas as pd
from utils.setting_utils import get_snp500_keys


def main(args):
    before_train = time.time_ns()
    data_file = os.path.join('dataset', args.dataset + '.csv')
    if args.project_way == 'linsat':
        save_dir = os.path.join(
            'output', f'{args.dataset}_{args.project_way}_{args.temp}_{args.max_iter}_{time.strftime("%Y%m%dT%H%M%S")}')
    elif args.project_way == 'none':
        save_dir = os.path.join(
            'output', f'{args.dataset}_{args.project_way}_{time.strftime("%Y%m%dT%H%M%S")}')
    else:
        save_dir = os.path.join(
            'output', f'{args.dataset}_{args.project_way}_{args.temp}_{time.strftime("%Y%m%dT%H%M%S")}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = pd.read_csv(data_file)
    snp500 = get_snp500_keys()
    data = data[snp500].pct_change().dropna().values
    data = data[1:-1,1:].astype(float)
    train_ratio = args.train_length / (args.train_length + args.test_length)
    test_ratio = 1 - train_ratio
    train_data = data[:int(train_ratio * len(data))]
    test_data = data[int(train_ratio * len(data))-60:]
    torch.manual_seed(0)
    train_and_test_loop(train_data, test_data, args, save_dir)
    after_train = time.time_ns()
    print(f'Training and testing took {(after_train - before_train) / 1e9} seconds')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='snp500')
        parser.add_argument('--window_size', type=int, default=120)
        parser.add_argument('--horizon', type=int, default=120)
        parser.add_argument('--train_length', type=float, default=7.5)
        parser.add_argument('--test_length', type=float, default=2.5)
        parser.add_argument('--epoch', type=int, default=50)
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--multi_layer', type=int, default=2)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--norm_method', type=str, default=None)
        parser.add_argument('--optimizer', type=str, default='RMSProp')
        parser.add_argument('--early_stop', type=bool, default=False)
        parser.add_argument('--exponential_decay_step', type=int, default=5)
        parser.add_argument('--decay_rate', type=float, default=1.)
        parser.add_argument('--dropout_rate', type=float, default=0.)
        parser.add_argument('--leakyrelu_rate', type=float, default=0.1)
        parser.add_argument('--sharpe_weight', type=float, default=1)
        parser.add_argument('--pred_weight', type=float, default=1)
        parser.add_argument('--temp', type=float, default=1e-1, help="Temperature to control the closeness to integer")
        parser.add_argument('--max_iter', type=int, default=int(1e2), help="Max number of iterations only used for projection using LinSAT")
        parser.add_argument('--project_way', type=str, choices=[
            'none', 'linsat', 'sparse_linsat', 'qpth', 'cvxpylayers',
            'dense_apdagd_direct', 'dense_apdagd_kkt',
            'sparse_apdagd_direct', 'sparse_apdagd_kkt'
        ], help="none: do not project\n"
                "linsat: use linsat to project and backward directly\n"
                "sparse_linsat: use sparse linsat to project and backward directly\n"
                "qpth: use qpth to project and backward\n"
                "cvxpylayers: use cvxpylayers to project\n"
                "dense_apdagd_direct: use dense apdagd to project and backward directly\n"
                "dense_apdagd_kkt: use dense apdagd to project and backward via kkt condition\n"
                "sparse_apdagd_direct: use sparse apdagd to project and backward directly\n"
                "sparse_apdagd_kkt: use sparse apdagd to project and backward via kkt condition")
        parser.add_argument('--project_dtype', type=str, default='float32', choices=['float32', 'float64'], help="Dtype for projection")
        args = parser.parse_args()
        if args.project_dtype == 'float32':
            args.project_dtype = torch.float32
        elif args.project_dtype == 'float64':
            args.project_dtype = torch.float64
        else:
            raise ValueError(f"Undefined project_dtype: {args.project_dtype}")
        print(f'Training configs: {args}')
        main(args)
    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')
    print('Done')
