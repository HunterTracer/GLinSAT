import os
import torch
import time
from models.handler import validation
import argparse
import pandas as pd
from utils.setting_utils import get_snp500_keys


def main(args):
    before_validation = time.time_ns()
    data_file = os.path.join('dataset', args.dataset + '.csv')
    data = pd.read_csv(data_file)
    snp500 = get_snp500_keys()
    data = data[snp500].pct_change().dropna().values
    data = data[1:-1,1:].astype(float)
    train_ratio = args.train_length / (args.train_length + args.test_length)
    test_ratio = 1 - train_ratio
    train_data = data[:int(train_ratio * len(data))]
    test_data = data[int(train_ratio * len(data))-60:]
    torch.manual_seed(0)
    validation(test_data, args)
    after_validation = time.time_ns()
    print(f'Testing took {(after_validation - before_validation) / 1e9} seconds')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_dir', type=str)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--epoch', type=int, default=None)
        parser.add_argument('--dataset', type=str, default='snp500')
        parser.add_argument('--window_size', type=int, default=120)
        parser.add_argument('--horizon', type=int, default=120)
        parser.add_argument('--train_length', type=float, default=7.5)
        parser.add_argument('--test_length', type=float, default=2.5)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--temp', type=float, default=1e-1, help="Temperature to control the closeness to integer")
        parser.add_argument('--max_iter', type=int, default=int(1e2), help="Max number of iterations for projection")
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
        parser.add_argument('--project_dtype', type=str, choices=['float32', 'float64'], help="Dtype for projection")
        args = parser.parse_args()
        print(f'Training configs: {args}')
        main(args)
    except KeyboardInterrupt:
        print('-' * 99)
        print('Exiting from training early')
    print('Done')