import os
import time
import argparse
from src.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from pathlib import Path


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', '--config', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=None, type=int)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number', default=None, type=int)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--output_path', type=str, default=None)
    # parser.add_argument('--pretrained_path', type=str, default='', help="Path of pretrained model")
    parser.add_argument('--project_temp', type=float, default=1e-1, help="Temperature to control the closeness to integer")
    parser.add_argument('--project_max_iter', type=int, default=int(1e2), help="Max number of iterations for projection")
    parser.add_argument('--project_dtype', type=str, choices=['float32', 'float64'], help="Dtype for projection")
    parser.add_argument('--project_way', type=str, choices=[
        'none', 'linsat', 'sparse_linsat', 'cvxpylayers', 'qpth',
        'dense_block_diag_apdagd_direct', 'dense_block_diag_apdagd_kkt',
        'sparse_block_diag_apdagd_direct', 'sparse_block_diag_apdagd_kkt'
    ], help="none: do not project\n"
            "linsat: use linsat to project and backward directly\n"
            "sparse_linsat: use sparse linsat to project and backward directly\n"
            "qpth: use qpth to project\n"
            "cvxpylayers: use cvxpylayers to project\n"
            "dense_block_diag_apdagd_direct: use dense block diag apdagd to project and backward directly\n"
            "dense_block_diag_apdagd_kkt: use dense block diag apdagd to project and backward via kkt condition\n"
            "sparse_block_diag_apdagd_direct: use sparse block diag apdagd to project and backward directly\n"
            "sparse_block_diag_apdagd_kkt: use sparse block diag apdagd to project and backward via kkt condition")
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # load cfg from arguments
    if args.batch_size is not None:
        cfg_from_list(['BATCH_SIZE', args.batch_size])
    if args.epoch is not None:
        cfg_from_list(['TRAIN.START_EPOCH', args.epoch, 'EVAL.EPOCH', args.epoch])
    cfg_from_list(['PROJECT_TEMP', args.project_temp,
                   'PROJECT_MAX_ITER', args.project_max_iter,
                   'PROJECT_WAY', args.project_way,
                   'PROJECT_DTYPE', args.project_dtype])
    # cfg_from_list(['PRETRAINED_PATH', args.pretrained_path])

    assert len(cfg.MODULE) != 0, 'Please specify a module name in your yaml file (e.g. MODULE: models.PCA.model).'
    assert len(cfg.DATASET_FULL_NAME) != 0, 'Please specify the full name of dataset in your yaml file (e.g. DATASET_FULL_NAME: PascalVOC).'

    if args.output_path is None:
        if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
            # outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
            if args.project_way == 'linsat' or args.project_way == 'sparse_linsat':
                outp_path = os.path.join(
                    'output', f'{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_{cfg.BATCH_SIZE}_{cfg.PROJECT_WAY}_{cfg.PROJECT_TEMP}_'
                              f'{cfg.PROJECT_DTYPE}_{cfg.PROJECT_MAX_ITER}_{time.strftime("%Y%m%dT%H%M%S")}')
            else:
                outp_path = os.path.join(
                    'output', f'{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_{cfg.BATCH_SIZE}_{cfg.PROJECT_WAY}_{cfg.PROJECT_TEMP}_'
                              f'{cfg.PROJECT_DTYPE}_{time.strftime("%Y%m%dT%H%M%S")}')
            cfg_from_list(['OUTPUT_PATH', outp_path])
    else:
        cfg_from_list(['OUTPUT_PATH', args.output_path])
    assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args
