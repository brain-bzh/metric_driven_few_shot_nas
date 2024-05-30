import argparse
from pathlib import Path
from easydict import EasyDict as EDict

import numpy as np
import torch

from pkg.utils.paths import *

from pkg.lib.procedures import prepare_logger

from pkg.utils.out_utils import Output

def convert_mask_to_newformat(mask):

    mask_ = torch.ones_like(mask)
    for e in range(mask.shape[0]):
        for o in range(mask.shape[1]):
            if mask[e, o] <= -500:
                mask_[e, o] = 0
    return(mask_)

def main(args):

    logger = prepare_logger(args.log_name, seed=args.rand_seed)

    output_vars = ['metric', 'idx', 'mask']
    out = Output(args.output_name, output_vars, mode=args.output_mode)

    for metric in args.metrics:

        logger.log('##################### {:} #####################'.format(metric))
        for n in range(args.n_nets):
            net_name = 'split_{:}{:}_subsupernet{:}'.format(metric, args.rand_seed, n)

            checkpoint_path = '{:}/{:}.pth'.format(CHECKPOINTS_PATH, net_name)

            model = torch.load(checkpoint_path)
            arch_params = model.arch_parameters
            mask_legacy = arch_params.clone().detach().cpu()
            mask = convert_mask_to_newformat(mask_legacy).numpy()

            logger.log('Found mask for subsupernet {:} : {:}'.format(n, repr(mask)))

            out.add_output('metric', metric)
            out.add_output('idx', n)
            out.add_output('mask', repr(mask))
            out.write_output()

    out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random search")

    parser.add_argument('--log_name', type=str, help='Folder to save logs.', required=True)
    parser.add_argument('--output_name', type=str, help='Name of output file.', required=True)
    parser.add_argument('--output_mode', type=str, default='online', choices=['online', 'offline'], help='Online : output updated in running time. \
        Offline : output created afterwards.')

    parser.add_argument('--metrics', type=str, nargs='*', help='Name of the metrics in checkpoints.', required=True)
    parser.add_argument('--n_nets', type=int, help='Number of supernets for each metric.', required=True)
    parser.add_argument('--rand_seed', type=int, default=0, help='Seed of the splitting checkpoints.', required=True)

    args = parser.parse_args()

    main(args)
