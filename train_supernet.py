import argparse
from pathlib import Path
from easydict import EasyDict as EDict

import numpy as np
import torch
import torch.nn as nn

from pkg.utils.paths import *

from pkg.algs.supernet_training import supernet_training
from pkg.lib.datasets import get_datasets, get_nas_search_loaders
from pkg.lib.procedures import prepare_logger

from pkg.utils.train_utils import AdamOptim, CosineAnnealingScheduler, SGDOptim, StepScheduler, LinearScheduler, ConstantScheduler
from pkg.utils.out_utils import Output

from pkg.lib.natsbench import create
from pkg.utils.misc_utils import arch2alphas

INF = 1000
NASBENCH201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
LOSSES = {'cross_entropy' : nn.functional.cross_entropy}

def main(args):

    data_path = DATASET_PATHS[args['dataset']]

    logger = prepare_logger(args['log_name'], seed=args['rand_seed'])

    if args['initial_params_name'] is not None:
        initial_params_path = CHECKPOINTS_PATH + '/{:}.pth'.format(args['initial_params_name'])
    else:
        initial_params_path = None

    train_data, valid_data, xshape, class_num = get_datasets(args['dataset'], data_path, -1)
    t_search_loader, t_train_loader, t_valid_loader = get_nas_search_loaders(train_data, valid_data, args['dataset'], CONFIGS_PATH, args['training_batch_size'], args['workers'])

    loss_f = LOSSES[args['loss_function']]

    if args['optimizer'] == 'adam':
        optimizer = AdamOptim(lr=args['lr'], betas=tuple(args['adam_betas'][:2]), weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'sgd':
        optimizer = SGDOptim(lr=args['lr'], momentum=args['sgd_momentum'], dampening=args['sgd_dampening'], weight_decay=args['weight_decay'], nesterov=False)
    elif args['optimizer'] == 'sgd_nesterov':
        optimizer = SGDOptim(lr=args['lr'], momentum=args['sgd_momentum'], dampening=args['sgd_dampening'], weight_decay=args['weight_decay'], nesterov=True)

    if args['scheduler'] == 'step':
        scheduler = StepScheduler(args['decay_step_size'], args['gamma'])
    elif args['scheduler'] == 'linear':
        scheduler = LinearScheduler(args['start_gamma'], args['end_gamma'], args['decat_total_epochs'])
    elif args['scheduler'] == 'constant':
        scheduler = ConstantScheduler(args['gamma'], args['decay_total_epochs'])
    elif args['scheduler'] == 'cosine_annealing':
        scheduler = CosineAnnealingScheduler(args['cosine_initial_period'], args['cosine_period_mult'], args['cosine_min_lr'])

    model_config = EDict({
        'super_type' : 'basic', 'name' : 'generic',
        'C' : 16, 'N' : 5,
        'max_nodes' : 4, 'num_classes' : class_num, 'space' : NASBENCH201,
        'affine' : True, 'track_running_stats' : True
    })

    output_vars = ['avg_batch_time']
    for milestone in args['milestones']:
        output_vars += [str(milestone)+'_epochs_min_loss', str(milestone)+'_epochs_max_top1', str(milestone)+'_epochs_max_top5']
    out = Output(args['output_name'], output_vars, mode=args['output_mode'])
    
    supernet_parameters = np.zeros((6, 5))
    logger.log('Initial supernet parameters : {:}'.format(supernet_parameters))

    logger.log('------------- Training -------------')

    save_path = CHECKPOINTS_PATH + '/{:}.pth'.format(args['output_name'])

    milestones = [int(ms) for ms in args['milestones']]
    model, batch_times, losses, top1s, top5s = supernet_training(save_path, model_config, supernet_parameters, t_train_loader, t_valid_loader, loss_f, optimizer, scheduler, args['n_epochs'], args['lr'], logger, initial_params_path=initial_params_path, load_arch_from_params=args['load_arch_from_params'], load_weights_from_params=args['load_weights_from_params'], random_sampling=False, milestones=milestones, output=None, seed=args['rand_seed'])

    out.add_output('avg_batch_time', sum(batch_times)/len(batch_times))
    for m, milestone in enumerate(args['milestones']):
        out.add_output(str(milestone)+'_epochs_min_loss', losses[m])
        out.add_output(str(milestone)+'_epochs_max_top1', top1s[m])
        out.add_output(str(milestone)+'_epochs_max_top5', top5s[m])

    out.write_output()

    out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random search")

    parser.add_argument('--log_name', type=str, help='Folder to save logs.', required=True)
    parser.add_argument('--output_name', type=str, help='Name of output file.', required=True)
    parser.add_argument('--output_mode', type=str, default='online', choices=['online', 'offline'], help='Online : output updated in running time. \
        Offline : output created afterwards.')

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between CIFAR10/CIFAR100/ImageNet16-120.', required=True)
    parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy'], help='Loss function for metrics calculations and training.')

    parser.add_argument('--milestones', type=int, nargs='*', help='Training milestones to save.')
    parser.add_argument('--training_batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs to train for.', required=True)
    parser.add_argument('--samples_per_net', type=int, default=1, help='Number of sampled architectures for each subsupernet.')
    parser.add_argument('--initial_params_name', type=str, help='Name of model to load initial parameters from.')
    parser.add_argument('--load_arch_from_params', action='store_true', help='Whether to load the architecture from loaded model.')
    parser.add_argument('--load_weights_from_params', action='store_true', help='Whether to load weights from loaded model.')

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'sgd_nesterov'], help='Optimizer to use for training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--adam_betas', type=float, nargs='*', help='Adam coefficients.')
    parser.add_argument('--sgd_momentum', type=float, default=0, help='SGD momentum.')
    parser.add_argument('--sgd_dampening', type=float, default=0, help='SGD momentum dampening')

    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'linear', 'constant', 'cosine_annealing'], help='Scheduler to use for training.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay for schedulers.')
    parser.add_argument('--start_gamma', type=float, default=1.0 / 3, help='Starting learning rate factor for linear scheduler.')
    parser.add_argument('--end_gamma', type=float, default=1.0, help='Ending learning rate factor for linear scheduler.')
    parser.add_argument('--decay_total_epochs', type=int, default=5, help='Number of epochs to decay for.')
    parser.add_argument('--decay_step_size', type=int, default=5, help='Learning rate decay step size for step scheduler.')
    parser.add_argument('--cosine_initial_period', type=int, default=5, help='Initial number of iterations in cosine annealing restarts.')
    parser.add_argument('--cosine_period_mult', type=int, default=1, help='Factor with which to multiply cosine annealing number of iterations at each restart.')
    parser.add_argument('--cosine_min_lr', type=float, default=0, help='Minimum of cosine annealing learning rate at each restart.')

    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 0).')
    parser.add_argument('--rand_seed', type=int, default=0, help='Manual seed.')

    args = vars(parser.parse_args())
    if args['adam_betas'] == [] or args['adam_betas']==None:
        args['adam_betas'] = [0.9, 0.999]
    if args['milestones'] == [] or args['milestones']==None:
        args['milestones'] = [args['n_epochs']-1]
    main(args)