import argparse
from pathlib import Path
from easydict import EasyDict as EDict

import numpy as np
import torch
import torch.nn as nn

from pkg.utils.paths import *

from pkg.metrics.gradient import GradientCollector, LossGradientCollector
from pkg.metrics.grasp import HessianCollector, GRASP
from pkg.metrics.synaptic import SynflowCollector, Synaptic
from pkg.metrics.linear_regions import PostActivationsCollector, LinearRegions

from pkg.metrics.grad_norm import GradNorm
from pkg.metrics.gradient_matching import GradientMatching
from pkg.metrics.jacob_cov import JacobCov
from pkg.metrics.ntk import NTKCond
from pkg.metrics.parameters import NParametersCollector, NParameters

from pkg.policies.splitting import SplittingPolicy
from pkg.policies.trainer import Trainer

from pkg.algs.supernet_training import supernet_training, supernet_training_trainer_scheme
from pkg.lib.datasets import get_datasets, get_nas_search_loaders
from pkg.lib.procedures import prepare_logger
from pkg.utils.train_utils import make_subsupernet

from pkg.utils.train_utils import AdamOptim, CosineAnnealingScheduler, SGDOptim, StepScheduler, LinearScheduler, ConstantScheduler
from pkg.utils.out_utils import Output

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

    if args['split_edges_mode']=='order':
        split_edges = args['split_order']
    elif args['split_edges_mode']=='random':
        split_edges = None

    train_data, valid_data, xshape, class_num = get_datasets(args['dataset'], data_path, -1)
    m_search_loader, m_train_loader, m_valid_loader = get_nas_search_loaders(train_data, valid_data, args['dataset'], CONFIGS_PATH, args['metrics_batch_size'], args['workers'])
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

    eval_config = model_config

    if args['metric'] == 'random':
        metric_name = 'random'
        metric = None

    elif args['metric'] == 'gradnorm':
        metric_name = 'gradnorm'
        gc_model = GradientCollector(m_train_loader, args['metrics_batches'], model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = GradNorm([gc_model], repeat=args['repeat'], seed=args['rand_seed'])

    elif args['metric'] == 'gradientmatching':
        metric_name = 'gradientmatching'
        gc_model = GradientCollector(m_train_loader, args['metrics_batches'], model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = GradientMatching([gc_model], repeat=args['repeat'], seed=args['rand_seed'])

    elif args['metric'] == 'jacobcov':
        metric_name = 'jacobcov'
        gc_model = GradientCollector(m_train_loader, args['metrics_batches'], model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = JacobCov([gc_model], repeat=args['repeat'], seed=args['rand_seed'])

    elif args['metric'] == 'ntk':
        metric_name = 'ntk'
        gc_model = GradientCollector(m_train_loader, args['metrics_batches'], model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = NTKCond([gc_model], repeat=args['repeat'], seed=args['rand_seed'])

    elif args['metric'] == 'snip':
        metric_name = 'snip'
        lgc_model = LossGradientCollector(m_train_loader, args['metrics_batches'], loss_f=loss_f, model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = Synaptic([lgc_model], repeat=args['repeat'], seed=args['rand_seed'], mode='snip')

    elif args['metric'] == 'grasp':
        metric_name = 'grasp'
        hc_model = HessianCollector(m_train_loader, args['metrics_batches'], loss_f=loss_f, model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = GRASP([hc_model], repeat=args['repeat'], seed=args['rand_seed'])

    elif args['metric'] == 'synflow':
        metric_name = 'synflow'
        sfc_model = SynflowCollector(m_train_loader, args['metrics_batches'], model_config=eval_config, init='kaiming_normal_fanout', seed=args['rand_seed'])
        metric = Synaptic([sfc_model], repeat=args['repeat'], seed=args['rand_seed'], mode='synflow')

    elif args['metric'] == 'linearregions':
        metric_name = 'linearregions'

        '''
        eval_config = EDict({
            'super_type' : 'basic', 'name' : 'generic',
            'C' : 1, 'N' : 1,
            'max_nodes' : 4, 'num_classes' : class_num, 'space' : NASBENCH201,
            'affine' : True, 'track_running_stats' : True,
            'depth' : 1, 'use_stem' : False
        })
        '''

        pac_model = PostActivationsCollector(m_train_loader, args['metrics_batches'], crop_size=args['crop_size'], model_config=eval_config, init='kaiming_normal_fanin', seed=args['rand_seed'])
        metric = LinearRegions([pac_model], repeat=args['repeat'], seed=args['rand_seed'])

    elif args['metric'] == 'nparams':
        metric_name = 'nparams'
        nparams_model = NParametersCollector(m_train_loader, args['metrics_batches'], model_config=eval_config, seed=args['rand_seed'])
        metric = NParameters([nparams_model], repeat=args['repeat'], seed=args['rand_seed'])

    if len(args['warmup_epochs'])==0:
        trainer=None
    else:
        trainer = Trainer(eval_config, 'joint', 'normal', supernet_training_trainer_scheme, train_loader=t_train_loader, valid_loader=t_valid_loader, criterion=loss_f, optimizer=optimizer, scheduler=scheduler, logger=logger, random_sampling=args['enable_random_sampling'], seed=args['rand_seed'])

    output_vars = ['subsupernet_idx', 'avg_batch_time']
    for milestone in args['milestones']:
        output_vars += [str(milestone)+'_epochs_min_loss', str(milestone)+'_epochs_max_top1', str(milestone)+'_epochs_max_top5']
    out = Output(args['output_name'], output_vars, mode=args['output_mode'])
    
    supernet_parameters = np.zeros((6, 5))
    logger.log('Initial supernet parameters : {:}'.format(supernet_parameters))

    logger.log('##################### {:} #####################'.format(metric_name))
    logger.log('------------- Splitting -------------')
    branching_factors = [int(bf) for bf in args['branching_factors']]
    policy = SplittingPolicy(args['target_n_subsupernets'], branching_factors, edges=split_edges, metric=metric, trainer=trainer, warmup_epochs=args['warmup_epochs'], branch_mode=args['branch_mode'], drop_leaves=args['drop_leaves'], seed=args['rand_seed'])
    subsupernet_parameters = policy.split(supernet_parameters)

    logger.log('Found {:} supernets'.format(len(subsupernet_parameters)))
    for i, ssp in enumerate(subsupernet_parameters):
        logger.log('Index {:} subsupernet | {:}'.format(i, ssp))

        save_path = CHECKPOINTS_PATH + '/{:}{:}_subsupernet{:}.pth'.format(args['save_prefix'], args['rand_seed'], i)
        ssp_net = make_subsupernet(model_config, ssp)
        ssp_net.set_cal_mode('joint')
        torch.save(ssp_net, save_path)

    out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random search")

    parser.add_argument('--log_name', type=str, help='Folder to save logs.', required=True)
    parser.add_argument('--output_name', type=str, help='Name of output file.', required=True)
    parser.add_argument('--output_mode', type=str, default='online', choices=['online', 'offline'], help='Online : output updated in running time. \
        Offline : output created afterwards.')
    parser.add_argument('--save_prefix', type=str, help='Prefix for checkpoint files (followed by idx of the subsupernet and seed).')

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between CIFAR10/CIFAR100/ImageNet16-120.', required=True)
    parser.add_argument('--initial_params_name', type=str, help='Name of model to load initial weights from. If not specified, training will be started from scratch.')

    parser.add_argument('--metrics_batch_size', type=int, default=16, help='Batch size for metrics calculations.')
    parser.add_argument('--metrics_batches', type=int, default=1, help='Number of batches for metrics calculations.')
    parser.add_argument('--crop_size', type=int, default=3, help='Crop size for linear regions calculations.')
    parser.add_argument('--repeat', type=int, default=1, help='Number of repeat calculations of the metric.')

    parser.add_argument('--metric', type=str, default='random', choices=['random', 'gradnorm', 'gradientmatching', 'jacobcov', 'ntk', 'snip', 'grasp', 'synflow', 'linearregions', 'nparams'], help='Metric to split the supernet on.')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy'], help='Loss function for metrics calculations and training.')

    parser.add_argument('--target_n_subsupernets', type=int, default=4, help='Target number of subsupernets to split into.')
    parser.add_argument('--branch_mode', type=str, default='ascending', choices=['ascending'], help='Branching mode for splitting.')
    parser.add_argument('--branching_factors', type=int, nargs='*', help='List of branching factors for splitting.')
    parser.add_argument('--split_edges_mode', type=str, default='random', choices=['random', 'order'], help='Method of choosing edges to split on.')
    parser.add_argument('--split_order', type=int, nargs='*', help='List of edges to split in order mode.')
    parser.add_argument('--drop_leaves', action='store_true', help='Whether to drop leaves of the branching tree.')
    
    parser.add_argument('--warmup_epochs', type=int, nargs='*', help='List of number of epochs to warmup the networks for at each level. Leave empty to bypass warmup.')

    parser.add_argument('--milestones', type=int, nargs='*', help='Training milestones to save.')
    parser.add_argument('--training_batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--training_samples', type=int, default=1, help='Number of sampled architectures per step in training.')
    parser.add_argument('--enable_random_sampling', action='store_true', help='Whether to train by sampling random architectures.')

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
    if args['branching_factors'] == [] or args['branching_factors'] == None:
        args['branching_factors'] = [2]
    if args['split_order'] == [] or args['split_order'] == None:
        args['split_order'] = [0]
    if args['adam_betas'] == [] or args['adam_betas'] == None:
        args['adam_betas'] = [0.9, 0.999]
    if args['warmup_epochs'] == [] or args['warmup_epochs'] == None:
        args['warmup_epochs'] = []
    if args['milestones'] == [] or args['milestones']==None:
        if args['warmup_epochs'] == []:
            args['milestones'] = [0]
        else:
            args['milestones'] = [max(args['warmup_epochs'])-1]
    main(args)
