import time

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

from ..lib.procedures.eval_funcs import obtain_accuracy
from ..lib.procedures.metric_utils import AverageMeter
from ..utils.train_utils import make_subsupernet, sample_arch

def supernet_training(save_path, model_config, arch_parameters, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs, lr, logger, valid_only=False, recalc_batchnorm_stats=False, cuda=True, initial_params_path=None, load_arch_from_params=False, load_weights_from_params=False, n_batches=0, random_sampling=True, batches_per_sample=1, milestones=[], output=None, seed=0):

    torch.manual_seed(seed)
    logger.log('Initial architecture : {:}'.format(arch_parameters))
    
    network = make_subsupernet(model_config, arch_parameters, initial_params_path, load_arch_from_params, load_weights_from_params, cuda=cuda)
    network.set_cal_mode('joint')
    logger.log('Arch parameters : {:}'.format(network.arch_parameters))
    optim_ = optimizer.make_optimizer(network.parameters())
    schedul_ = scheduler.make_scheduler(optim_)

    batch_times, losses, top1_accs, top5_accs = [], [], [], []
    batch_times_ms, losses_ms, top1_accs_ms, top5_accs_ms = [], [], [], []

    bestloss = np.inf

    for n in range(n_epochs):
        logger.log('Epoch {:}'.format(n))
        start_time = time.time()

        if not valid_only:
            batch_time_t, losses_t, top1_t, top5_t = supernet_training_step(network, train_loader, criterion, optim_, schedul_, logger, cuda=cuda, random_sampling=random_sampling, n_batches=n_batches, batches_per_sample=batches_per_sample)
        else:
            batch_time_t, losses_t, top1_t, top5_t = [], [], [], []

        batch_time_v, losses_v, top1_v, top5_v = supernet_valid(network, valid_loader, criterion, logger, cuda=cuda, n_batches=n_batches, recalc_batchnorm_stats=recalc_batchnorm_stats)

        schedul_.step()

        epoch_time = time.time() - start_time
        logger.log('Elapsed time : {:}'.format(epoch_time))

        if not valid_only:
            logger.log('Average batch time (train): {batch_time}'.format(batch_time=batch_time_t.val))
            logger.log('Loss (train): {losses}'.format(losses=losses_t.avg))
            if cuda:
                logger.log('Precision (train): top1 {top1} | top5 {top5}'.format(top1=top1_t.avg.detach().clone().cpu().numpy()[0], top5=top5_t.avg.detach().clone().cpu().numpy()[0]))
            else:
                logger.log('Precision (train): top1 {top1} | top5 {top5}'.format(top1=top1_t.avg.detach().clone().numpy()[0], top5=top5_t.avg.detach().clone().numpy()[0]))

            batch_times.append(batch_time_t.val)

        logger.log('Average batch time (valid): {batch_time}'.format(batch_time=batch_time_v.val))
        logger.log('Loss (valid): {losses}'.format(losses=losses_v.avg))
        if cuda:
            logger.log('Precision (valid): top1 {top1} | top5 {top5}'.format(top1=top1_v.avg.detach().clone().cpu().numpy()[0], top5=top5_v.avg.detach().clone().cpu().numpy()[0]))
        else:
            logger.log('Precision (valid): top1 {top1} | top5 {top5}'.format(top1=top1_v.avg.detach().clone().numpy()[0], top5=top5_v.avg.detach().clone().numpy()[0]))

        losses.append(losses_v.avg)
        if cuda:
            top1_accs.append(top1_v.avg.detach().clone().cpu().numpy()[0])
            top5_accs.append(top5_v.avg.detach().clone().cpu().numpy()[0])
        else:
            top1_accs.append(top1_v.avg.detach().clone().numpy()[0])
            top5_accs.append(top5_v.avg.detach().clone().numpy()[0])

        if not valid_only and save_path is not None and losses_v.avg < bestloss:
            logger.log('New best validation loss found, saving network parameters')
            torch.save(network, save_path)
            bestloss = losses_v.avg

        if n in milestones:

            if not valid_only:
                batch_times_ms.append(sum(batch_times)/len(batch_times))
            losses_ms.append(min(losses))
            top1_accs_ms.append(max(top1_accs))
            top5_accs_ms.append(max(top5_accs))

        if output is not None:
            output.add_output('epoch', n)
            output.add_output('epoch_time', epoch_time)

            if not valid_only:
                output.add_output('avg_batch_time_train', batch_time_t.val)
                output.add_output('loss_train', losses_t.avg)
                if cuda:
                    output.add_output('top1_precision_train', top1_t.avg.detach().clone().cpu().numpy()[0])
                    output.add_output('top5_precision_train', top5_t.avg.detach().clone().cpu().numpy()[0])
                else:
                    output.add_output('top1_precision_train', top1_t.avg.detach().clone().numpy()[0])
                    output.add_output('top5_precision_train', top5_t.avg.detach().clone().numpy()[0])

            output.add_output('avg_batch_time_valid', batch_time_v.val)
            output.add_output('loss_valid', losses_v.avg)
            if cuda:
                output.add_output('top1_precision_valid', top1_v.avg.detach().clone().cpu().numpy()[0])
                output.add_output('top5_precision_valid', top5_v.avg.detach().clone().cpu().numpy()[0])
            else:
                output.add_output('top1_precision_valid', top1_v.avg.detach().clone().numpy()[0])
                output.add_output('top5_precision_valid', top5_v.avg.detach().clone().numpy()[0])

            output.write_output()

        del batch_time_t, losses_t, top1_t, top5_t, batch_time_v, losses_v, top1_v, top5_v, start_time, epoch_time

    return(network, batch_times_ms, losses_ms, top1_accs_ms, top5_accs_ms)

def supernet_training_trainer_scheme(network, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs, logger, cuda=True, n_batches=0, random_sampling=True, batches_per_sample=1, seed=0):

    torch.manual_seed(seed)
    
    logger.log('################ Warmup ################')
    logger.log('Arch parameters : {:}'.format(network.arch_parameters))
    optim_ = optimizer.make_optimizer(network.parameters())
    schedul_ = scheduler.make_scheduler(optim_)

    bestloss = np.inf

    for n in range(n_epochs):
        logger.log('Epoch {:}'.format(n))
        start_time = time.time()

        batch_time_t, losses_t, top1_t, top5_t = supernet_training_step(network, train_loader, criterion, optim_, schedul_, logger, cuda=cuda, random_sampling=random_sampling, n_batches=n_batches, batches_per_sample=batches_per_sample)

        batch_time_v, losses_v, top1_v, top5_v = supernet_valid(network, valid_loader, criterion, logger, cuda=cuda, n_batches=n_batches, recalc_batchnorm_stats=False)

        schedul_.step()

        epoch_time = time.time() - start_time
        logger.log('Elapsed time : {:}'.format(epoch_time))

        logger.log('Average batch time (train): {batch_time}'.format(batch_time=batch_time_t.val))
        logger.log('Loss (train): {losses}'.format(losses=losses_t.avg))
        if cuda:
            logger.log('Precision (train): top1 {top1} | top5 {top5}'.format(top1=top1_t.avg.detach().clone().cpu().numpy()[0], top5=top5_t.avg.detach().clone().cpu().numpy()[0]))
        else:
            logger.log('Precision (train): top1 {top1} | top5 {top5}'.format(top1=top1_t.avg.detach().clone().numpy()[0], top5=top5_t.avg.detach().clone().numpy()[0]))

        logger.log('Average batch time (valid): {batch_time}'.format(batch_time=batch_time_v.val))
        logger.log('Loss (valid): {losses}'.format(losses=losses_v.avg))
        if cuda:
            logger.log('Precision (valid): top1 {top1} | top5 {top5}'.format(top1=top1_v.avg.detach().clone().cpu().numpy()[0], top5=top5_v.avg.detach().clone().cpu().numpy()[0]))
        else:
            logger.log('Precision (valid): top1 {top1} | top5 {top5}'.format(top1=top1_v.avg.detach().clone().numpy()[0], top5=top5_v.avg.detach().clone().numpy()[0]))

        if losses_v.avg < bestloss:
            logger.log('New best validation loss found, saving network parameters')
            bestloss = losses_v.avg
            bestweights = network.parameters()

        del batch_time_t, losses_t, top1_t, top5_t, batch_time_v, losses_v, top1_v, top5_v, start_time, epoch_time

    return(bestweights)

def supernet_training_step(network, train_loader, criterion, optimizer, scheduler, logger, cuda=True, random_sampling=True, n_batches=0, batches_per_sample=1):

    return(supernet_procedure(network, train_loader, criterion, optimizer, scheduler, logger, mode='train', cuda=cuda, random_sampling=random_sampling, n_batches=n_batches, batches_per_sample=batches_per_sample))

def supernet_valid(network, valid_loader, criterion, logger, cuda=True, recalc_batchnorm_stats=False, n_batches=0):

    return(supernet_procedure(network, valid_loader, criterion, None, None, logger, mode='valid', cuda=cuda, recalc_batchnorm_stats=recalc_batchnorm_stats, n_batches=n_batches, batches_per_sample=None))

def supernet_procedure(network, loader, criterion, optimizer, scheduler, logger, mode='train', cuda=True, recalc_batchnorm_stats=False, n_batches=None, random_sampling=True, batches_per_sample=1):

    arch_parameters_backup = deepcopy(network.arch_parameters.detach().clone())

    batch_time, losses, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    if mode=='train':
        network.train()
    elif mode=='valid':
        network.eval()
    else:
        raise ValueError('Unexpected mode {:}'.format(mode))

    if n_batches == 0:
        n_batches = len(loader)

    start_time = time.time()

    if mode=='valid' and recalc_batchnorm_stats:
        network.train()
        logger.log('Recalculating batch norm statistics')
        _loader = deepcopy(loader)
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(_loader):
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)
                features, logits = network(inputs)
        network.eval()

    for i, (inputs, targets) in enumerate(loader):
        if i < n_batches:
            arch_param_ = None
            #logger.log('Batch {:}'.format(i))

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)
            if mode=='train' and random_sampling and i%batches_per_sample == 0:
                arch_param_ = sample_arch(arch_parameters_backup)
                network.arch_parameters = nn.Parameter(arch_param_)
                #logger.log('Sampled architecture : {:}'.format(arch_param_))
            
            if mode=='train':
                optimizer.zero_grad()
                features, logits = network(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            elif mode=='valid':
                with torch.no_grad():
                    features, logits = network(inputs)
                    loss = criterion(logits, targets)

            network.arch_parameters = nn.Parameter(arch_parameters_backup)

            torch.cuda.empty_cache()
            
            batch_time.update(time.time() - start_time)
            prec1, prec5 = obtain_accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            del features, logits, loss, prec1, prec5, arch_param_

    del arch_parameters_backup, inputs, targets
    return(batch_time, losses, top1, top5)