import time
import os
import logging
import torch
import functools
from .networks import SamplingMixIn
import numpy as np
from tqdm import tqdm
from .train_utils import communicate_grads

logger = logging.getLogger(__name__)

def supernet_training(network, architecture, criterion, optimizer, scheduler, train_loader, valid_loader, start_epoch, end_epoch, writer, device, test_loader=None, optimal_path=None, genotype=None, arch_train=False, save=None, disable_tdqm=False, n_proc=1):

    best_loss = np.inf

    for epoch in range(start_epoch, end_epoch):
        
        supernet_training_step(network, architecture, train_loader, valid_loader, criterion, optimizer, scheduler, device, arch_train, epoch, writer, disable_tdqm, n_proc)

        loss, ac = supernet_validation_step(network, valid_loader, criterion, epoch, device, writer, eval=False, how="valid")
        if test_loader:
            # want to use this loss
            loss, ac = supernet_validation_step(network, test_loader, criterion, epoch, device, writer, eval=False, how="test")
        scheduler.step()

        if optimal_path:
            final_path, arch_parameter_form, performance = optimal_path(network)
            logger.info(f'Arch Parameter: \n{arch_parameter_form}')
            logger.info(f'Accuracy: {performance:.05f}')
            writer.add_scalar("NasBench/Acc", performance, epoch)
            
        if (loss <= best_loss) and save:
            best_loss = loss
            save()
        
        if hasattr(network, "lam_scheduler"):
            network.lam = network.lam_scheduler.step()
            writer.add_scalar("Rates/Lam", network.lam, epoch)
        
        if arch_train and  hasattr(architecture, "reg_scheduler"):
            architecture.reg = architecture.reg_scheduler.step()
            writer.add_scalar("Rates/Reg", architecture.reg, epoch)
            logger.info(f"Arch regularization: {architecture.reg:0.2f}")

        if hasattr(architecture, "threshold"):
            arch_alphas_list = network.arch_alphas()
            arch_parameter_list = network.arch_parameters_iter()
            for arch_alphas, arch_parameter, typ in zip(arch_alphas_list, arch_parameter_list, ["normal", "reduced"]):
                #alphas_sorted, indices = torch.sort(arch_alphas, dim=-1, descending=True)
                num_edges, num_ops = network.n_edges, network._layerN
                for e in range(num_edges):
                    if torch.sum(arch_parameter[e]) < 2:
                        continue
                    mask = arch_parameter[e]
                    active_op = torch.where(mask > 0)[0]
                    alphas_sorted, indices = torch.sort(arch_alphas[e][active_op], descending=True)
                    if alphas_sorted[0]-alphas_sorted[1] >= architecture.threshold:
                        arch_parameter[e] = torch.zeros(num_ops)
                        arch_parameter[e][active_op[indices[0]]] = 1
                        logger.info(f"Edge-{typ} prunned {e}")
                        logger.info(f"Operation selected {indices[0]}")
                        logger.info(f"{network.arch_parameters_iter()}")

        writer.flush()
        
    return network


def supernet_training_step(network, architecture, train_loader, valid_loader, criterion, optimizer, scheduler, device, update_arch, epoch, writer, disable_tdqm, n_proc):
    network.train()
    start_time = time.time()
    arch_running_time = 0
    running_loss, correct, total, train_loss, acc = 0, 0, 0, 0, 0
   
    valid_loader_iter = iter(valid_loader)
    #for second order darts
    momentum = optimizer.param_groups[0]["momentum"]
    weight_decay = optimizer.param_groups[0]["weight_decay"]

    for idx, (inputs, targets) in enumerate((pbar:=tqdm(train_loader, disable=disable_tdqm))):
        pbar.set_description(f"Train - Epoch {epoch}")

        if torch.cuda.is_available():
            inputs = inputs.to(device, non_blocking=True) # https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234
            targets = targets.to(device, non_blocking=True)
        else:
            raise ValueError('gpu not available')

        
        architecture.optimizer.zero_grad()
        optimizer.zero_grad()
        if isinstance(network, (SamplingMixIn)):
            loss = 0
            for _ in range(network.n_samples):
                logits = network(inputs)
                loss += criterion(logits, targets)
                total+=targets.size(0)
                _, predicted = torch.max(logits, 1)
                correct += predicted.eq(targets).sum()
            loss /= network.n_samples
            running_loss += loss.item()
        else:
            logits = network(inputs)
            loss = criterion(logits, targets)
            running_loss += loss.item()
            total+=targets.size(0)
            _, predicted = torch.max(logits, 1)
            correct += predicted.eq(targets).sum()
        loss.backward() # could move loss inside loop. Pay with time cost in doing extra backwards, but save all forward activation memory costs -> larger batch sizes. 
        if n_proc > 1: communicate_grads(network, n_proc)

        torch.nn.utils.clip_grad_norm_(iter(optimizer.param_groups[0]['params']), max_norm=5) #TODO add clipping max_norm in args
        optimizer.step()
        #scheduler.step(epoch + idx / iters)
     
        if update_arch:
            arch_start_time = time.time()
            try:
                inputs_valid, targets_valid = next(valid_loader_iter)
                inputs_valid = inputs_valid.to(device, non_blocking=True)
                targets_valid = targets_valid.to(device, non_blocking=True)
            except StopIteration:
                inputs_valid, targets_valid = None, None
        
            architecture.step(inputs, targets, inputs_valid, targets_valid, weight_decay, momentum, optimizer, n_proc)
            arch_running_time += time.time() - arch_start_time

        acc = float(correct)/total
        train_loss = running_loss/total
        pbar.set_postfix(accuracy=acc, loss=train_loss)

    epoch_time = time.time() - start_time

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", acc, epoch)
    writer.add_scalar("1 Epoch Time/Train", epoch_time, epoch)
    writer.add_scalar("Rates/Train", scheduler.get_last_lr()[0], epoch)
    softmax_alphas_list = network.alphas_for_logging()
    alphas_list = network.arch_alphas()
    for softmax_alphas,alphas, typ in zip(softmax_alphas_list, alphas_list,["normal", "reduction"]):
        for i in range(softmax_alphas.shape[0]):
            writer.add_scalar(f"entropy-{typ}/E{i}", -torch.sum(softmax_alphas[i][softmax_alphas[i] != 0]*torch.log2(softmax_alphas[i][softmax_alphas[i] != 0])), epoch)
            for j in range(softmax_alphas.shape[1]):
                writer.add_scalar(f"alpha-{typ}-E{i}/P{j}",softmax_alphas[i][j], epoch)
                writer.add_scalar(f"True-Alpha-{typ}-E{i}/P{j}",alphas[i][j], epoch)

    out = f"Train - Epoch: {epoch} Loss: {train_loss:.06f} Acc: {acc:.06f} Time: {epoch_time:.01f} Arch Time: {arch_running_time:.01f}"
    logger.info(out)

def supernet_validation_step(network, loader, criterion, epoch, device, writer, weights=None, eval=True, how="valid"):
    assert how in ["valid", "test"]

    network.eval() if eval else network.train()
    start_time = time.time()
    correct, total, running_loss = 0, 0, 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = network(inputs, p=weights)
            loss = criterion(logits, targets)

            total += targets.size(0)
            running_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()


    total_time = time.time() - start_time
    test_loss = running_loss / (total + 1e-5)
    acc = float(correct) / (total + 1e-5)

    writer.add_scalar(f"Loss/{how}", test_loss, epoch)
    writer.add_scalar(f"Accuracy/{how}", acc, epoch)
    writer.add_scalar(f"1 Epoch Time/{how}", total_time, epoch)

    out = f"{how} - Epoch: {epoch} Loss: {test_loss:.06f} Acc: {acc:.06f} Time: {total_time:.01f}"
    logger.info(out)

    return test_loss, acc