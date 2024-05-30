##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch, random, PIL, copy, numpy as np
from os import path as osp
from shutil import copyfile

# NEEDED 

def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def get_machine_info():
    info = "Python  Version  : {:}".format(sys.version.replace("\n", " "))
    info += "\nPillow  Version  : {:}".format(PIL.__version__)
    info += "\nPyTorch Version  : {:}".format(torch.__version__)
    info += "\ncuDNN   Version  : {:}".format(torch.backends.cudnn.version())
    info += "\nCUDA available   : {:}".format(torch.cuda.is_available())
    info += "\nCUDA GPU numbers : {:}".format(torch.cuda.device_count())
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        info += "\nCUDA_VISIBLE_DEVICES={:}".format(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        info += "\nDoes not set CUDA_VISIBLE_DEVICES"
    return info


def save_checkpoint(state, filename, logger):
    if osp.isfile(filename):
        if hasattr(logger, "log"):
            logger.log(
                "Find {:} exist, delete is at first before saving".format(filename)
            )
        os.remove(filename)
    torch.save(state, filename)
    assert osp.isfile(
        filename
    ), "save filename : {:} failed, which is not found.".format(filename)
    if hasattr(logger, "log"):
        logger.log("save checkpoint into {:}".format(filename))
    return filename


def copy_checkpoint(src, dst, logger):
    if osp.isfile(dst):
        if hasattr(logger, "log"):
            logger.log("Find {:} exist, delete is at first before saving".format(dst))
        os.remove(dst)
    copyfile(src, dst)
    if hasattr(logger, "log"):
        logger.log("copy the file from {:} into {:}".format(src, dst))
