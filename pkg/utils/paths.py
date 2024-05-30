from pathlib import Path

CHECKPOINTS_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/output/checkpoints'
OUTPUTS_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/output/outputs'
LOG_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/output/logs'

CIFAR10_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/datasets/CIFAR10'
CIFAR100_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/datasets/CIFAR100'
IMAGENET16_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/datasets/ImageNet'
CONFIGS_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/datasets/configs'

NASBENCH_PATH = str(Path(__file__).resolve().parent.parent.parent) + '/nas_bench/NATSBenchTSS'

DATASET_PATHS = {'cifar10' : CIFAR10_PATH, 'cifar100' : CIFAR100_PATH, 'ImageNet16-120' : IMAGENET16_PATH}

SEARCH_CONFIGS_PATH = str(Path(__file__).resolve().parent.parent) + '/search_configs'