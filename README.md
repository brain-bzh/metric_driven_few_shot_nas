This is the code supplementary material for the paper **Analyzing Few-shot Neural Architecture Search in a Metric-Driven Framework**

<link>

Using this code allows :
- training a NAS-Bench-201 supernet
- creating sub-supernets splitted from a full supernet based on one of the implemented zero-cost proxies
- running diverse NAS algorithms on the generated sub-supernets

# Preparations

## Datasets

This code is compatible with the following datasets : CIFAR10, CIFAR100, ImageNet
Using the standard torch-compatible format, place the data files in their respective folders in `./datasets`

## Benchmark

This code uses the NAS-Bench-201 benchmark for architecture evaluation.
Download the benchmark from the NATS-Bench repository :

https://github.com/D-X-Y/NATS-Bench

Download the `NATS-tss-v1_0-3ffb9-simple` uncompressed archive and place all pickle files in `./nas_bench/NATSBenchTSS`

# Training a supernet

Run `train_supernet.py` to train a NAS-Bench-201 supernet.
Required arg options are as follow :
- `log_name` : name of the log folder for the training, which will be found in `./output/logs`.
- `output_name` : name of the csv output file tracking training metrics, which will be found in `./output/outputs`. This is also the name for the trained network checkpoint which will be found in `./output/checkpoints`.
- `dataset` : dataset for the training. Choices are `cifar10`, `cifar100` or `ImageNet16-120`.
- `n_epochs` : number of epochs to train for.

Please refer to args help for a brief description of other, non-required arguments, which include hyperparameter related arguments.

Example command line :
```
python train_supernet.py --log_name supernet_training --output_name supernet --dataset cifar10 --n_epochs 200
```

# Splitting the supernet into sub-supernets

Run `split.py` to split the supernet into any number of sub-supernets. This script will create checkpoints for the requested sub-supernets in `./output/checkpoints` with the following name format `{metric_name}{random_seed}_subsupernet{index}`.
Required arg options are as follow :
- `log_name` : name of the log folder for the splitting, which will be found in `./output/logs`.
- `output_name` : name of the csv output file tracking splitting metrics, which will be found in `./output/outputs`.
- `dataset` : dataset for the splitting. Choices are `cifar10`, `cifar100` or `ImageNet16-120`.

Please refer to args help for a brief description of other, non-required arguments, which include tuning the number of obtained sub-supernets, using a specific zero-cost proxy, warmup settings and other hyperparameter related arguments.

Example command line :
```
python split.py --log_name split_gm --output_name split_gm --dataset cifar10 --metric gradientmatching --target_n_subsupernets 8
```

**NB** : the format of the arch parameter matrix obtained from splitting (which is integral to the sub-supernet) is different from the format used for NAS search configs. Therefore, to bridge the gap between the two formats, we add an `export_arch_params.py` script for easier conversion from the former to the latter. This script will parse through every checkpoint in `./output/checkpoints` and write the converted arch parameter matrix in the output file.
Its required arg options are as follow :
- `log_name` : name of the log folder, which will be found in `./output/logs`.
- `output_name` : name of the csv output file, which will be found in `./output/outputs`. This output file will contain the converted arch parameter matrices, which can then be copied to the appropriate yaml config file.
- `metrics` : name of the metrics for which to convert arch parameters (these should be present in the checkpoint name).
- `n_nets` : number of sub-supernets that exist for each metric.
- `rand_seed` : the random seed that was used when generating the sub-supernets.

Example command line :
```
python export_arch_params.py --log_name export --output_name export --metrics gradientmatching gradnorm --n_nets 8 --rand_seed 0
```

# Running one-shot NAS algorithms on generated sub-supernets

Run `oneshot_search.py` to run a NAS algorithm with specified hyperparameters using the specified sub-supernet to reduce the search space.
Required arg options are as follow :
- `config_name` : name of the yaml config file which should be located in `.pkg/search_configs`.

Example command line :
```
python oneshot_search.py template.yml
```

For an example of a config file that runs DARTS on a sub-supernet obtained from the gradientmatching metric, see `.pkg/search_configs/template.yml`.
Notably, the `default:arg` argument contains the arch parameter matrix in the binary format (0=deactivated operation, 1=activated operation).
The `training:epochs` argument controls how many epochs to search for.
Set desired dataset using the `dataset` argument.

For transparency, we have included every config file which we used to run our experiments in `.pkg/search_configs`. This includes 9 metrics across 3 datasets and 4 one-shot NAS algorithms (DARTS, DARTS 2nd, SNAS, DSNAS).