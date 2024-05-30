import datetime
import random
import functools

from pkg.utils.paths import *
from pkg.search_pkg.utils import *
from pkg.search_pkg.datasets import get_datasets, get_iterators
from pkg.search_pkg.networks import RunManager
from pkg.search_pkg.training import supernet_training
from pkg.search_pkg.arch_selection import optimal_path
from pkg.lib.natsbench import create

def main(args):
    args = load_yaml(args.config_name) 
    for k, v in recursive_items(args):
        if k == 'subset':    
            assert 0  < v <= 1
        if k == 'valid_split':
            assert 0 < v <= 1
        if k == 'space':
            assert isinstance(v, list)
        if k == 'train_or_valid' and v == 'train':
            assert args['default']['dataloader']['valid_split'] == 1

    # Multi-processing initialization
    n_proc, rank, gpu = 1, 0, 0

    # Paths & Logger init
    TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_path = LOG_PATH + '/{:}/{:}'.format(args['output'], TIME)
    TRAIN_CHECKPOINTS = CHECKPOINTS_PATH + '/{:}/{:}/train'.format(args['output'], TIME)
    WARMUP_CHECKPOINTS = CHECKPOINTS_PATH + '/{:}/{:}/warmup'.format(args['output'], TIME)
    DATA_PATH = DATASET_PATHS[args['dataset']]

    if is_main_process():
        path_exist(log_path, TRAIN_CHECKPOINTS, WARMUP_CHECKPOINTS)
        save_yaml(args, log_path)

    create_logger(log_path, TIME)
    logger.info(args)
    logger.info("Number of processes: {}".format(n_proc))

    writer = SecondaryWriter()

    # global hyperparams
    if args.get("seed") is not None:
        seed = args.get("seed")
    else:
        seed = int(random.uniform(0, 1000))
    logger.info(f"Seed: {seed}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    default_args = args["default"]
    train_data, test_data, xshape, class_num = get_datasets(args['dataset'], DATA_PATH, **default_args.get("dataset", {}))
    args['base_config']['network']['config']['num_classes'] = class_num

    #default
    train_loader, valid_loader, test_loader = get_iterators(train_data, test_data, **default_args["dataloader"])
    if default_args.get("arch"):
        default_arch = np.array(default_args["arch"])
    else:
        default_arch = [np.ones((args.get("num_edges"), args.get("num_ops")))]*2 # TODO temporary, won't provide correct default for NasBench

    # Warm-Up
    warmup_args = args["warmup"]
    warm_up_save = None

    if warmup_args and warmup_args.get("epochs", 0) > 0:

        warm_up_save = os.path.join(WARMUP_CHECKPOINTS, 'model.pt')

        start_time = time.time()
        logger.info('============ Warm Up Begin ============')

        rm = RunManager(
            default_arch,
            args["base_config"]
        )
        if n_proc > 1:
            torch.manual_seed(seed+rank)
            torch.cuda.manual_seed_all(seed+rank)
            np.random.seed(seed+rank)
        rm.custom_optimization_init(warmup_args)
        logger.info(f"Model Param Size: {count_parameters_in_MB(rm.network)} MB")
        logger.info(f"Num Params: {sum(p.numel() for p in rm.network.parameters() if p.requires_grad)}")

        train_args = {
            "network": rm.network,
            "architecture": rm.architecture,
            "criterion": rm.loss,
            "optimizer": rm.optimizer,
            "scheduler": rm.scheduler,
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "start_epoch": rm.start_epoch,
            "end_epoch": warmup_args['epochs'],
            "writer": writer,
            "device": rm.device,
            "disable_tdqm": args["disable_tdqm"],
            "n_proc": n_proc
        }
        
        supernet_training(**train_args)

        rm.start_epoch = warmup_args['epochs']

        if rank == 0: rm.save_all(warm_up_save, {'args': args})
        if is_dist_avail_and_initialized(): dist.barrier() # hold other threads here until save is done.

        logger.info(f"Total Warm-Up Time: {time.time()-start_time:.01f}")
        
        torch.cuda.empty_cache()
    
    splitting_args = args['splitting']
    arch_param_keys = ["parent", "arch"]

    arch_parameter_leaves = [dict(zip(arch_param_keys, [warm_up_save, default_arch]))]
    level = -1
    
    logger.info('============ Training Begin ============')

    api = create(NASBENCH_PATH, 'tss', fast_mode=True, verbose=False)
    training_args = args["training"]
    start_time = time.time()

    trained_arch_parameter_leaves = []
    for index, arch_parameter_dict in enumerate(arch_parameter_leaves):
        logger.info(f'============ Train Architecture: {index} ============')

        if rank == 0:
            SAVE_PATH = TRAIN_CHECKPOINTS + f"/level{level+1}_node{index}"
            MODEL_SAVE_PATH = SAVE_PATH + '_best_model.pt'

        if arch_parameter_dict["parent"] is not None:
            rm = RunManager.load(arch_parameter_dict["parent"], arch_parameter_dict["arch"])
        else:
            rm = RunManager(
                arch_parameter_dict["arch"],
                args["base_config"]
            )
        rm.custom_optimization_init(training_args)

        logger.info(f"Model Param Size: {count_parameters_in_MB(rm.network)} MB")
        logger.info(f"Num Params: {sum(p.numel() for p in rm.network.parameters() if p.requires_grad)}")

        logger.info(f'Arch parameters : \n{rm.network.arch_parameters}') # edge/operation matrix
        supernet_training(
            rm.network,
            rm.architecture,
            rm.loss,
            rm.optimizer,
            rm.scheduler,
            train_loader,
            valid_loader,
            test_loader=test_loader if args["test"] else None,
            start_epoch=rm.start_epoch,
            end_epoch=training_args['epochs'] + rm.start_epoch, # need cumulative epochs
            writer=writer,
            device=rm.device,
            optimal_path=functools.partial(optimal_path, api=api, dataset=args["dataset"], mask=rm.network.arch_parameters) if not args["fast"] and rank==0 else None,
            genotype=(rank==0),
            arch_train=True, # update arch_params
            save=functools.partial(rm.save_all, MODEL_SAVE_PATH, {'args': args}) if rank==0 else None,
            disable_tdqm=args["disable_tdqm"],
            n_proc=n_proc,
        )
        rm.start_epoch += training_args["epochs"]
        if rank == 0: trained_arch_parameter_leaves.append({'path': MODEL_SAVE_PATH, 'arch': rm.network.arch_parameters})

    logger.info(f"Total Training Time: {time.time()-start_time:.01f}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser("NAS algorithms")
    parser.add_argument("--config_name", type=str, default='template.yml', help="Name of YAML config file.")

    args = parser.parse_args()
    main(args)