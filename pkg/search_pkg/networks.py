import logging
import torch
from torch.autograd import Variable
import numpy as np
import torch.optim
import functools
import importlib
from copy import deepcopy
import torch.distributed as dist
import torch.nn.functional as F
from .train_utils import TempScheduler, RegScheduler
from .models.cell_searchs.generic_model import GenericNAS201Model
from .models.cell_searchs.search_cells import NAS201SearchCell as SearchCell
from .utils import flat_concat

logger = logging.getLogger(__name__)

reg_schedulers = {
    'TempScheduler' : TempScheduler,
    'RegScheduler' : RegScheduler
}

class RunManager:
    def __init__(self, arch_parameter, base_config) -> None:
        self.arch_parameter = arch_parameter
        self.base_config = base_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0

        self.network = self._build_network(**self.base_config["network"])
        self.optimizer = self._build_optimizer(**self.base_config["optimizer"])
        self.scheduler = self._build_scheduler(**self.base_config["scheduler"])
        self.architecture = self._build_architecture(**self.base_config["architecture"])
        self.loss = self._build_loss(**self.base_config["loss"])

    @classmethod
    def load(cls, path, arch=None):
        logger.info(f"Loading Checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cuda")
        if arch is not None:
            arch_params = arch
        else:
            arch_params = checkpoint['arch_parameters']
        rm = cls(arch_params, checkpoint['args']['base_config'])
        rm._load_state(checkpoint)

        return rm
    
    def custom_optimization_init(self, args):
        #temp scheduler for snas
        if hasattr(self.network, 'lam_scheduler') and args.get("temperature"):
            self.network.reinit_lam_scheduler(**args.get("temperature"))
            logger.info(f"New Temperature: {self.network.lam_scheduler}")

        #regularization scheduler for arch
        if hasattr(self.architecture, 'reg_scheduler') and args.get("regularization"):
            self.architecture.reinit_reg_scheduler(**args.get("regularization"))
            logger.info(f"New Regularization: {self.architecture.reg_scheduler}")

        # check optimizer
        if args.get("optimizer"):
            self.optimizer = self._build_optimizer(**args.get("optimizer"))
            logger.info(f"New Optimizer: {self.optimizer}")
        if args.get("scheduler"):
            self.scheduler = self._build_scheduler(**args.get("scheduler"))
            logger.info(f"New scheduler: {self.scheduler}")
        if args.get("loss"):
            self.loss = self._build_loss(**args.get("loss"))
            logger.info(f"New Loss: {self.loss}")

    def _build_network(self, name, config):
        module = importlib.import_module('pkg.search_pkg.networks')
        network = getattr(module, name)(self.arch_parameter, **config)
        network.to(self.device)
        return network

    def _build_optimizer(self, name, config):
        net_parameters = []
        for n, p in self.network.named_parameters():
            if (not n == "_arch_alphas") and (not n == 'arch_parameters') and (not n == "alphas_normal") and (not n == "alphas_reduce"):
                net_parameters.append(p)

        return getattr(torch.optim, name)(net_parameters, **config)

    def _build_scheduler(self, name, config):
        return getattr(torch.optim.lr_scheduler, name)(self.optimizer, **config)

    def _build_architecture(self, name, config):
        module = importlib.import_module('pkg.search_pkg.networks')
        return getattr(module, name)(self.network, **config)

    def _build_loss(self, name, config):
        return functools.partial(getattr(torch.nn.functional, name), **config)

    def _load_state(self, checkpoint):
        # network
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.network.set_arch_alphas(checkpoint["_arch_alphas"] if isinstance(checkpoint["_arch_alphas"], list) else list(checkpoint["_arch_alphas"])) # should be backward compatible

        #architecture
        self.architecture.optimizer.state = checkpoint["arch"]["optimizer_state_dict"]["state"]
        self.architecture.optimizer.param_groups = checkpoint["arch"]["optimizer_state_dict"]["param_groups"]
        #self.architecture.optimizer.load_state_dict(checkpoint["arch"]["optimizer_state_dict"])

        if hasattr(self.network, 'lam_scheduler'):
             self.network.lam_scheduler = TempScheduler.load_state_dict(checkpoint["lam_scheduler"])

    def save_all(self, path, config_append):
        config = {
            "arch_parameters": self.network.arch_parameters_iter(), # list of paramters
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.start_epoch,
            "_arch_alphas": self.network.arch_alphas(), # list alphas
            "lam_scheduler": self.network.lam_scheduler.state_dict() if hasattr(self.network, 'lam_scheduler') else None, 
            "arch":{
                "reg_scheduler": self.architecture.reg_scheduler.state_dict() if hasattr(self.architecture, 'regularization') else None, 
                "optimizer_state_dict": {
                   'state': self.architecture.optimizer.state,
                   'param_groups': self.architecture.optimizer.param_groups
                }
            },
        }
        if config_append:
            for key, value in config_append.items():
                if config.get(key) is not None:
                    raise ValueError("Overwriting existing key!")
                config[key] = value
        torch.save(config, path)

class Architecture():

    def __init__(self, network, loss, optimizer, second_order=True, regularization=None):
        self.network = network
        self.loss = self._build_loss(**loss)
        self.optimizer = self._build_optimizer(**optimizer)
        self.second_order = second_order
        if regularization:
            self.reg_scheduler = self._build_reg_scheduler(**regularization)
            self.reg = self.reg_scheduler.curr_lam

    def _build_optimizer(self, name, config):
        return getattr(torch.optim, name)(self.network.arch_alphas(), **config)
  
    def _build_loss(self, name, config):
          return functools.partial(getattr(torch.nn.functional, name), **config)

    def save_model(checkpoint: dict, path: str):
          torch.save(checkpoint, path)

    def _build_reg_scheduler(self, name, config):
          return reg_schedulers[name](**config)
  
    def reinit_reg_scheduler(self, name ,config):
          self.reg_scheduler = self._build_reg_scheduler(name, config)
          self.reg = self.reg_scheduler.curr_lam

    def _loss(self, logits, target):
        if hasattr(self, "regularization"):
            p = self.network.get_probabilities()
            entropy = torch.sum(-p*torch.log(p+1e-5))
            loss = self.loss(logits, target) + self.reg*entropy    
        else:
            loss = self.loss(logits, target)
        return loss
    
    def full_loss(self, input, target):
        if hasattr(self.network, "n_samples"):
            loss = 0
            for _ in range(self.network.n_samples):
                logits = self.network(input)
                loss += self._loss(logits, target)
            loss /= self.network.n_samples
        else:
            logits = self.network(input)
            loss = self._loss(logits, target)
        return loss


    # first order update
    def _step_fo(self, input_valid, target_valid, n_proc):
        loss = self.full_loss(input_valid, target_valid)
        loss.backward()
        if n_proc > 1: self.communicate_alphas(self.network, n_proc)
        self.optimizer.step()
    
    def network_parameters(self, net=None):
        if net is None:
            net = self.network
        params = list(net._stem.parameters()) + list(net._cells.parameters())
        params += list(net.lastact.parameters()) + list(net.global_pooling.parameters())
        params += list(net.classifier.parameters())
        return params

    # second order update
    def _step_so(self, input_train, target_train, input_valid, target_valid, net_wdecay, net_momentum, net_optimizer, eta = 0.01):
        net_parameters = flat_concat(self.network_parameters()).data
        moment = flat_concat([net_optimizer.state[p]['momentum_buffer'] if p.grad is not None else torch.zeros_like(p).data for p in self.network_parameters()])
        loss = self.full_loss(input_train, target_train) 
        grads_all = torch.autograd.grad(loss, self.network_parameters(), allow_unused=True)
        model_params = list(self.network_parameters())
        grads = [] # set to zero the gradients of the parameters that do not appear in the graph
        for i, grad in enumerate(grads_all):
            if grad is None: 
                grad = torch.zeros_like(model_params[i].data)
            assert grad.shape == model_params[i].shape
            grads.append(grad)
        grads = flat_concat(grads).data
        updated_parameters = net_parameters.sub(moment + grads + net_wdecay * net_parameters, alpha=eta)
        # create an auxiliary network with the updated parameters
        aux_net = deepcopy(self.network).cuda()
        net_dict = aux_net.state_dict()
        aux_params, index = {}, 0
        for n, p in self.network.named_parameters():
            if (not n == "_arch_alphas") and (not n == 'arch_parameters'):
                l = np.prod(p.size())
                aux_params[n] = updated_parameters[index: index + l].reshape(p.size())
                index += l
        net_dict.update(aux_params)
        aux_net.load_state_dict(net_dict)
        # compute gradients of the loss of the updated model
        logits = aux_net(input_valid)
        loss_fo = self.loss(logits, target_valid)  
        loss_fo.backward()
        grad_alpha = aux_net._arch_alphas.grad
        grad_w = [v.grad.data if v.grad is not None else torch.zeros_like(v).data for v in self.network_parameters(aux_net)]
        implicit_grad = self._finite_difference_appox(grad_w, input_train, target_train)[0]
        # expression (7) from paper 
        grad_alpha.data.sub_(implicit_grad, alpha=eta)
        # set the computed update as the model gradient
        self.network._arch_alphas.data.copy_(grad_alpha.data)
        self.optimizer.step()

    def step(self, input_train, target_train, input_valid, output_valid, net_wdecay=None, net_momentum=None, net_optimizer = None, n_proc=1):
        if self.second_order:
            return self._step_so(input_train, target_train, input_valid, output_valid, net_wdecay, net_momentum, net_optimizer)
        return self._step_fo(input_valid, output_valid, n_proc)


    def _finite_difference_appox(self, vector, input, target):
        epsilon = 0.001 / flat_concat(vector).norm() # value suggested in the DARTS paper
        for p, v in zip(self.network_parameters(), vector):
            p.data.add_(v, alpha=epsilon)
        loss = self.full_loss(input, target)
        grads_p = torch.autograd.grad(loss, self.network.arch_alphas())

        for p, v in zip(self.network_parameters(), vector):
            p.data.sub_(v, alpha = 2 * epsilon)
        loss = self.full_loss(input, target)
        grads_n = torch.autograd.grad(loss, self.network.arch_alphas())

        for p, v in zip(self.network_parameters(), vector):
            p.data.add_(v, alpha=epsilon)

        return [(x - y).div_(2 * epsilon) for x, y in zip(grads_p, grads_n)]

    def communicate_alphas(self, network, n_proc):
        dist.all_reduce(network._arch_alphas.grad, op=dist.ReduceOp.SUM)
        network._arch_alphas.grad /= n_proc

class SamplingArchitecture(Architecture):
    def __init__(self, network, loss, optimizer, train_or_valid, threshold, method, mixture_weights=None, second_order=False, regularization=None):
        super().__init__(network, loss, optimizer, second_order, regularization)
        assert train_or_valid in ["train", "valid"]
        self.train_or_valid = train_or_valid
        self.threshold = threshold
        self.mixture_weights = mixture_weights
        self.gradient = getattr(self, method)
    
        
    def _path_gradient(self, inputs, targets, net_optimizer = None, **kwargs):
        grad_alpha = torch.zeros_like(self.network._arch_alphas)
        for _ in range(self.network.n_samples):
            self.optimizer.zero_grad()
            net_optimizer.zero_grad()
            logits = self.network(inputs)
            loss = self.loss(logits, targets)
            self.network.op_prob.grad = torch.zeros_like(self.network.op_prob) # zero out train backward
            (loss+self.network.logl).backward()
            cost = self.network.op_prob.grad.data.sum(-1) 
            self.network._arch_alphas.grad.data.mul_(cost.view(-1,1))
            grad_alpha += self.network._arch_alphas.grad.data
        grad_alpha /= self.network.n_samples
        self.network._arch_alphas.grad.data = grad_alpha

    def _gf_lds(self, inputs, targets, n_proc, **kwargs):
        losses = []
        paths = []
        with torch.no_grad():
            samples = self.generate_lds(self.network.n_samples)
            for w in samples:
                logits = self.network(inputs, w)
                losses.append(self.loss(logits, targets).detach()) # we don't need autograd of loss,
                paths.append(self.network.op_prob)
        losses = torch.tensor(losses).cuda()
        loss = torch.mean(losses).cuda()
        if n_proc > 1: torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        self.network._arch_alphas.grad = sum([paths[k]*(losses[k]-loss) for k in range(self.network.n_samples)])/(self.network.n_samples-1) # this wasn't loorf!

    
    def _derivativefree(self, inputs, targets, n_proc, **kwargs):
        losses = []
        paths = []
        with torch.no_grad():
            for _ in range(self.network.n_samples):
                logits = self.network(inputs)
                losses.append(self.loss(logits, targets).detach()) # we don't need autograd of loss,
                paths.append(self.network.op_prob)
        losses = torch.tensor(losses).cuda()
        loss = torch.mean(losses).cuda()
        if n_proc > 1: torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        self.network._arch_alphas.grad = sum([paths[k]*(losses[k]-loss) for k in range(self.network.n_samples)])/self.network.n_samples
    
    def _reinforce(self, inputs, targets, **kwargs):
        grad_alpha = torch.zeros_like(self.network._arch_alphas)
        for _ in range(self.network.n_samples):
            self.optimizer.zero_grad()
            logits = self.network(inputs)
            loss = self.loss(logits, targets).detach()
            (self.network.logl).backward()
            grad_alpha += self.network._arch_alphas.grad.data*loss
        grad_alpha /= self.network.n_samples
        self.network._arch_alphas.grad.data = grad_alpha

    def _reinforce_loo(self, inputs, targets, n_proc, **kwargs):
        losses = []
        paths = []
        with torch.no_grad():
            for _ in range(self.network.n_samples):
                logits = self.network(inputs)
                losses.append(self.loss(logits, targets).detach()) # we don't need autograd of loss,
                paths.append(self.network.op_prob)
        losses = torch.tensor(losses).cuda()
        loss = torch.mean(losses).cuda()
        if n_proc > 1: torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        p = torch.softmax(self.network._arch_alphas, dim=1)
        self.network._arch_alphas.grad = sum([(paths[k]-p)*(losses[k]-loss) for k in range(self.network.n_samples)])/(self.network.n_samples-1)

    def _shrimple_loo(self, inputs, targets, n_proc, **kwargs):
        losses = []
        paths = []
        with torch.no_grad():
            for _ in range(self.network.n_samples):
                logits = self.network(inputs)
                losses.append(self.loss(logits, targets).detach()) # we don't need autograd of loss,
                paths.append(self.network.op_prob)
        losses = torch.tensor(losses).cuda()
        loss = torch.sum(losses).cuda()


        mean_loo = (loss-losses)/(self.network.n_samples-1)
        if n_proc > 1: torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        p_hat = sum(paths)/self.network.n_samples
        self.network._arch_alphas.grad = sum([(paths[k]-p_hat)*(losses[k]-mean_loo[k]) for k in range(self.network.n_samples)])/self.network.n_samples


    def _derivativefree_loo(self, inputs, targets, n_proc, **kwargs):
        losses = []
        paths = []
        with torch.no_grad():
            for _ in range(self.network.n_samples):
                logits = self.network(inputs)
                losses.append(self.loss(logits, targets).detach()) # we don't need autograd of loss,
                paths.append(self.network.op_prob)
            if hasattr(self.network, "reset_sampling"):
                self.network.reset_sampling()
        losses = torch.tensor(losses).cuda()
        loss = torch.sum(losses).cuda()
        mean_loo = (loss-losses)/(self.network.n_samples-1)
        if n_proc > 1: torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        self.network._arch_alphas.grad = sum([paths[k]*(losses[k]-mean_loo[k]) for k in range(self.network.n_samples)])/self.network.n_samples

    def _mixture(self, inputs, targets, n_proc, **kwargs):
        losses = []
        paths = []
        with torch.no_grad():
            for _ in range(self.network.n_samples):
                logits = self.network(inputs)
                losses.append(self.loss(logits, targets).detach()) # we don't need autograd of loss,
                paths.append(self.network.op_prob)
        losses = torch.tensor(losses).cuda()
        mean_loss = torch.mean(losses).cuda()
        P = torch.zeros_like(self.network._arch_alphas)
        for e in range(self.network.arch_parameters.shape[0]):
            P[e][self.network.arch_parameters[e] == 1] = torch.softmax(self.network._arch_alphas[e][self.network.arch_parameters[e] == 1], dim=-1)
        if n_proc > 1: torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.AVG)
        P_hat = sum([paths[k] for k in range(self.network.n_samples)])/self.network.n_samples
        s_1 = sum([paths[k]*losses[k] for k in range(self.network.n_samples)])/self.network.n_samples
        self.network._arch_alphas.grad = self.mixture_weights[0]*s_1 + self.mixture_weights[1]*mean_loss*P + self.mixture_weights[2]*mean_loss*P_hat

    def generate_lds(self, n_samples):
        p_o = torch.zeros_like(self.network._arch_alphas)
        samples = []
        p = np.zeros((self.network.n_edges, self.network._layerN))
        for e in range(self.network.arch_parameters.shape[0]):
                p_o[e][self.network.arch_parameters[e] == 1] = torch.softmax(self.network._arch_alphas[e][self.network.arch_parameters[e] == 1], dim=-1)
                p[e]= p_o[e].clone().detach().cpu().numpy()
        p_hat = p   
        for i in range(n_samples):
            weights = torch.zeros_like(self.network._arch_alphas)
            for e in range(self.network.arch_parameters.shape[0]):
                #alpha = np.minimum(p[e]/(1-p[e]), (1-p[e])/p[e])
                alpha = 1
                q = np.minimum(1, np.maximum(p[e]*(1 + alpha) - alpha * p_hat[e], 0))
                k = np.random.choice(5, 1, p=q/np.sum(q))
                weights[e, k] = 1
            samples.append(weights)
            p_hat = ((i)*p_hat + weights.cpu().numpy())/(i+1)
        return samples
        
    def step(self, input_train, target_train, input_valid, output_valid, net_wdecay=None, net_momentum=None, net_optimizer = None, n_proc=1):
        if torch.sum(self.network.arch_parameters) < self.network._num_edge + 1: # if we have already prunned
            return
        if self.train_or_valid == "valid": # if valid, need to compute loss and backprop, train we already backproped in train loop
            inputs = input_valid
            targets = output_valid
        else:
            inputs = input_train
            targets = target_train
        self.optimizer.zero_grad()
        self.gradient(inputs=inputs, targets=targets, net_optimizer=net_optimizer, n_proc=n_proc)
        if n_proc > 1: self.communicate_alphas(self.network, n_proc)
        self.optimizer.step()

    def communicate_alphas(self, network, n_proc):
        dist.all_reduce(network._arch_alphas.grad, op=dist.ReduceOp.SUM)
        network._arch_alphas.grad /= n_proc
        dist.all_reduce(network.logl, op=dist.ReduceOp.SUM)
        network.logl /= n_proc


class NetworkDarts(GenericNAS201Model):
    def __init__(self, arch_parameters, C, N, max_nodes, num_classes, space, affine,
                    track_running_stats=True, depth=-1, use_stem=True):
        super(NetworkDarts, self).__init__(C, N, max_nodes, num_classes, space,
                                                    affine, track_running_stats, depth, use_stem)
        
        self.arch_parameters = torch.tensor(arch_parameters).clone().detach().cuda()

    def get_probabilities(self):
        p = torch.zeros_like(self.arch_parameters, dtype=self._arch_alphas.dtype, device=self._arch_alphas.device)
        for e in range(self.arch_parameters.shape[0]):
            p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1], dim=-1)
        return p

    def forward(self, inputs, p=None):
        if p is None:
            weights = self.get_probabilities()
        else:
            weights = p
        feature = self._stem(inputs)

        for i, cell in enumerate(self._cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, weights)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.reshape(out.size(0), -1)
        logits = self.classifier(out)

        return logits

    def alphas_for_logging(self):
        return [self.get_probabilities()]

    
class NetworkRethinkingDarts(NetworkDarts):
    def __init__(self, arch_parameters, C, N, max_nodes, num_classes, space, affine,
                    track_running_stats=True, depth=-1, use_stem=True):
        super().__init__(arch_parameters, C, N, max_nodes, num_classes, space, affine,
                    track_running_stats, depth, use_stem)

        self.candidate_flags = torch.tensor(len(self.arch_parameters) * [True], requires_grad=False, dtype=torch.bool).cuda()

    def project_op(self, eid, opid):
        self.arch_parameters[eid] = torch.zeros_like(self.arch_parameters[eid]) # zero out row
        self.arch_parameters[eid][opid] = 1 ## hard by default
        self.candidate_flags[eid] = False

    def genotype(self): # think this is redundant, can likely use the base class genotype
        from pkg.lib.models.cell_searchs.genotypes import Structure
        theta = self.get_probabilities()
        genotypes = []
        final_arch = torch.zeros_like(self.arch_parameters)
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = theta[ self.edge2index[node_str] ]
                    final_arch[self.edge2index[node_str], weights.argmax().item()] = 1
                    op_name = self._op_names[ weights.argmax().item() ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return Structure( genotypes ), genotypes, final_arch

class SamplingMixIn:
    pass

class NetworkSnas(GenericNAS201Model, SamplingMixIn):
    def __init__(self, arch_parameters, C, N, max_nodes, num_classes, space, affine, temperature,
                    track_running_stats=True, depth=-1, use_stem=True, forward_samples=1):
        super(NetworkSnas, self).__init__(C, N, max_nodes, num_classes, space,
                                                    affine, track_running_stats, depth, use_stem)

        self.lam_scheduler = self._build_lam_scheduler(**temperature)
        self.lam = self.lam_scheduler.curr_lam
        self.arch_parameters =  torch.tensor(arch_parameters, dtype=torch.float).clone().detach().cuda() # https://discuss.pytorch.org/t/why-model-to-device-wouldnt-put-tensors-on-a-custom-layer-to-the-same-device/17964/11 could use register buffer
        self._arch_alphas = torch.nn.Parameter(torch.zeros(6, 5).normal_(1, 0.01).requires_grad_())
        self.n_samples = forward_samples

    def _build_lam_scheduler(self, name, config):
        return reg_schedulers[name](**config)
    
    def reinit_lam_scheduler(self, name, config):
        self.lam_scheduler = self._build_lam_scheduler(name, config)
        self.lam = self.lam_scheduler.curr_lam

    def get_probabilities(self):
        return self._get_gumbel_softmax()

    def _get_gumbel_softmax(self):
        p = torch.zeros_like(self.arch_parameters, device='cuda')
        for e in range(p.shape[0]):
            p_e = self._get_gumbel_softmax_edge(e)
            p[e][self.arch_parameters[e] != 0] = p_e
        return p

    def _get_gumbel_softmax_edge(self, e):
        log_alpha = self._arch_alphas[e][self.arch_parameters[e] != 0]
        while True:
            gumbels = torch.nn.functional.gumbel_softmax(log_alpha, tau=self.lam)
            if torch.isinf(gumbels).any():
                continue
            else: break
        return gumbels

    # this can go into the parent class
    def forward(self, inputs, p=None):
        if p is None:
            weights = self.get_probabilities()
        else:
            weights = p
        feature = self._stem(inputs)

        for i, cell in enumerate(self._cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, weights)
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.reshape(out.size(0), -1)
        logits = self.classifier(out)

        return logits

    def alphas_for_logging(self):
        p = torch.zeros_like(self.arch_parameters, dtype=self._arch_alphas.dtype, device=self._arch_alphas.device)
        for e in range(self.arch_parameters.shape[0]):
            p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1], dim=-1)
        return [p]

    def genotype(self):
        # just to match SNAS
        from pkg.lib.models.cell_searchs.genotypes import Structure
        theta = torch.softmax(self._arch_alphas, dim=-1) * self.arch_parameters
        genotypes = []
        final_arch = torch.zeros_like(self._arch_alphas)
        for i in range(1, self._max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = theta[ self.edge2index[node_str] ]
                    final_arch[self.edge2index[node_str], weights.argmax().item()] = 1
                    op_name = self._op_names[ weights.argmax().item() ]
                xlist.append((op_name, j))
            genotypes.append( tuple(xlist) )
        return Structure( genotypes ), genotypes, final_arch


# single path sampling network
class NetworkRP(GenericNAS201Model, SamplingMixIn):
    def __init__(self, arch_parameters, C, N, max_nodes, num_classes, space, affine,
                track_running_stats=True, depth=-1, use_stem=True, forward_samples=1, independence=True, alpha=1):
        super(NetworkRP, self).__init__(C, N, max_nodes, num_classes, space,
                                        affine, track_running_stats, depth, use_stem)
        
        self.arch_parameters = torch.tensor(arch_parameters).clone().detach().cuda()
        self._arch_alphas = torch.nn.Parameter(
            1 + 1e-2 * torch.randn(self.n_edges, self._layerN).cuda()
        )
        # if independence:
        #     self._arch_alphas = torch.nn.Parameter(
        #             1 + 1e-2 * torch.randn(self.n_edges, self._layerN).cuda()
        #     )
        # else:
        #     self._arch_alphas = torch.nn.Parameter(
        #             1 + 1e-2 * torch.randn(self.n_edges+self._layerN, self._layerN).cuda()
        #     )
        self.logl = 0
        self.op_prob = torch.zeros_like(self._arch_alphas)
        self.n_samples = forward_samples
        self.independence = independence
        self.get_probabilities = self._set_sampling()
        self.sample_counter = 0
        self.sample_memory = np.zeros_like(self._arch_alphas.detach().cpu())
        self.alpha = alpha

    def _set_sampling(self):
        if self.independence:
            sampler = lambda : self.ind_probabilities()
        else:
            # sampler = lambda : self.dep_probabilities()
            sampler = lambda : self.db_sample_probabilities()
        return sampler

    def ind_probabilities(self):
        p = torch.zeros_like(self._arch_alphas)
        weights = torch.zeros_like(self._arch_alphas)
        logl = torch.tensor([0.0], device=self._arch_alphas.device)
        for e in range(self.arch_parameters.shape[0]):
            p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1], dim=-1)
            probs = p[e].clone().detach().cpu().numpy()
            k = np.random.choice(5, 1, p=probs/np.sum(probs))
            weights[e, k] = 1
            logl += torch.log(p[e, k])
        return weights, logl
    
    def db_sample_probabilities(self):
        p = torch.zeros_like(self._arch_alphas)
        weights = torch.zeros_like(self._arch_alphas)
        logl = torch.tensor([0.0], device=self._arch_alphas.device)
        for e in range(self.arch_parameters.shape[0]):
            p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1], dim=-1)
            probs = p[e].clone().detach().cpu().numpy()

            if self.sample_counter > 0:
                probs = np.clip(probs * (1 + self.alpha) - self.alpha * self.sample_memory[e] / self.sample_counter, 0, 1)

            k = np.random.choice(5, 1, p=probs/np.sum(probs)) # normalize by sum.
            weights[e, k] = 1
            self.sample_memory[e, k] += 1
            logl += torch.log(p[e, k])
        
        self.sample_counter += 1 # track which sample we are on
        return weights, logl
    
    def reset_sampling(self):
        self.sample_counter = 0
        self.sample_memory = np.zeros_like(self.sample_memory)

    def dep_probabilities(self):
        p = torch.zeros_like(self._arch_alphas)
        weights = torch.zeros_like(self._arch_alphas)
        self.op_dep = torch.zeros(5, 5)
        logl = torch.tensor([0.0], device=self._arch_alphas.device)
        ops = []
        ref = [-1, -1, 0, -1, 0, 2] # edges that we condition on
        for e, prev in enumerate(ref):
            if prev < 0:
                p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1], dim=-1)
            else:
                p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1]+self._arch_alphas[6+ops[prev]][self.arch_parameters[e] == 1], dim=-1)
            probs = p[e].clone().detach().cpu().numpy()
            k = np.random.choice(5, 1, p=probs/np.sum(probs))
            weights[e, k] = 1
            ops.append(int(k))
            logl += torch.log(p[e, k])
        weights[self.n_edges+ops[0], ops[2]]=1
        weights[self.n_edges+ops[0], ops[4]]=1
        weights[self.n_edges+ops[2], ops[5]]=1
        
        return weights, logl


    def forward(self, inputs, p=None):
        if p is None:
            self.op_prob, self.logl = self.get_probabilities()
        else:
            # avoid calling self.get_probabilities() to keep same architecture
            self.op_prob = p
        
        self.op_prob.requires_grad_()
        feature = self._stem(inputs)

        for i, cell in enumerate(self._cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, self.op_prob[:self.n_edges,:])
            else:
                feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.reshape(out.size(0), -1)
        logits = self.classifier(out)

        return logits

    @torch.no_grad()
    def alphas_for_logging(self):
        p = torch.zeros_like(self._arch_alphas)
        for e in range(self.arch_parameters.shape[0]):
            p[e][self.arch_parameters[e] == 1] = torch.softmax(self._arch_alphas[e][self.arch_parameters[e] == 1], dim=-1)
        return [p]