import torch
import torch.distributed as dist

# Parent class for arch schedulers
class ArchScheduler(object):
    def __init__(self, total_epochs, curr_lam, lam_max, lam_min, last_epoch=-1):
        self.curr_lam = curr_lam
        self.lam_max = lam_max
        self.lam_min = lam_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
    
    def state_dict(self):
        return {
            'curr_lam': self.curr_lam,
            'lam_max': self.lam_max,
            'lam_min': self.lam_min,
            'last_epoch': self.last_epoch,
            'total_epochs': self.total_epochs,
        }
    
    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(
            total_epochs=state_dict['total_epochs'],
            curr_lam=state_dict['curr_lam'],
            lam_max=state_dict['lam_max'],
            lam_min=state_dict['lam_min'],
            last_epoch=state_dict['last_epoch']
        )
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + ' ( \n'
        format_string += str(self.state_dict())
        format_string += "\n)"
        return format_string

# Snas temperature scheduler
class TempScheduler(ArchScheduler):
    def __init__(self, total_epochs, curr_lam, lam_max, lam_min, last_epoch=-1):
       super(TempScheduler, self).__init__(total_epochs, curr_lam, lam_max, lam_min, last_epoch)
       self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process(epoch)

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.curr_lam = (1 - self.last_epoch / self.total_epochs) * (self.lam_max - self.lam_min) + self.lam_min
        if self.curr_lam < self.lam_min:
            self.curr_lam = self.lam_min
        return self.curr_lam

# Scheduler for the architecture regularization
class RegScheduler(ArchScheduler):
    def __init__(self, total_epochs, curr_lam, lam_max, lam_min, last_epoch=-1):
        super(RegScheduler, self).__init__(total_epochs, curr_lam, lam_max, lam_min, last_epoch)
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.increase(epoch)

    def increase(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.curr_lam = (self.last_epoch / self.total_epochs) * (self.lam_max - self.lam_min) + self.lam_min
        if self.curr_lam < self.lam_min:
            self.curr_lam = self.lam_min
        return self.curr_lam

def communicate_grads(network, n_proc):
    for name, param in network.named_parameters():
        if (not name == "_arch_alphas") and (not name == 'arch_parameters') and (not name == "alphas_normal") and (not name == "alphas_reduce"):
            if param.grad is None: 
                param.grad = torch.zeros_like(param.data)
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= n_proc