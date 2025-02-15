import torch
import torch.optim as optim
import math


# 自定义调度器：预热 + 余弦退火
class CosineAnnealingWarmupLR(optim.lr_scheduler._LRScheduler):
    """
    Overview:
        Cosine annealing scheduler with warmup.
    Interfaces:
        ``__init__``, ``get_lr``
    """

    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
        """
        Overview:
            Initialize the scheduler.
        Arguments:
            - optimizer (:obj:`torch.optim.Optimizer`): The optimizer.
            - T_max (:obj:`int`): The maximum number of iterations.
            - eta_min (:obj:`float`): The minimum learning rate.
            - warmup_steps (:obj:`int`): The number of warmup steps.
            - last_epoch (:obj:`int`): The index of the last epoch.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Overview:
            Get the learning rate.
        Returns:
            - lr (list): The learning rate.
        """
        if self.last_epoch < self.warmup_steps:
            # 预热阶段：线性增加学习率
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            cos_anneal_factor = (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps)
                    / (self.T_max - self.warmup_steps)
                )
            ) / 2
            return [
                self.eta_min + (base_lr - self.eta_min) * cos_anneal_factor
                for base_lr in self.base_lrs
            ]
