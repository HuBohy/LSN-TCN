import math

def change_lr_on_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CosineScheduler:
    def __init__(self, lr_ori, epochs):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer, epoch):
        reduction_ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori*reduction_ratio)

    def __format__(self, __format_spec: str) -> str:
        return f'CosineScheduler, lr: {self.lr_ori}, epochs: {self.epochs}'