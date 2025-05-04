from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

def make_scheduler(optimizer, warmup_epochs, args, train_loader):
    """
    Creates a learning rate scheduler for the optimizer.
    Args:
        optimizer: The optimizer you're using.
        warmup_epochs: Number of epochs for the warmup phase.
        args: Arguments, must include total epochs `args.epochs`.
        train_loader: The DataLoader for the training set (to determine steps per epoch).
    """
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / warmup_steps if step < warmup_steps else 1.0)
    total_steps = steps_per_epoch * args.epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return lr_scheduler