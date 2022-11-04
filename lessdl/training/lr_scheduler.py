"""
Refer allennlp, add step_batch for lr_scheduler.
https://github.com/allenai/allennlp/blob/main/allennlp/training/learning_rate_schedulers/learning_rate_scheduler.py
"""
import torch
from torch.optim import lr_scheduler
from typing import Dict, Any
from overrides import overrides


def scheduler_wrapper(cls):
    """Add step_batch get_lr function to torch.optim.lr_scheduler.CLS
    """
    
    class LRWrapper(cls):
        
        @property
        def lr(self):
            return self.get_last_lr()

        def step_batch(*args, **kargs):
            """Do nothing
            """
            pass
    
    LRWrapper.__name__ = 'LRWrapper_{}'.format(cls.__name__)

    return LRWrapper


StepLR = scheduler_wrapper(lr_scheduler.StepLR)


class Scheduler:
    """
    A `Scheduler` is a generalization of PyTorch learning rate schedulers.
    A scheduler can be used to update any field in an optimizer's parameter groups,
    not just the learning rate.
    During training using the AllenNLP `Trainer`, this is the API and calling
    sequence for `step` and `step_batch`::
       scheduler = ... # creates scheduler
       batch_num_total = 0
       for epoch in range(num_epochs):
           for batch in batchs_in_epoch:
               # compute loss, update parameters with current learning rates
               # call step_batch AFTER updating parameters
               batch_num_total += 1
               scheduler.step_batch(batch_num_total)
           # call step() at the END of each epoch
           scheduler.step(validation_metrics, epoch)
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, param_group_field: str, last_epoch: int = -1
    ) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if last_epoch == -1:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(
                        f"{self._initial_param_group_field} missing from param_groups[{i}]"
                    )
        self.base_values = [
            group[self._initial_param_group_field] for group in self.optimizer.param_groups
        ]
        self.last_epoch = last_epoch

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the scheduler as a `dict`.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the schedulers state.
        # Parameters
        state_dict : `Dict[str, Any]`
            Scheduler state. Should be an object returned from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_values(self):
        raise NotImplementedError

    def step(self, metric: float = None) -> None:
        """
        TODO: be compatible with torch.optim.lr_scheduler
        """
        self.last_epoch += 1
        self.metric = metric
        assert metric is None, 'To compatible, define step like: def step(epoch)'
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = value

    def step_batch(self, batch_num_total: int = None) -> None:
        """
        By default, a scheduler is assumed to only update every epoch, not every batch.
        So this does nothing unless it's overriden.
        """

        return


class LearningRateScheduler(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, "lr", last_epoch)

    @overrides
    def get_values(self):
        raise NotImplementedError


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class InverseSquareRootSchedule(LearningRateScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, optimizer, warmup_updates=4000, warmup_init_lr=-1, warmup_end_lr=0.0005, last_epoch=-1):
        """
        last_epoch 上一次的epoch数目
        """
        assert last_epoch == -1, '只考虑last_epoch=-1的情形'
        if warmup_init_lr < 0:
            warmup_init_lr = 0 if warmup_updates > 0 else warmup_end_lr
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates**0.5

        # initial learning rate
        self.lr = warmup_init_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_values(self):
        return [self.lr for _ in self.optimizer.param_groups]

    def set_lr(self):
        for param_group, value in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = value

    def step_batch(self, batch_num_total: int = None):
        """Update the learning rate after each update."""
        if batch_num_total < self.warmup_updates:
            self.lr = self.warmup_init_lr + batch_num_total*self.lr_step
        else:
            self.lr = self.decay_factor * batch_num_total**-0.5
        self.set_lr()
        return self.lr


class ExponentialDecayLR(LearningRateScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, decay_rate, decay_steps, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.total_steps = 0
        self.init_lr = [pg['lr'] for pg in optimizer.param_groups]
    
    @property
    def lr(self):
        return self.get_values()[0]

    @overrides
    def get_values(self):
        return [lr * (self.decay_rate ** (self.total_steps / self.decay_steps)) for lr, _ in zip(self.init_lr, self.optimizer.param_groups)]

    def step_batch(self, batch_num_total: int = 1) -> None:
        self.total_steps += 1
        self.step()
