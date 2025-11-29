from tortreinador.Events.event_system import Event, EventType
from tortreinador.utils.WarmUpLR import WarmUpLR
import torch

class LRSchedule(Event):
    def __init__(self, trainer, **kwargs):
        super(LRSchedule, self).__init__()
        self.lr_configs = kwargs
        self.WARMUP_ON = False
        self.WARMUP_EPOCHS = None

        self.LR_MILESTONE_ON = False

        self.LR_RESTART_ON = False
        self.LR_SCHEDULE_MODE = None

        self.DATASET_LENGTH = self.lr_configs['dataset_length']

        if self.lr_configs['warmup']['on']:
            self.WARMUP_ON = True
            self.WARMUP_EPOCHS = self.lr_configs['warmup']['warmup_epochs']
            self.warmup = WarmUpLR(trainer.optimizer, self.DATASET_LENGTH * self.WARMUP_EPOCHS)

        # Schedular 2
        if self.lr_configs['lr_milestones']['on']:
            self.LR_MILESTONE_ON = True
            self.lr_schedular = torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer,
                                                                milestones=self.lr_configs['lr_milestones']['stone_list'],
                                                                gamma=self.lr_configs['lr_milestones']['gamma'])

        if self.lr_configs['lr_restart']['on']:
            self.LR_RESTART_ON = True
            self.restart_schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(trainer.optimizer,
                                                                                     T_0=self.lr_configs['lr_restart']['t_0'],
                                                                                     T_mult=self.lr_configs['lr_restart'][
                                                                                         't_mult'],
                                                                                     eta_min=self.lr_configs['lr_restart'][
                                                                                         'eta_min'])

    def on_fire(self, event_type: EventType, trainer, **kwargs):

        if event_type == EventType.TRAIN_EPOCH_START:
            if self.WARMUP_ON and self.LR_MILESTONE_ON and trainer.current_epoch >= self.WARMUP_EPOCHS:
                self.lr_schedular.step()

            if self.WARMUP_ON and self.LR_RESTART_ON is True and trainer.current_epoch >= self.WARMUP_EPOCHS and self.LR_SCHEDULE_MODE == 'epoch':
                self.restart_schedular.step()

        if event_type == EventType.TRAIN_BATCH_END:
            if self.WARMUP_ON and trainer.current_epoch < self.WARMUP_EPOCHS:
                self.warmup.step()

            if not self.WARMUP_ON and self.LR_RESTART_ON and self.LR_SCHEDULE_MODE == 'batch':
                self.restart_schedular.step(trainer.current_epoch + kwargs['batch_idx'] / self.DATASET_LENGTH)

            if self.WARMUP_ON and self.LR_RESTART_ON is True and trainer.current_epoch >= self.WARMUP_EPOCHS and self.LR_SCHEDULE_MODE == 'batch':
                self.restart_schedular.step(trainer.current_epoch - self.WARMUP_EPOCHS + 1 + kwargs['batch_idx'] / self.DATASET_LENGTH)

        if event_type == EventType.TRAIN_EPOCH_END:
            if self.WARMUP_ON is False and self.LR_MILESTONE_ON is True:
                self.lr_schedular.step()

            if self.WARMUP_ON is False and self.LR_RESTART_ON and self.LR_SCHEDULE_MODE == 'epoch':
                self.restart_schedular.step()