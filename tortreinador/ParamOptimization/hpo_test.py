import inspect
import pandas as pd
from tortreinador.ParamOptimization.seach_task import Task
from tortreinador.ParamOptimization.seach_taskmanager import TaskManager
from tortreinador.ParamOptimization.hyperparam import IntParam, FloatParam, LogFloatParam, ChoiceParam, HyperParam
from tortreinador.utils.preprocessing import ScalerConfig, load_data, get_dataloader
from tortreinador.utils.Recorder import MetricManager, MetricDefine
import torch
from tortreinador.train import TorchTrainer, config_generator
import numpy as np
from tortreinador.utils.metrics import r2_score
from test_dir.cVAEPaperspace import cVAE, CombineLoss


sig = inspect.signature(cVAE.__init__)
sig_2 = inspect.signature(Task.__init__)
sig_3 = inspect.signature(TaskManager.__init__)
print(sig)
print(sig_2)
print(sig_3)

df_chunk_0 = pd.read_parquet("D:\\Resource\\rockyExoplanetV3\\data_chunk_0.parquet")
df_chunk_1 = pd.read_parquet("D:\\Resource\\rockyExoplanetV3\\data_chunk_1.parquet")
df_all = pd.concat([df_chunk_0, df_chunk_1])
input_parameters = [
    'Mass',
    'Radius',
    'FeMg',
    'SiMg',
]
output_parameters = [
    'WRF',
    'MRF',
    'CRF',
    'WMF',
    'CMF',
    'CPS',
    'CTP',
    'k2'
]

model_hps = {
    'i_dim': 8,
    'c_dim': 4,
    'z_dim': IntParam(32, 64),
    'o_dim': 8,
    'num_hidden': IntParam(512, 2048),
    'mode': 'condition'
}
optim_hps = {
    'lr': LogFloatParam(2e-4, 2e-2),
    'weight_decay': LogFloatParam(1e-6, 1e-4)
}
dataset_hps = {
    'input_parameters': input_parameters,
    'output_parameters': output_parameters,
    'normal': ScalerConfig(on=True, method='standard', normal_y=True),
    'if_shuffle': True,
    'batch_size': 1024,
    'add_noise': True,
    'error_rate': [0.14, 0.04, 0.12, 0.13],
    'only_noise': True
}
trainer_hps = {
    'epoch': 10,
    'metric_manager': MetricManager([MetricDefine(metric_name='Loss_avg', metric_mode=0, use_as_criterion=True),
                                     MetricDefine(metric_name='Recon_loss', metric_mode=0),
                                     MetricDefine(metric_name='KLD', metric_mode=0),
                                     MetricDefine(metric_name='R2', metric_mode=0, use_as_baseline=True)])
}

trainer_configs = {
    'warmup_epochs': 5,
    'best_metric': 0.8,
    'auto_save': 2,
    'validation_cycle': 10
}
class Trainer(TorchTrainer):

    def calculate(self, x, y, mode=1):
        """Single prediction"""
        B = x.shape[0]
        fake_y = torch.from_numpy(np.random.normal(0, 1, (B, 8))).to(torch.float).to(self.device)

        recon_x, l_z_mu, l_z_logvar = self.model(fake_y, x)

        loss, re_loss, kld = self.criterion(recon_x, y, l_z_mu, l_z_logvar, auto_adj=(
        0.02, 0.03))  # z, mu, logvar min(0.5 + self.current_epoch * 0.1, 1.5)
        metric_per = r2_score(y, recon_x)
        update_values = [loss, re_loss, kld, metric_per]

        return self._standard_return(mode=mode, update_values=update_values)

tasks = [Task(model_class=cVAE, model_hps=model_hps, criterion=CombineLoss, optimizer_class=torch.optim.Adam,
              optimizer_hps=optim_hps, dataset=df_all, dataset_hps=dataset_hps, task_name='grid_search test', trainer=Trainer, trainer_hps=trainer_hps, train_configs=trainer_configs, search_times=2)]

tasks_manager = TaskManager(tasks)
tasks_manager.search()
tasks_manager.check_all_hps(tasks_manager.tasks[0])