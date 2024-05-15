# Torch Terinador

A trainer based on pytorch including a train loop for MDN (Mixture Density Network), a data loader, plot line chart and 
a couple of techniques for avoid over fitting

## Installation
This package needs Python>=3.7 and the version of Pytorch used in development is 1.13.1 and cuda11.2, considering the different version of cuda, the package will
not install Pytorch automatically. You should check your cuda's version, install the suitable [pytorch](https://pytorch.org/get-started/previous-versions/) first. Then, run the command below:
```
pip install tortreinador 
```
## Quick Start
```python
from tortreinador.utils.plot import plot_line_2
from tortreinador.utils.preprocessing import load_data
from tortreinador.train import TorchTrainer
from tortreinador.models.MDN import mdn, Mixture, NLLLoss
from tortreinador.utils.tools import xavier_init
from tortreinador.utils.View import init_weights, split_weights
import torch
import pandas as pd

data = pd.read_excel('D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx')
data['M_total (M_E)'] = data['Mcore (M_J/10^3)'] + data['Menv (M_E)']

# Support index, e.g input_parameters = [0, 1, 2]
input_parameters = [
    'Mass (M_J)',
    'Radius (R_E)',
    'T_sur (K)',
]

output_parameters = [
    'M_total (M_E)',
    'T_int (K)',
    'P_CEB (Mbar)',
    'T_CEB (K)'
]
# Load Data
t_loader, v_loader, t_x, t_y, m_x, m_y = load_data(data, input_parameters, output_parameters, batch_size=256)

model = mdn(len(input_parameters), len(output_parameters), 20, 512)
criterion = NLLLoss()
optim = torch.optim.Adam(xavier_init(model), lr=0.0001, weight_decay=0.001)

trainer = TorchTrainer(is_gpu=True, epoch=50, optimizer=optim, model=model, criterion=criterion)

'''
        kwargs: model_save_path -> m_p, warmup_epoch(option) -> w_e, lr_milestones and gamma(option) -> l_m, best_metric(eg: r2) -> b_m
'''
config = {
    'b_m': 0.8,
    'm_p': 'D:\\Resource\\MDN\\PackageTest\\savedModel\\',
    'w_e': 5,
    'l_m': {
        's_l': [20, 40],
        'gamma': 0.7
    }
}
# Training
result = trainer.fit(t_loader, v_loader, **config)


# Plot line chart
result_pd = pd.DataFrame()
result_pd['epoch'] = len(result[0])
result_pd['train_r2_avg'] = result[4]
result_pd['val_r2_avg'] = result[3]

plot_line_2(y_1='train_r2_avg', y_2='val_r2_avg', df=result_pd, fig_size=(10, 6), output_path=".\\imgs\\GasGiants_MDN20240116_TrainValR2_2.png", dpi=300)
```
## Functions
This package just support MDN for now, but the ```load_data``` is suitable for every condition as long as the type of data is Dataframe

- tortreinador.train.TorchTrainer():
   + Parameters:
     + batch_size: int = 512
     + is_gpu: bool = True
     + epoch: int = 150
     + log_dir: Optional[str] = None, ***Specify a file path to start up tensorboardX***

   + Functions:
     + load_data()
       + Describe: ***Processing Dataframe according to the input/output parameters and split size to train set, validation set and test set,
       you can freely choose if normalization, if shuffle.***
       + Parameters:
         + data: DataFrame,
         + input_parameters: list,
         + output_parameters: list,
         + feature_range: Any = None,
         + train_size: float = 0.8,
         + val_size: float = 0.1,
         + test_size: float = 0.1,
         + if_normal: bool = True,
         + if_shuffle: bool = True,
         + n_workers: int = 8
       + Return:
         + DataLoader
         + DataLoader
         + Numpy array
         + Numpy array
         + MinMaxScaler
         + MinMaxScaler
     + plot_line_2()
       + Describe: ***This function is usually used after training to compare the validation loss and train loss, validation R2 and train R2*** 
       + Parameters:
         + y_1: str
         + y_2: str
         + df: DataFrame
         + output_path: str
         + fig_size: tuple = (10, 6)
         + dpi: int = 300
     + xavier_init()
       + Describe: ***A technique for prevent over fitting***
       + Parameters:
         + net: Module
     + _calculate()
       + Describe: ***A private method for calculate loss, it is able to overwrite(Testing)***
       + Parameters:
         + model: Any,
         + pdf: Any,
         + x: Any,
         + y: Any,
         + criterion: Any,
         + t: str = 'train'
       ```python
       # Please call self._standard_return() at the end
       def calculate(self, x, y, mode='t'):

            pi, mu, sigma = self.model(x)

            loss = self.criterion(pi, mu, sigma, y)

            pdf = mixture(pi, mu, sigma)

            y_pred = pdf.sample()

            metric_per = r2_score(y, y_pred)
            
            return self._standard_return(loss, metric_per, mode, y, y_pred)
         ```
     + fit_for_MDN()
       + Describe: ***Train loop for MDN(Mixture Density Network)*** 
       + Parameters:
         + t_l: Dataloader
         + v_l: Dataloader
         + criterion: Module
         + optim: Optimizer
         + model: Module
         + model_save_path: str ***The model which has the best performance will save according ```model_save_path```***
         + mixture: Module ***The sampling class inherited from nn.Module***
         + warmup_epoch: Optional[int] = None  ***A technique for prevent over fitting, specify a number such as 5, then it will use warm up in the first 5 epoch***
         + lr_milestones: Optional[list] = None ***Decrease learning rate according the input list and gamma, for example: lr_milestones=[10], gamma=0.7, then the learning rate will x0.7 at the 10 epoch***
         + gamma: float = 0.7
         + best_r2: float = 0.80
       + Return:
         + train loss: List
         + validation loss: list
         + validation R2: list
         + train R2: list
         + validation mse: list












