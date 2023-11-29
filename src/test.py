import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from Rock.Model.MDN_From_Kaggle import mdn as mdn_advance, RelativeError, Mixture, NLLLoss
from tqdm import tqdm
from Rock.Utils.Recorder import Recorder
from Rock.Utils.WarmUpLR import WarmUpLR

from Rock.Utils.View import init_weights, split_weights, visualize_lastlayer, visualize_train_loss, visualize_test_loss
from tensorboardX import SummaryWriter


# Develop in MAC: /Users/xuhongtao/pycharmprojects/resource/Gas_Giants_Core_Earth20W.xlsx
# Develop in Windows: D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx


class TorchLoop:

    """
        1. Split data, Data Normalization(choose), convert data to tensor
        2. plot Line
        3. Implemented the train loop based on pytorch
    """

    def __init__(self, batch_size: int = 512, learning_rate: float = 0.001984,
                 hidden_size: int = 256, n_gaussian: int = 5, is_gpu: bool = True,
                 epoch: int = 150, weight_decay: float = 0.1, log_dir: str = "D:\\Resource\\MDN\\Log\\"):
        self.lr = learning_rate
        self.b_s = batch_size
        self.h_s = hidden_size
        self.n_g = n_gaussian
        self.epoch = epoch
        self.w_d = weight_decay
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')
        # self.writer = SummaryWriter(log_dir=log_dir)

        print("Batch size: {}, Learning rate:{}, Hidden size:{}, is GPU: {}".format(self.b_s, self.lr, self.h_s, is_gpu))

    def calculate(self, ):
        pass

    def fit_for_regression(self, t_l, v_l, i_s, o_s, criterion: nn.Module = NLLLoss(), optim: 'str' = 'Adam'):
        """
        :param optim:
        :param criterion:
        :param t_l: Train Loader
        :param v_l: Validation Loader
        :param i_s: Input Size
        :param o_s: Output Size
        :return: Recorded lists
        """

        # Parameters
        input_size = i_s
        output_size = o_s

        # load model
        model = mdn_advance(input_size, output_size, self.n_g, self.h_s)

        # Xavier Init
        init_weights(model)
        model = nn.DataParallel(model)
        model.to(self.device)

        # Loss Function
        criterion = criterion

        # MSE
        mse = nn.MSELoss()

        # Sample Function
        pdf = Mixture()

        # Relative Error
        r_e = RelativeError()

        # Optimizer
        # optimizer = torch.optim.Adam(split_weights(model), lr=self.lr, weight_decay=self.w_d)
        optimizer = eval("torch.optim.{}(split_weights(model), lr=self.lr, weight_decay=self.w_d)".format(optim))

        # Schedular 1
        warmup = WarmUpLR(optimizer, len(t_l) * 5)

        # Schedular 2
        # lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[115], gamma=0.7)

        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())

        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_r2 = []
        epoch_val_r2 = []
        epoch_r_e = []
        epoch_mse = []
        best_r2 = 0.80

        # Epoch
        for e in range(self.epoch):
            model.train()

            train_loss = Recorder()
            val_loss = Recorder()
            train_r2_recorder = Recorder()
            val_r2_recorder = Recorder()
            r_e_recorder = Recorder()
            mse_recorder = Recorder()

            # if e > 5:
            #     lr_schedular.step()

            # lr_schedular.step()

            i = 0

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if e <= 5:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    optimizer.zero_grad()

                    pi, mu, sigma = model(mini_batch_x)

                    mixture = pdf(pi, mu, sigma)

                    y_pred = mixture.sample()

                    y_pred = y_pred.cpu().numpy()
                    val_batch_y = mini_batch_y.cpu().numpy()

                    r2_per = r2_score(val_batch_y, y_pred)

                    loss = criterion(pi, mu, sigma, mini_batch_y)     # NLLLoss Function

                    train_loss.update(loss.item())
                    train_r2_recorder.update(r2_per.astype('float32'))

                    loss.backward()

                    optimizer.step()

                    n_iter = (e - 1) * len(t_l) + i + 1
                    visualize_lastlayer(self.writer, model, n_iter)
                    visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(loss="{:.4f}".format(train_loss.val), lr="{:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr']), loss_avg="{:.4f}".format(train_loss.avg), r2="{:.4f}".format(train_r2_recorder.avg))

                epoch_train_r2.append(train_r2_recorder.avg)
                epoch_train_loss.append(train_loss.avg)

            with torch.no_grad():
                model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:

                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        pi_val, mu_val, sigma_val = model(val_batch_x)

                        mixture = pdf(pi_val, mu_val, sigma_val)

                        # loss = criterion(val_batch_y, pi_val, mu_val, sigma_val)         # NLLLoss Function Probability Density Function
                        loss = criterion(pi_val, mu_val, sigma_val, val_batch_y)         # NLLLoss Function Sampling

                        y_pred = mixture.sample()

                        # loss = criterion(y_pred, val_batch_y)       # MSE
                        mse_per = mse(y_pred, val_batch_y)

                        y_pred = y_pred.cpu().numpy()
                        val_batch_y = val_batch_y.cpu().numpy()

                        r2_per = r2_score(val_batch_y, y_pred)
                        r_e_per = r_e(val_batch_y, y_pred)

                        val_loss.update(loss.item())
                        mse_recorder.update(mse_per.item())
                        val_r2_recorder.update(r2_per.astype('float32'))
                        r_e_recorder.update(r_e_per.astype('float32'))

                        v_epoch.set_postfix(loss_val="{:.4f}".format(val_loss.val), loss_avg="{:.4f}".format(val_loss.avg), r2="{:.4f}".format(val_r2_recorder.avg), relative_error="{:.4f}".format(r_e_recorder.avg), mse="{:.4f}".format(mse_recorder.avg))

                epoch_val_loss.append(val_loss.avg)
                epoch_val_r2.append(val_r2_recorder.avg)
                epoch_r_e.append(r_e_recorder.avg)
                epoch_mse.append(mse_recorder.avg)

                visualize_test_loss(self.writer, epoch_val_loss[-1], e)

                if e >= 15:
                    if val_r2_recorder.avg > best_r2:
                        torch.save(model.state_dict(), 'D:\\Resource\\MDN\\model_best_mdn_normalization.pth')
                        best_r2 = val_r2_recorder.avg
                        print("Save Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(val_r2_recorder.avg, val_loss.avg))

                    else:
                        print(" ")
                else:
                    print(" ")

        return epoch_train_loss, epoch_val_loss, epoch_val_r2, epoch_train_r2, epoch_r_e, epoch_mse

