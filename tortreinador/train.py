import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
from tortreinador.utils.Recorder import Recorder
from tortreinador.utils.WarmUpLR import WarmUpLR

from tortreinador.utils.View import init_weights, visualize_lastlayer, visualize_train_loss, visualize_test_loss, split_weights
from tensorboardX import SummaryWriter


class TorchTrainer:

    """
        1. Split data, Data Normalization(choose), convert data to tensor
        2. plot Line
        3. Implemented the train loop based on pytorch
    """

    def __init__(self, batch_size: int = 512, is_gpu: bool = True,
                 epoch: int = 150, log_dir: str = None):
        self.b_s = batch_size
        self.epoch = epoch
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')
        self.writer = eval("{}".format(SummaryWriter(log_dir=log_dir) if log_dir is not None else log_dir))
        # MSE
        self.mse = nn.MSELoss()

        print("Batch size: {}, Epoch:{}, is GPU: {}".format(self.b_s, self.epoch, is_gpu))

    def load_data(self, data: pd.DataFrame, input_parameters: list, output_parameters: list,
                  feature_range=None, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1,
                  if_normal: bool = True, if_shuffle: bool = True, n_workers: int = 8):
        """
        Load Data and Normalization
        :return:
            Train DataLoader, Validation DataLoader, Test X, Test Y, Scaler X, Scaler Y
        """
        if (train_size + test_size + val_size) != 1:
            raise ValueError("train_size + test_size + val_size must equals 1")

        # Data Normalization
        if feature_range is None:
            feature_range = [0, 1]
        scaler_x = MinMaxScaler(feature_range=feature_range)
        scaler_y = MinMaxScaler(feature_range=feature_range)

        train_x = None
        train_y = None
        val_x = None
        val_y = None
        test_x = None
        test_y = None

        data_x = eval("data.{}[:, input_parameters]".format('iloc' if type(input_parameters[0]) == int else 'loc'))
        data_y = eval("data.{}[:, output_parameters]".format('iloc' if type(output_parameters[0]) == int else 'loc'))

        v_t_sum = (val_size + test_size)

        val_rate = val_size / v_t_sum

        if if_normal:
            data_x_nor = pd.DataFrame(scaler_x.fit_transform(data_x))
            data_y_nor = pd.DataFrame(scaler_y.fit_transform(data_y))

            if if_shuffle:
                train_x = data_x_nor.sample(frac=train_size)
                train_y = data_y_nor.iloc[train_x.index]

                val_x = data_x_nor.drop(train_x.index).sample(frac=val_rate)
                val_y = data_y_nor.iloc[val_x.index]

                test_x = data_x_nor.drop(train_x.index).drop(val_x.index)
                test_y = data_y_nor.iloc[test_x.index]

                train_x.reset_index(inplace=True, drop=True)
                train_y.reset_index(inplace=True, drop=True)
                val_x.reset_index(inplace=True, drop=True)
                val_y.reset_index(inplace=True, drop=True)
                test_x.reset_index(inplace=True, drop=True)
                test_y.reset_index(inplace=True, drop=True)

            elif not if_shuffle:
                train_x = data_x_nor.iloc[:int(len(data) * train_size), :]
                train_y = data_y_nor.iloc[:int(len(data) * train_size), :]

                val_x = data_x_nor.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]
                val_y = data_y_nor.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]

                test_x = data_x_nor.iloc[int(len(data) * (train_size + val_size)):, :]
                test_y = data_y_nor.iloc[int(len(data) * (train_size + val_size)):, :]

        else:
            if if_shuffle:
                train_x = data_x.sample(frac=train_size)
                train_y = data_y.iloc[train_x.index]

                val_x = data_x.drop(train_x.index).sample(frac=val_rate)
                val_y = data_y.iloc[val_x.index]

                test_x = data_x.drop(train_x.index).drop(val_x.index)
                test_y = data_y.iloc[test_x.index]

                train_x.reset_index(inplace=True, drop=True)
                train_y.reset_index(inplace=True, drop=True)
                val_x.reset_index(inplace=True, drop=True)
                val_y.reset_index(inplace=True, drop=True)
                test_x.reset_index(inplace=True, drop=True)
                test_y.reset_index(inplace=True, drop=True)

            elif not if_shuffle:
                train_x = data_x.iloc[:int(len(data) * train_size), :]
                train_y = data_y.iloc[:int(len(data) * train_size), :]

                val_x = data_x.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]
                val_y = data_y.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]

                test_x = data_x.iloc[int(len(data) * (train_size + val_size)):, :]
                test_y = data_y.iloc[int(len(data) * (train_size + val_size)):, :]

        train_x = torch.from_numpy(train_x.to_numpy()).double()
        train_y = torch.from_numpy(train_y.to_numpy()).double()
        val_x = torch.from_numpy(val_x.to_numpy()).double()
        val_y = torch.from_numpy(val_y.to_numpy()).double()
        test_x = torch.from_numpy(test_x.to_numpy()).double()
        test_y = torch.from_numpy(test_y.to_numpy()).double()

        t_set = TensorDataset(train_x, train_y)
        train_loader = DataLoader(t_set, batch_size=self.b_s, shuffle=False, num_workers=n_workers)

        v_set = TensorDataset(val_x, val_y)
        validation_loader = DataLoader(v_set, batch_size=self.b_s, shuffle=False, num_workers=n_workers)

        return train_loader, validation_loader, test_x, test_y, scaler_x, scaler_y

    def plot_line_2(self, y_1: str, y_2: str, df: pd.DataFrame, output_path: str, fig_size: tuple = (10, 6), dpi: int = 300):
        """
        Plot Merge Line (2 Lines) using Seaborn
        :param y_1: Name of Line 1
        :param y_2: Name of Line 2
        :param df: Dataframe
        :param fig_size:
        :param output_path:
        :param dpi:
        :return: Show Line picture and save to the specific location
        """
        fig = plt.figure(figsize=fig_size)
        sns.lineplot(x='epoch', y=y_1, data=df)
        sns.lineplot(x='epoch', y=y_2, data=df)
        plt.show()
        fig.savefig(output_path, dpi=dpi)

    def _calculate(self, model, pdf, x, y, criterion, t='train'):
        pi, mu, sigma = model(x)

        mixture = pdf(pi, mu, sigma)

        y_pred = mixture.sample()

        if t == 'train':
            return criterion(pi, mu, sigma, y), y_pred.cpu().numpy(), y.cpu().numpy()

        else:
            return criterion(pi, mu, sigma, y), self.mse(y_pred, y), y_pred.cpu().numpy(), y.cpu().numpy()

    # Xavier init
    def xavier_init(self, net: nn.Module):
        return split_weights(init_weights(net))

    def fit_for_MDN(self, t_l, v_l, criterion: nn.Module, optim, model: nn.Module, model_save_path: str,
                    mixture: nn.Module, warmup_epoch: int = None,
                    lr_milestones: list = None, gamma: float = 0.7, best_r2: float = 0.80):

        model = nn.DataParallel(model)

        model.to(self.device)

        # Loss Function
        criterion = criterion

        # Sample Function
        pdf = mixture

        # Optimizer
        optimizer = optim

        # Schedular 1
        if warmup_epoch is not None:
            warmup = WarmUpLR(optimizer, len(t_l) * warmup_epoch)

        # Schedular 2
        if lr_milestones is not None:
            lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)

        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())

        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_r2 = []
        epoch_val_r2 = []
        epoch_mse = []
        best_r2 = best_r2

        # Epoch
        for e in range(self.epoch):
            model.train()

            train_loss = Recorder()
            val_loss = Recorder()
            train_r2_recorder = Recorder()
            val_r2_recorder = Recorder()
            mse_recorder = Recorder()

            if warmup_epoch is not None and lr_milestones is not None and e > warmup_epoch:
                lr_schedular.step()

            # lr_schedular.step()

            i = 0

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if warmup_epoch is not None and e <= warmup_epoch :
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    optimizer.zero_grad()

                    loss, y_pred, y_cpu = self._calculate(model, pdf, mini_batch_x, mini_batch_y, criterion, t='train')

                    r2_per = r2_score(y_cpu, y_pred)

                    train_loss.update(loss.item())
                    train_r2_recorder.update(r2_per.astype('float32'))

                    loss.backward()

                    optimizer.step()

                    n_iter = (e - 1) * len(t_l) + i + 1

                    if self.writer is not None:
                        visualize_lastlayer(self.writer, model, n_iter)
                        visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(loss="{:.4f}".format(train_loss.val),
                                        lr="{:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr']),
                                        loss_avg="{:.4f}".format(train_loss.avg),
                                        r2="{:.4f}".format(train_r2_recorder.avg))

                epoch_train_r2.append(train_r2_recorder.avg)
                epoch_train_loss.append(train_loss.avg)

            with torch.no_grad():
                model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:
                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        loss, mse_per, y_pred, val_batch_y = self._calculate(model, pdf, val_batch_x, val_batch_y, criterion, t='validate')

                        r2_per = r2_score(val_batch_y, y_pred)

                        val_loss.update(loss.item())
                        mse_recorder.update(mse_per.item())
                        val_r2_recorder.update(r2_per.astype('float32'))

                        v_epoch.set_postfix(loss_val="{:.4f}".format(val_loss.val),
                                            loss_avg="{:.4f}".format(val_loss.avg),
                                            r2="{:.4f}".format(val_r2_recorder.avg),
                                            mse="{:.4f}".format(mse_recorder.avg))

                epoch_val_loss.append(val_loss.avg)
                epoch_val_r2.append(val_r2_recorder.avg)
                epoch_mse.append(mse_recorder.avg)

                if self.writer is not None:
                    visualize_test_loss(self.writer, epoch_val_loss[-1], e)


                if val_r2_recorder.avg > best_r2:
                    torch.save(model.state_dict(), '{}best_model.pth'.format(model_save_path))
                    best_r2 = val_r2_recorder.avg
                    print("Save Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(val_r2_recorder.avg, val_loss.avg))



        return epoch_train_loss, epoch_val_loss, epoch_val_r2, epoch_train_r2, epoch_mse




