import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from tqdm import tqdm
from TorchLoop.Recorder import Recorder
from TorchLoop.WarmUpLR import WarmUpLR

from TorchLoop.View import init_weights, visualize_lastlayer, visualize_train_loss, visualize_test_loss
from tensorboardX import SummaryWriter


# Develop in MAC: /Users/xuhongtao/pycharmprojects/resource/Gas_Giants_Core_Earth20W.xlsx
# Develop in Windows: D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx


class TorchLoop:

    """
        1. Split data, Data Normalization(choose), convert data to tensor
        2. plot Line
        3. Implemented the train loop based on pytorch
    """

    def __init__(self, batch_size: int = 512, learning_rate: float = 0.001984, is_gpu: bool = True,
                 epoch: int = 150, weight_decay: float = None, log_dir: str = "D:\\Resource\\MDN\\Log\\"):
        self.lr = learning_rate
        self.b_s = batch_size
        self.epoch = epoch
        self.w_d = weight_decay
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=log_dir)

        print("Batch size: {}, Learning rate:{}, is GPU: {}".format(self.b_s, self.lr, is_gpu))

    def _calculate(self, model, pdf, x, y, criterion):
        pi, mu, sigma = model(x)

        mixture = pdf(pi, mu, sigma)

        y_pred = mixture.sample()

        return criterion(pi, mu, sigma, y), y_pred.cpu().numpy(), y.cpu().numpy()

    def fit_for_MDN(self, t_l, v_l, criterion: nn.Module = None, optim: str = 'Adam',
                    xavier_init: bool = True, model: nn.Module = None, mixture: nn.Module = None, warmup_epoch: int = None,
                    lr_milestones: list = False, gamma: float = 0.7, best_r2: float = 0.80):

        # Xavier Init
        if xavier_init:
            init_weights(model)

        model = nn.DataParallel(model)

        model.to(self.device)

        # Loss Function
        criterion = criterion

        # MSE
        mse = nn.MSELoss()

        # Sample Function
        pdf = mixture

        # Optimizer
        optimizer = eval("torch.optim.{}({}, lr={}, weight_decay={})".format(optim, 'split_weights(model)' if xavier_init else 'model.parameters()', self.lr, self.w_d if self.w_d is not None else None))

        # Schedular 1
        if warmup_epoch is not None:
            warmup = WarmUpLR(optimizer, len(t_l) * warmup_epoch)

        # Schedular 2
        if lr_milestones is not None:
            lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=lr_milestones, gamma=gamma)

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

            if e > warmup_epoch and lr_milestones is not None:
                lr_schedular.step()

            # lr_schedular.step()

            i = 0

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if e <= warmup_epoch:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    optimizer.zero_grad()

                    loss, y_pred, y_cpu = self._calculate(model, pdf, mini_batch_x, mini_batch_y, criterion)

                    r2_per = r2_score(y_cpu, y_pred)

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

                        loss, y_pred, val_batch_y = self._calculate(model, pdf, val_batch_x, val_batch_y, criterion)

                        mse_per = mse(y_pred, val_batch_y)

                        r2_per = r2_score(val_batch_y, y_pred)

                        val_loss.update(loss.item())
                        mse_recorder.update(mse_per.item())
                        val_r2_recorder.update(r2_per.astype('float32'))

                        v_epoch.set_postfix(loss_val="{:.4f}".format(val_loss.val), loss_avg="{:.4f}".format(val_loss.avg), r2="{:.4f}".format(val_r2_recorder.avg), mse="{:.4f}".format(mse_recorder.avg))

                epoch_val_loss.append(val_loss.avg)
                epoch_val_r2.append(val_r2_recorder.avg)
                epoch_mse.append(mse_recorder.avg)

                visualize_test_loss(self.writer, epoch_val_loss[-1], e)

                if e >= warmup_epoch:
                    if val_r2_recorder.avg > best_r2:
                        torch.save(model.state_dict(), 'D:\\Resource\\MDN\\model_best_mdn_normalization.pth')
                        best_r2 = val_r2_recorder.avg
                        print("Save Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(val_r2_recorder.avg, val_loss.avg))

                    else:
                        print(" ")
                else:
                    print(" ")

        return epoch_train_loss, epoch_val_loss, epoch_val_r2, epoch_train_r2, epoch_mse

