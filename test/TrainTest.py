import torch
import torch.nn as nn
from tortreinador.utils.metrics import r2_score
from tqdm import tqdm
from tortreinador.utils.Recorder import Recorder
from tortreinador.utils.WarmUpLR import WarmUpLR
from tortreinador.utils.View import init_weights, visualize_lastlayer, visualize_train_loss, visualize_test_loss, \
    split_weights
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
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else log_dir
        # MSE
        self.mse = nn.MSELoss()

        print("Batch size: {}, Epoch:{}, is GPU: {}".format(self.b_s, self.epoch, is_gpu))

    def _calculate(self, model, pdf, x, y, criterion, loss_recorder, metric_recorder, t='train'):
        pi, mu, sigma = model(x)

        mixture = pdf(pi, mu, sigma)

        y_pred = mixture.sample()

        loss = criterion(pi, mu, sigma, y)

        metric_per = r2_score(y, y_pred)

        loss_recorder.update(loss.item())
        metric_recorder.update(metric_per.item())

        """
        :update in 2024/3/11 change the dict from 1 level to 2 level
        """

        if t == 'train':
            return {'show': {
                'loss': (loss_recorder.val, '.4f'),
                'loss_avg': (loss_recorder.avg, '.4f'),
                'r2': (metric_recorder.avg, '.4f'),
            }, 'curve': {'loss': loss}}


        else:
            mse = self.mse(y, y_pred).item()
            return {'show': {
                'loss': (loss_recorder.val, '.4f'),
                'loss_avg': (loss_recorder.avg, '.4f'),
                'r2': (metric_recorder.avg, '.4f'),
                'mse': (mse, '.4f')
            }, 'curve': mse}

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

            train_loss_recorder = Recorder()
            val_loss_recorder = Recorder()
            train_metric_recorder = Recorder()
            val_metric_recorder = Recorder()
            mse_recorder = Recorder()

            if warmup_epoch is not None and lr_milestones is not None and e > warmup_epoch:
                lr_schedular.step()

            # lr_schedular.step()

            i = 0
            # todo: 測試Update的功能
            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if warmup_epoch is not None and e <= warmup_epoch:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    optimizer.zero_grad()

                    param_options = self._calculate(model, pdf, mini_batch_x, mini_batch_y, criterion,
                                                          train_loss_recorder, train_metric_recorder, t='train')

                    param_options['show']['lr'] = (optimizer.state_dict()['param_groups'][0]['lr'], '.6f')

                    """
                    :update start
                    """
                    # todo: 加入對param_options['curve']['loss']的類型檢測
                    params = {key: "{value:{format}}".format(value=value, format=f)
                              for key, (value, f) in param_options['show'].items()}

                    param_options['curve']['loss'].backward()

                    optimizer.step()

                    n_iter = (e - 1) * len(t_l) + i + 1

                    if self.writer is not None:
                        visualize_lastlayer(self.writer, model, n_iter)
                        visualize_train_loss(self.writer, param_options['curve']['loss'].item(), n_iter)

                    t_epoch.set_postfix(**params)

                    """
                    :update end
                    """

                epoch_train_r2.append(train_metric_recorder.avg)
                epoch_train_loss.append(train_loss_recorder.avg)

            with torch.no_grad():
                model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:
                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        param_options, mse = self._calculate(model, pdf, val_batch_x, val_batch_y,
                                                             criterion, val_loss_recorder, val_metric_recorder,
                                                             t='validate')

                        params = {key: "{value:{format}}".format(value=value, format=f)
                                  for key, (value, f) in param_options.items()}

                        v_epoch.set_postfix(**params)
                
                """
                :update start
                """
                        for i in param_options['curve'].keys():
                            locals()[i] = param_options['curve'][i]

                        mse_recorder.update(mse)

                epoch_val_loss.append(val_loss_recorder.avg)
                epoch_val_r2.append(val_metric_recorder.avg)

                '''
                :update
                '''
                epoch_mse.append(mse_recorder.avg)

                if self.writer is not None:
                    visualize_test_loss(self.writer, epoch_val_loss[-1], e)

                if val_metric_recorder.avg > best_r2:
                    torch.save(model.state_dict(), '{}best_model.pth'.format(model_save_path))
                    best_r2 = val_metric_recorder.avg
                    print("Save Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(val_metric_recorder.avg,
                                                                               val_loss_recorder.avg))

        return epoch_train_loss, epoch_val_loss, epoch_val_r2, epoch_train_r2, epoch_mse
