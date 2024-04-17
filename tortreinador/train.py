import torch
import torch.nn as nn
from torch.optim import Optimizer
from utils.metrics import r2_score, mixture
from tqdm import tqdm
from utils.Recorder import Recorder
from utils.WarmUpLR import WarmUpLR
from utils.View import init_weights, visualize_lastlayer, visualize_train_loss, visualize_test_loss, \
    split_weights
from tensorboardX import SummaryWriter


class TorchTrainer:
    """
        Implemented the train loop based on pytorch

        kwargs: is_gpu, epoch, model, optimizer, criterion, validation_metric(eg: validation MSE)

    """

    def __init__(self,
                 is_gpu: bool = True,
                 epoch: int = 150, log_dir: str = None, model: nn.Module = None,
                 optimizer: Optimizer = None, extra_metric: nn.Module = None, criterion: nn.Module = None):

        if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer) or not isinstance(criterion,
                                                                                                      nn.Module) or epoch is None:
            raise ValueError("Please provide the correct type of model, optimizer, criterion and the not none epoch")

        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else log_dir

        self.train_loss_recorder = Recorder()
        self.val_loss_recorder = Recorder()
        self.train_metric_recorder = Recorder()
        self.val_metric_recorder = Recorder()

        if extra_metric is not None:
            self.extra_metric = extra_metric
            self.extra_recorder = Recorder()

        print("Epoch:{}, is GPU: {}".format(epoch, is_gpu))

    def calculate(self, x, y, mode='t'):

        pi, mu, sigma = self.model(x)

        loss = self.criterion(pi, mu, sigma, y)

        pdf = mixture(pi, mu, sigma)

        y_pred = pdf.sample()

        metric_per = r2_score(y, y_pred)

        if mode == 't':
            return [loss, metric_per, 't']

        elif mode == 'v' and 'extra_metric' in self.__dict__:
            return [loss, metric_per, y, y_pred, 'v']

        elif mode == 'v':
            return [loss, metric_per, 'v']

    def cal_result(self, *args):
        if args[-1] == 't':
            self.train_loss_recorder.update(args[0].item())
            self.train_metric_recorder.update(args[1].item())
            return {
                'loss': (self.train_loss_recorder.val, '.4f'),
                'loss_avg': (self.train_loss_recorder.avg, '.4f'),
                'train_metric': (self.train_metric_recorder.avg, '.4f')
            }, args[0]

        elif args[-1] == 'v':
            self.val_loss_recorder.update(args[0].item())
            self.val_metric_recorder.update(args[1].item())
            if len(args[:-1]) == 4:
                self.extra_recorder.update(self.extra_metric(args[2], args[3]).item())
                return {
                    'loss': (self.val_loss_recorder.val, '.4f'),
                    'loss_avg': (self.val_loss_recorder.avg, '.4f'),
                    'val_metric': (self.val_metric_recorder.avg, '.4f'),
                    'extra_metric': (self.extra_recorder.avg, '.4f')
                }

            else:
                return {
                    'loss': (self.val_loss_recorder.val, '.4f'),
                    'loss_avg': (self.val_loss_recorder.avg, '.4f'),
                    'val_metric': (self.val_metric_recorder.avg, '.4f'),
                }

    '''
        Parameter Check

        :param b_m: A param like R-Square, this param is used for judging the model can be save or not
        :return: bool
    '''

    def _check_best_metric_for_regression(self, b_m):
        if b_m >= 1.0:
            return False

        else:
            return True

    def _check_param_exist(self, b_m):
        if b_m is not None:
            return True

        else:
            return False

    '''
        kwargs: model_save_path -> m_p, warmup_epoch(option) -> w_e, lr_milestones and gamma(option) -> l_m, best_metric(eg: r2) -> b_m
    '''

    def fit_for_MDN(self, t_l, v_l, **kwargs):
        if not self._check_param_exist(kwargs['b_m']):
            raise ValueError('Best metric does not exist')

        else:
            if not self._check_best_metric_for_regression(kwargs['b_m']):
                raise ValueError("Best metric can't higher than 1.0")

        IS_WARMUP = False
        IS_LR_MILESTONE = False

        self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

        # Schedular 1
        if 'w_e' in kwargs.keys():
            IS_WARMUP = True
            warmup = WarmUpLR(self.optimizer, len(t_l) * kwargs['w_e'])

        # Schedular 2
        if 'l_m' in kwargs.keys():
            IS_LR_MILESTONE = True
            lr_schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=kwargs['l_m']['s_l'],
                                                                gamma=kwargs['l_m']['gamma'])

        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())

        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_metric = []
        epoch_val_metric = []
        epoch_extra_metric = None

        if 'extra_metric' in self.__dict__:
            epoch_extra_metric = []

        # Epoch
        for e in range(self.epoch):

            self.model.train()

            if IS_WARMUP is True and IS_LR_MILESTONE is True and e >= kwargs['w_e']:
                lr_schedular.step()

            # lr_schedular.step()

            i = 0

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if IS_WARMUP is True and e < kwargs['w_e']:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    self.optimizer.zero_grad()

                    param_options, loss = self.cal_result(*self.calculate(mini_batch_x, mini_batch_y, mode='t'))

                    param_options['lr'] = (self.optimizer.state_dict()['param_groups'][0]['lr'], '.6f')

                    params = {key: "{value:{format}}".format(value=value, format=f)
                              for key, (value, f) in param_options.items()}

                    loss.backward()

                    self.optimizer.step()

                    if self.writer is not None:
                        n_iter = (e - 1) * len(t_l) + i + 1
                        visualize_lastlayer(self.writer, self.model, n_iter)
                        visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(**params)

                epoch_train_metric.append(self.train_metric_recorder.avg)
                epoch_train_loss.append(self.train_loss_recorder.avg)

            with torch.no_grad():
                self.model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:
                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        param_options = self.cal_result(*self.calculate(val_batch_x, val_batch_y, mode='v'))

                        params = {key: "{value:{format}}".format(value=value, format=f)
                                  for key, (value, f) in param_options.items()}

                        v_epoch.set_postfix(**params)

                epoch_val_loss.append(self.val_loss_recorder.avg)
                epoch_val_metric.append(self.val_metric_recorder.avg)
                if epoch_extra_metric is not None:
                    epoch_extra_metric.append(self.extra_recorder.avg)

                if self.writer is not None:
                    visualize_test_loss(self.writer, epoch_val_loss[-1], e)

                if self.val_metric_recorder.avg > kwargs['b_m']:
                    kwargs['b_m'] = self.val_metric_recorder.avg

                    if 'm_p' in kwargs.keys():
                        torch.save(self.model.state_dict(), '{}best_model.pth'.format(kwargs['m_p']))

                        print("Save Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(self.val_metric_recorder.avg,
                                                                                   self.val_loss_recorder.avg))

                    else:
                        print("Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(self.val_metric_recorder.avg,
                                                                              self.val_loss_recorder.avg))

            self.train_loss_recorder = Recorder()
            self.val_loss_recorder = Recorder()
            self.train_metric_recorder = Recorder()
            self.val_metric_recorder = Recorder()
            if 'extra_metric' in self.__dict__:
                self.extra_recorder = Recorder()

            if IS_WARMUP is False and IS_LR_MILESTONE is True:
                lr_schedular.step()

        if 'extra_metric' in self.__dict__:
            return epoch_train_loss, epoch_val_loss, epoch_val_metric, epoch_train_metric, epoch_extra_metric

        else:
            return epoch_train_loss, epoch_val_loss, epoch_val_metric, epoch_train_metric
