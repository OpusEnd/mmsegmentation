import os
from torch.utils.tensorboard import SummaryWriter
from mmengine.hooks import Hook

class CustomTensorboardLoggerHook(Hook):
    def __init__(self, log_dir='./work_dirs/tf_logs', interval=50):
        self.log_dir = log_dir
        self.interval = interval
        self.writer = None

    def before_train(self, runner):
        """在训练开始前初始化 TensorBoard 日志目录。"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """每隔指定的 interval 记录日志。"""
        if self.every_n_train_iters(runner, self.interval):
            if 'loss' in outputs:
                self.writer.add_scalar('Train/Loss', outputs['loss'].item(), runner.iter)

    def after_train_epoch(self, runner):
        """在每个 epoch 结束后关闭 writer。"""
        self.writer.flush()

    def after_train(self, runner):
        """在训练结束后关闭 TensorBoard writer。"""
        if self.writer:
            self.writer.close()
