from collections import OrderedDict
import torch
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from nlfd import build_model, weights_init
from loss import Loss
from tools.visual import Viz_visdom


class Solver(object):
    def __init__(self, train_loader, val_loader, test_loader, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
        self.beta = 0.3
        if config.visdom:
            self.visual = Viz_visdom("NLFD", 1)
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            self.net.load_state_dict(torch.load(self.config.model))
            self.net.eval()
            self.test_output = open("%s/test.txt" % config.test_fold, 'w')

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        self.net = build_model()
        if self.config.mode == 'train': self.loss = Loss(self.config.contour_th)
        if self.config.cuda: self.net = self.net.cuda()
        if self.config.cuda and self.config.mode == 'train': self.loss = self.loss.cuda()
        self.net.train()
        self.net.apply(weights_init)
        if self.config.load == '': self.net.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '': self.net.load_state_dict(torch.load(self.config.load))
        self.optimizer = Adam(self.net.parameters(), self.config.lr)
        self.print_network(self.net, 'NLFD')

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def clip(self, y):
        return torch.clamp(y, 0.0, 1.0)

    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    def f_measure(self, y_pred, y, threshold):
        y_pred = (y_pred >= threshold).float()
        y_flip, y_pred_flip = (y < 0.5).float(), (y < 0.5).float()
        tp, fp, fn = (y_pred * y).sum(), (y_pred * y_flip).sum(), (y_pred_flip * y).sum()
        prec, recall = tp / (tp + fp), tp / (tp + fn)
        return (1 + self.beta ** 2) * (prec * recall) / (self.beta ** 2 * prec + recall)

    def eval_fmeasure(self, y_pred, y):
        max_f = self.f_measure(y_pred, y, 0.0)
        for i in range(1, 11, 1):
            f = self.f_measure(y_pred, y, i * 0.1)
            max_f = f if (f > max_f).data[0] else max_f
        return max_f

    def validation(self):
        avg_mae = 0.0
        self.net.eval()
        for i, data_batch in enumerate(self.val_loader):
            images, labels = data_batch
            images, labels = Variable(images), Variable(labels)
            if self.config.cuda:
                images, labels = images.cuda(), labels.cuda()
            prob_pred = self.net(images)
            avg_mae += self.eval_mae(prob_pred, labels).cpu().data[0]
        self.net.train()
        return avg_mae / len(self.val_loader)

    def test(self):
        avg_mae = 0.0
        for i, data_batch in enumerate(self.test_loader):
            images, labels = data_batch
            images, labels = Variable(images, requires_grad=False), Variable(labels, requires_grad=False)
            if self.config.cuda:
                images, labels = images.cuda(), labels.cuda()
            prob_pred = F.upsample(self.net(images), scale_factor=2, mode='bilinear')
            mae = self.eval_mae(prob_pred, labels).cpu().data[0]
            print("[%d] psnr: %.2f" % (i, mae))
            print("[%d] psnr: %.2f" % (i, mae), file=self.test_output)
            avg_mae += mae
        avg_mae /= len(self.test_loader)
        print('average psnr: %.2f' % avg_mae)
        print('average psnr: %.2f' % avg_mae, file=self.test_output)

    def train(self):
        x = torch.FloatTensor(self.config.batch_size, self.config.n_color, self.config.img_size, self.config.img_size)
        y = torch.FloatTensor(self.config.batch_size, self.config.n_color, self.config.img_size, self.config.img_size)
        if self.config.cuda:
            cudnn.benchmark = True
            x = x.cuda()
            y = y.cuda()
        x = Variable(x)
        y = Variable(y)
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        for epoch in range(self.config.epoch):
            loss_epoch = 0
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                images, labels = data_batch
                if self.config.cuda:
                    images, labels = images.cuda(), labels.cuda()
                x.data.resize_as_(images).copy_(images)
                y.data.resize_as_(labels).copy_(labels)
                y_pred = self.net(x)
                self.eval_fmeasure(y_pred, y)
                iou_loss, sail_loss = self.loss(y_pred, y)
                loss = iou_loss + sail_loss
                loss.backward()
                utils.clip_grad_norm(self.net.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                loss_epoch += loss.cpu().data[0]
                print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (
                    epoch, self.config.epoch, i, iter_num, loss.cpu().data[0]))
                if self.config.visdom:
                    error = OrderedDict([('loss:', loss.cpu().data[0])])
                    self.visual.plot_current_errors(epoch, i / iter_num, error)
            if (epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, self.config.epoch, loss_epoch / iter_num),
                      file=self.log_output)
                if self.config.visdom:
                    avg_err = OrderedDict([('avg_loss', loss_epoch / iter_num)])
                    self.visual.plot_current_errors(epoch, i / iter_num, avg_err, 1)
                    img = OrderedDict([('origin', self.mean + images.cpu()[0]), ('label', labels.cpu()[0][0]),
                                       ('pred_label', y_pred.cpu().data[0][0])])
                    self.visual.plot_current_img(img)
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1))
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_fold)
