import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.backends import cudnn
from nldf import build_model, weights_init
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
        self.device = torch.device('cpu')
        if self.config.cuda:
            cudnn.benchmark = True
            self.device = torch.device('cuda')
        if config.visdom:
            self.visual = Viz_visdom("NLDF", 1)
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
        if self.config.mode == 'train': self.loss = Loss(self.config.area, self.config.boundary)
        self.net = self.net.to(self.device)
        if self.config.cuda and self.config.mode == 'train': self.loss = self.loss.cuda()
        self.net.train()
        self.net.apply(weights_init)
        if self.config.load == '': self.net.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '': self.net.load_state_dict(torch.load(self.config.load))
        self.optimizer = Adam(self.net.parameters(), self.config.lr)
        self.print_network(self.net, 'NLDF')

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def clip(self, y):
        return torch.clamp(y, 0.0, 1.0)

    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    # TODO: write a more efficient version
    def eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
        return prec, recall

    def validation(self):
        avg_mae = 0.0
        self.net.eval()
        for i, data_batch in enumerate(self.val_loader):
            with torch.no_grad():
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                prob_pred = self.net(images)
            avg_mae += self.eval_mae(prob_pred, labels).cpu().item()
        self.net.train()
        return avg_mae / len(self.val_loader)

    def test(self, num):
        avg_mae, img_num = 0.0, len(self.test_loader)
        avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
        for i, data_batch in enumerate(self.test_loader):
            with torch.no_grad():
                images, labels = data_batch
                shape = labels.size()[2:]
                images = images.to(self.device)
                prob_pred = F.interpolate(self.net(images), size=shape, mode='bilinear', align_corners=True).cpu()
            mae = self.eval_mae(prob_pred, labels)
            prec, recall = self.eval_pr(prob_pred, labels, num)
            print("[%d] mae: %.4f" % (i, mae))
            print("[%d] mae: %.4f" % (i, mae), file=self.test_output)
            avg_mae += mae
            avg_prec, avg_recall = avg_prec + prec, avg_recall + recall
        avg_mae, avg_prec, avg_recall = avg_mae / img_num, avg_prec / img_num, avg_recall / img_num
        score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
        score[score != score] = 0  # delete the nan
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
        print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()), file=self.test_output)

    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        best_mae = 1.0 if self.config.val else None
        for epoch in range(self.config.epoch):
            loss_epoch = 0
            for i, data_batch in enumerate(self.train_loader):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                x, y = data_batch
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.net(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                self.optimizer.step()
                loss_epoch += loss.cpu().item()
                print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (
                    epoch, self.config.epoch, i, iter_num, loss.cpu().item()))
                if self.config.visdom:
                    error = OrderedDict([('loss:', loss.cpu().item())])
                    self.visual.plot_current_errors(epoch, i / iter_num, error)
            if (epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, self.config.epoch, loss_epoch / iter_num),
                      file=self.log_output)
                if self.config.visdom:
                    avg_err = OrderedDict([('avg_loss', loss_epoch / iter_num)])
                    self.visual.plot_current_errors(epoch, i / iter_num, avg_err, 1)
                    img = OrderedDict([('origin', self.mean + x.cpu()[0]), ('label', y.cpu()[0][0]),
                                       ('pred_label', y_pred.cpu()[0][0])])
                    self.visual.plot_current_img(img)
            if self.config.val and (epoch + 1) % self.config.epoch_val == 0:
                mae = self.validation()
                print('--- Best MAE: %.4f, Curr MAE: %.4f ---' % (best_mae, mae))
                print('--- Best MAE: %.4f, Curr MAE: %.4f ---' % (best_mae, mae), file=self.log_output)
                if best_mae > mae:
                    best_mae = mae
                    torch.save(self.net.state_dict(), '%s/models/best.pth' % self.config.save_fold)
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1))
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_fold)
