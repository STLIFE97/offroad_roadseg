import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adjust_learning_rate(object):
    def __init__(self, config):
        self.config = config

    def adjust_lr(self, optimizer=None, iter_lr=None, total_iters=None, epoch=None):
        if self.config.lr_policy == 'poly':
            current_lr = self.lr_poly(iter_lr, total_iters)
        elif self.config.lr_policy == 'lambda':
            current_lr = self.lr_lambda(epoch)
        elif self.config.lr_policy == 'cosine':
            current_lr = self.lr_cosine(iter_lr, total_iters)

        optimizer.param_groups[0]['lr'] = current_lr

        return current_lr

    def lr_poly(self, i_iter, total_iters):
        lr = self.config.learning_rate * ((1 - float(i_iter) / total_iters) ** self.config.power)
        return lr

    def lr_lambda(self, epoch):
        lr = self.config.learning_rate * self.config.lr_gamma ** (epoch // self.config.lr_decay_epochs)
        return lr

    def lr_cosine(self, step, total_iters):
        start_lr = 0.00001
        max_lr = self.config.learning_rate  # max_lr设置2e-4到5e-4，与batchsize大小有关.batchsize越大,学习率也可以更大
        min_lr = 1e-6
        warm_steps = self.config.warm_steps  # warm_steps设置1000-2000的整数
        total_steps = total_iters

        if step < warm_steps:
            lr = ((max_lr - start_lr) * step) / warm_steps + start_lr
        else:
            lr = max_lr * (math.cos(math.pi * (step - warm_steps) / (total_steps - warm_steps)) + 1) / 2

        lr = max(lr, min_lr)

        return lr


class LossFn(nn.Module):
    def __init__(self):
        super(LossFn, self).__init__()

        self.loss = nn.MSELoss()


    def forward(self, iou_tokens, preds, gt_mask):
        pass


import numpy as np


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(bool))

    # k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)


def getScores(num_classes,conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    if num_classes == 2:
        with np.errstate(divide='ignore',invalid='ignore'):
            globalacc = np.diag(conf_matrix).sum() / np.float64(conf_matrix.sum())
            classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float64)
            classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float64)
            # IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
            IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(float)

            
            pre = classpre[1]
            recall = classrecall[1]
            iou = IU[1]
            F_score = 2*(recall*pre)/(recall+pre)
        return globalacc, pre, recall, F_score, iou
    else :
        with np.errstate(divide='ignore',invalid='ignore'):
            globalacc = np.diag(conf_matrix).sum() / np.float64(conf_matrix.sum())
            classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float64)
            classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float64)
            # IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
            IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(float)

            
            pre_list = []
            recall_list = []
            iou_list = []
            F_score_list = []

            for i in range(num_classes):
                pre_i = classpre[i]
                recall_i = classrecall[i]
                iou_i = IU[i]
                F_score_i = 2 * (recall_i * pre_i) / (recall_i + pre_i)

                pre_list.append(pre_i)
                recall_list.append(recall_i)
                iou_list.append(iou_i)
                F_score_list.append(F_score_i)

            pre_avg = np.mean(pre_list)
            recall_avg = np.mean(recall_list)
            iou_avg = np.mean(iou_list)
            F_score_avg = np.mean(F_score_list)

            return globalacc, pre_avg, recall_avg, F_score_avg, iou_avg, pre_list, recall_list, iou_list, F_score_list