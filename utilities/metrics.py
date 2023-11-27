import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4): #Change the number of classes-accordingly for each dataset
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(1, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)

            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def mDice(pred_mask, mask, smooth=1e-10, n_classes=4): #Change the number of classes occordingly for each dataset [4,7,6,2]
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        dice_per_class = []
        for clas in range(1, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                dice_per_class.append(np.nan)

            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                dice = 2*(intersect + smooth) / (union + intersect + smooth)
                dice_per_class.append(dice)
        return np.nanmean(dice_per_class)


class TotalDiceIou(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersect = torch.zeros(num_classes)
        self.union = torch.zeros(num_classes)

    def reset(self):
        self.intersect = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, pred_mask, mask):
        with torch.no_grad():
            pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.contiguous().view(-1)
            mask = mask.contiguous().view(-1)

            for clas in range(1, self.num_classes):  # loop per pixel class
                true_class = pred_mask == clas
                true_label = mask == clas
                # 先算出每一个batch的交集和并集，在get_mIoU中去处理所有数据
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()
                self.intersect[clas] += intersect
                self.union[clas] += union
            return

    def get_mIoU(self):
        smooth = 1e-10
        # 计算每个类的IoU
        iou = (self.intersect[1:] + smooth) / (self.union[1:] + smooth)
        return iou.mean().item()

    def get_mdice(self):
        smooth = 1e-10
        dice = 2*(self.intersect[1:] + smooth) / (self.union[1:] + self.intersect[1:] + smooth)
        return dice.mean().item()


class F1PR(object):
    def __init__(self):
        self.predicts = []
        self.targets = []
        self.results = {}

    def update(self, predict, target):
        with torch.no_grad():
            predict = torch.softmax(predict, dim=1)
            score = predict[:, 1]
            self.predicts.append(score)
            self.targets.append(target)
        return

    def reset(self):
        self.predicts = []
        self.targets = []
        self.results = {}

    def get_f1_pr(self, mode='score'):
        self.targets = torch.cat(self.targets)
        self.predicts = torch.cat(self.predicts)
        self.targets = self.targets.cpu().numpy()
        self.predicts = self.predicts.cpu().numpy()

        eps = 1e-9

        # 初始化评估指标
        precision = []
        recall = []
        best_f1, best_p, best_r, best_thr = np.array(0), np.array(0), np.array(0), np.array(0)
        positive, negative = np.sum(self.targets == 1), np.sum(self.targets == 0)
        tp, fp, fn, tn = 0, 0, positive, negative

        # 遍历所有得分的方式计算
        if mode == 'score':
            # 倒序排列
            sort_index = np.argsort(self.predicts)[::-1]
            for i, index in enumerate(sort_index):
                target = self.targets[index]
                if target == 0:
                    fp += 1
                    tn -= 1
                else:
                    tp += 1
                    fn -= 1
                precision.append(tp / (tp + fp))
                recall.append(tp / (tp + fn))
                f1 = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = self.predicts[index]
                    best_p = precision[i]
                    best_r = recall[i]

        if mode == 'fixed':
            thresholds = np.linspace(0., 1., 11)
            for i, thr in enumerate(thresholds):
                tp = ((self.predicts > thr) * (self.targets == 1)).sum().item()
                fp = ((self.predicts > thr) * (self.targets == 0)).sum().item()
                fn = positive - tp
                precision.append(tp / (fp + tp + eps))
                recall.append(tp / (fn + tp + eps))
                f1 = (2 * precision[i] * recall[i]) / (precision[i] + recall[i] + eps)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
                    best_p = precision[i]
                    best_r = recall[i]

        self.results = {'precision': precision, 'recall': recall, 'f1': best_f1, 'best_p': best_p,
                        'best_r': best_r, 'threshold': best_thr}
        return self.results

    def draw(self, save_path):
        plt.clf()
        plt.figure(f"F1: {self.results['f1']}  thr: {self.results['threshold']}")
        plt.title(f"Recall: {self.results['best_r']} Precision: {self.results['best_p']}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(self.results['recall'], self.results['precision'])
        plt.savefig(save_path)
        return
