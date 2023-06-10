
import torch
from utils import plot_save_conf_matrix
from pprint import pprint


class Meter:
    # Simple class that counts stuff
    def __init__(self):
        self.avg = 0
        self.val = 0
        self.count = 0
        self.sum = 0

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.sum += self.val
        self.count += 1
        self.avg = self.sum / self.count


class ClassificationMeter():
    def __init__(self, classes, filename, eval=False):
        self.correct = 0
        self.total = 0
        self.avg_accuracy = 0
        self.eval = eval
        self.classes = classes
        self.filename = filename
        if self.eval:
            self.preds = []
            self.targets = []

    def reset(self):
        self.correct = 0
        self.total = 0
        self.avg_accuracy = 0

    def update(self, preds, targets):
        self.correct += torch.sum(preds == targets)
        self.total += len(targets)
        self.avg_accuracy = self.correct / self.total
        if self.eval:
            self.preds.extend(preds.tolist())
            self.targets.extend(targets.tolist())

    def get_final_metrics(self):
        metric_dict = {}
        metric_dict['accuracy'] = self.avg_accuracy
        pprint(metric_dict)
        if eval:
            plot_save_conf_matrix(self.preds, self.targets,
                                  self.classes, self.filename)


class SegmentationMeter():
    def __init__(self, classes, filename, eval=False):
        self.correct = 0
        self.total = 0
        self.avg_accuracy = 0
        self.eval = eval
        self.classes = classes
        self.filename = filename
        self.precision = {}
        self.recall = {}
        self.iou = {}
        self.first = True

    def reset(self):
        self.correct = 0
        self.total = 0
        self.avg_accuracy = 0
        self.precision = {}
        self.recall = {}
        self.iou = {}
        self.first = True

    def update(self, preds, targets):
        self.correct += torch.sum(preds == targets)
        self.total += len(targets)
        self.avg_accuracy = self.correct / self.total
        for i, obj_class in enumerate(self.classes):
            pred_class = (preds == i)
            targets_class = (targets == i)
            # Calculate the intersection and union
            intersection = torch.logical_and(pred_class, targets_class).sum()
            union = torch.logical_or(pred_class, targets_class).sum()

            # Calculate the IoU
            iou = intersection.float() / union.float()

            if self.first:
                self.precision[obj_class] = pred_class * \
                    targets_class / torch.sum(pred_class)
                self.recall[obj_class] = pred_class * \
                    targets_class / torch.sum(targets_class)
                self.iou[obj_class] = iou
            else:
                self.precision[obj_class] += pred_class * \
                    targets_class / torch.sum(pred_class)
                self.recall[obj_class] += pred_class * \
                    targets_class / torch.sum(targets_class)

        self.first = False

    def get_final_metrics(self):
        metric_dict = {}
        metric_dict['accuracy'] = self.avg_accuracy
        metric_dict['mAP'] = torch.mean(torch.stack(
            list(self.precision.values())).float())
        metric_dict['mAR'] = torch.mean(
            torch.stack(list(self.recall.values())).float())
        metric_dict['mAF1'] = 2*(self.precision*self.recall) / \
            (self.precision + self.recall)
        metric_dict['mIoU'] = torch.mean(torch.stack(
            list(self.iou.values())).float())
        pprint(metric_dict)
