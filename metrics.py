
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

class SuperMeter():
    def __init__(self, classes, filename, eval=False) -> None:
        self.total = 0
        self.eval = eval
        self.classes = classes
        self.num_classes = len(classes)
        self.filename = filename
        self.tp = torch.zeros([self.num_classes])
        self.fp = torch.zeros([self.num_classes])
        self.fn = torch.zeros([self.num_classes])
        self.first = True

    def reset(self):
        self.total = 0
        self.tp = torch.zeros([self.num_classes])
        self.fp = torch.zeros([self.num_classes])
        self.fn = torch.zeros([self.num_classes])
        self.first = True

    def update(self, preds, targets):
        self.total += torch.numel(targets)
        for i in range(self.num_classes):
            pred_class = (preds == i)
            target_class = (targets == i)
            # Calculate the correctness of prediction
            tp = torch.logical_and(pred_class, target_class).sum()
            fp = torch.logical_and(pred_class, ~target_class).sum()
            fn = torch.logical_and(~pred_class, target_class).sum()

            if self.first:
                self.tp[i] = tp
                self.fp[i] = fp 
                self.fn[i] = fn
            else:
                self.tp[i] += tp
                self.fp[i] += fp 
                self.fn[i] += fn

        self.first = False

    def get_final_metrics(self):
        metric_dict = {}
        metric_dict['mAP'] = torch.mean(self.tp / (self.tp + self.fp)).item()
        metric_dict['mAR'] = torch.mean(self.tp / (self.tp + self.fn)).item()
        metric_dict['mAF1'] = torch.mean(2*self.tp / (2*self.tp + self.fn + self.fp)).item()
        metric_dict = self.task_specific_metrics(metric_dict)
        pprint(metric_dict)
        return metric_dict

    def task_specific_metrics(self, metric_dict):
        return metric_dict

class Aids(SuperMeter):
    def __init__(self, classes, filename, eval=False) -> None:
        super().__init__(classes, filename, eval)

class ClassificationMeter(SuperMeter):
    def __init__(self, classes, filename, eval=False) -> None:
        super().__init__(classes, filename, eval)
        if self.eval:
            self.preds = []
            self.targets = []

    def reset(self):
        super().reset()
        if self.eval:
            self.preds = []
            self.targets = []

    def update(self, preds, targets):
        super().update(preds, targets)
        if self.eval:
            self.preds.extend(preds.tolist())
            self.targets.extend(targets.tolist())

    def get_final_metrics(self):
        metric_dict = super().get_final_metrics()
        if self.eval:
            plot_save_conf_matrix(self.preds, self.targets,
                                  self.classes, self.filename, self.num_classes)
            
        return metric_dict
    
    def task_specific_metrics(self, metric_dict):
        metric_dict['average accuracy'] = torch.mean((self.total - self.fn - self.fp) / self.total).item()
        return metric_dict


class SegmentationMeter(SuperMeter):

    def task_specific_metrics(self, metric_dict):
        metric_dict['mIoU'] = torch.mean(self.tp / (self.tp + self.fn + self.fp)).item()
        return metric_dict