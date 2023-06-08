
import torch
from utils import plot_save_conf_matrix
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
        self.avg_accuracy = 0
        self.eval = eval
        self.classes = classes
        self.filename = filename
        if eval:
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
        if eval:
            self.preds.extend(preds.tolist())
            self.targets.extend(targets.tolist())

    def get_final_metrics(self, classes, filename):
        metric_dict = {}
        metric_dict['accuracy'] = self.avg_accuracy
        plot_save_conf_matrix(self.preds, self.labels, self.classes, self.filename)

class SegmentationMeter():
    def __init__(self, classes, filename, eval=False):
        self.correct = 0
        self.avg_accuracy = 0
        self.eval = eval
        self.classes = classes
        self.filename = filename
        if eval:
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
        if eval:
            self.preds.extend(preds.tolist())
            self.targets.extend(targets.tolist())

    def get_final_metrics(self, classes, filename):
        metric_dict = {}
        metric_dict['accuracy'] = self.avg_accuracy
        plot_save_conf_matrix(self.preds, self.labels, self.classes, self.filename)

