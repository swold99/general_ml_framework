
import torch
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
    def __init__(self):
        self.correct = 0
        self.avg_accuracy = 0

    def reset(self):
        self.correct = 0
        self.total = 0
        self.avg_accuracy = 0

    def update(self, preds, targets):
        self.correct += torch.sum(preds == targets)
        self.total += len(targets)
        self.avg = self.correct / self.total