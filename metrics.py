
import torch
from utils import plot_save_conf_matrix
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment


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
    
class DetectionMeter():
    def __init__(self, iou_threshold=0.5) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.iou_threshold = iou_threshold
        self.correct_class = 0
        self.total = 0
        self.tot_iou = 0


    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.correct_class = 0
        self.total = 0
        self.tot_iou = 0

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_boxes, pred_labels, pred_scores = pred.values()
            target_boxes, target_labels = target.values()
            iou = box_iou(pred_boxes, target_boxes)
            n_preds = pred_boxes.shape[0]
            n_targets = target_boxes.shape[0]

            if n_preds == 0:
                self.tp += 0
                self.fp += 0
                self.fn += n_targets
                self.tot_iou += 0

            else:
                best_preds = torch.max(iou, dim=1).values
                tp = torch.sum(best_preds > self.iou_threshold)
                fp = n_preds - tp
                gt_matched = (best_preds > self.iou_threshold).any(dim=0).float()
                fn = n_targets - torch.sum(gt_matched)

                self.tp += tp
                self.fp += fp 
                self.fn += fn
                self.total += n_targets
                self.tot_iou += torch.sum(torch.max(iou, dim=0).values)

            self.total += n_targets

            # self.calc_avg_iou(pred_boxes, target_boxes)


    def get_final_metrics(self):
        metric_dict = {}
        metric_dict['precision'] = (self.tp / (self.tp + self.fp)).item()
        metric_dict['recall'] = (self.tp / (self.tp + self.fn)).item()
        metric_dict['F1'] = (2*self.tp / (2*self.tp + self.fn + self.fp)).item()
        metric_dict['IoU'] = (self.tot_iou / self.total).item()
        pprint(metric_dict)
        return metric_dict
    
    def calc_avg_iou(self, pred_boxes, target_boxes):
        n_preds = pred_boxes.shape[0]
        n_targets = target_boxes.shape[0]
        cost_matrix = torch.zeros((n_preds, n_targets))
        for i in range(n_preds):
            pred_box = pred_boxes[i, :].unsqueeze(0)
            for j in range(n_targets):
                target_box = target_boxes[j, :].unsqueeze(0)
                # Calculate the cost (e.g., IoU or distance) between pred_box and target_box
                cost_matrix[i, j] = box_iou(pred_box, target_box)

        # Step 2: Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        iou = 0
        for i, j in zip(row_ind, col_ind):
            iou += box_iou(pred_boxes[i, :].unsqueeze(0), target_boxes[j, :].unsqueeze(0))
            
        self.tot_iou += iou
        
    