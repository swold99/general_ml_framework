
import torch
from utils import plot_save_conf_matrix
from pprint import pprint
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment


class Meter:
    """Simple class that counts and tracks average values"""

    def __init__(self):
        self.avg = 0     # Average value
        self.val = 0     # Current value
        self.count = 0   # Number of updates
        self.sum = 0     # Sum of all values

    def reset(self):
        self.val = 0     # Reset current value
        self.sum = 0     # Reset sum of values
        self.count = 0   # Reset count
        self.avg = 0     # Reset average value

    def update(self, val):
        self.val = val                        # Set current value
        self.sum += self.val                  # Add current value to the sum
        self.count += 1                       # Increment count
        self.avg = self.sum / self.count      # Recalculate average


class SuperMeter():
    """Class for tracking metrics for multi-class classification tasks"""

    def __init__(self, classes, filename, eval=False) -> None:
        self.total = 0                         # Total number of samples
        self.eval = eval                       # Flag indicating evaluation mode
        self.classes = classes                 # List of class labels
        self.num_classes = len(classes)        # Number of classes
        self.filename = filename               # File name
        # True positives for each class
        self.tp = torch.zeros([self.num_classes])
        # False positives for each class
        self.fp = torch.zeros([self.num_classes])
        # False negatives for each class
        self.fn = torch.zeros([self.num_classes])
        self.first = True                      # Flag indicating first update

    def reset(self):
        self.total = 0                         # Reset total number of samples
        self.tp = torch.zeros([self.num_classes])   # Reset true positives
        self.fp = torch.zeros([self.num_classes])   # Reset false positives
        self.fn = torch.zeros([self.num_classes])   # Reset false negatives
        self.first = True                      # Reset first flag

    def update(self, preds, targets):
        # Increment total number of samples
        self.total += torch.numel(targets)
        for i in range(self.num_classes):
            # Predicted samples for current class
            pred_class = (preds == i)
            # Ground truth samples for current class
            target_class = (targets == i)

            # Calculate true positives, false positives, and false negatives
            tp = torch.logical_and(pred_class, target_class).sum()
            fp = torch.logical_and(pred_class, ~target_class).sum()
            fn = torch.logical_and(~pred_class, target_class).sum()

            if self.first:
                # Set true positives for current class
                self.tp[i] = tp
                # Set false positives for current class
                self.fp[i] = fp
                # Set false negatives for current class
                self.fn[i] = fn
            else:
                # Accumulate true positives for current class
                self.tp[i] += tp
                # Accumulate false positives for current class
                self.fp[i] += fp
                # Accumulate false negatives for current class
                self.fn[i] += fn

        self.first = False

    def get_final_metrics(self):
        metric_dict = {}
        metric_dict['mAP'] = torch.mean(
            self.tp / (self.tp + self.fp)).item()       # Mean Average Precision
        metric_dict['mAR'] = torch.mean(
            self.tp / (self.tp + self.fn)).item()       # Mean Average Recall
        metric_dict['mAF1'] = torch.mean(
            2*self.tp / (2*self.tp + self.fn + self.fp)).item()   # Mean Average F1-score
        # Calculate additional task-specific metrics
        metric_dict = self.task_specific_metrics(metric_dict)
        pprint(metric_dict)
        return metric_dict

    def task_specific_metrics(self, metric_dict):
        # Additional task-specific metrics can be calculated and added here
        return metric_dict


class ClassificationMeter(SuperMeter):
    """Meter for tracking metrics in classification tasks"""

    def __init__(self, classes, filename, eval=False) -> None:
        super().__init__(classes, filename, eval)
        if self.eval:
            self.preds = []       # List to store predicted labels
            self.targets = []     # List to store ground truth labels

    def reset(self):
        super().reset()
        if self.eval:
            self.preds = []       # Reset predicted labels list
            self.targets = []     # Reset ground truth labels list

    def update(self, preds, targets):
        super().update(preds, targets)    # Call the base class update method
        if self.eval:
            # Append predicted labels to the list
            self.preds.extend(preds.tolist())
            # Append ground truth labels to the list
            self.targets.extend(targets.tolist())

    def get_final_metrics(self):
        # Call the base class method to get metrics
        metric_dict = super().get_final_metrics()
        if self.eval:
            plot_save_conf_matrix(self.preds, self.targets,
                                  self.classes, self.filename, self.num_classes)
            # Plot and save the confusion matrix

        return metric_dict

    def task_specific_metrics(self, metric_dict):
        metric_dict['average accuracy'] = torch.mean(
            (self.total - self.fn - self.fp) / self.total).item()
        # Calculate and add the average accuracy metric
        return metric_dict


class SegmentationMeter(SuperMeter):
    """Meter for tracking metrics in segmentation tasks"""

    def task_specific_metrics(self, metric_dict):
        metric_dict['mIoU'] = torch.mean(
            self.tp / (self.tp + self.fn + self.fp)).item()
        # Calculate and add the mean Intersection over Union (mIoU) metric
        return metric_dict


class DetectionMeter():
    """Meter for tracking metrics in object detection tasks"""

    def __init__(self, iou_threshold=0.5) -> None:
        self.tp = 0                          # True positives
        self.fp = 0                          # False positives
        self.fn = 0                          # False negatives
        # IoU threshold for matching predictions with targets
        self.iou_threshold = iou_threshold
        self.correct_class = 0               # Number of correctly classified objects
        self.total = 0                       # Total number of objects
        self.tot_iou = 0                     # Total IoU of matched predictions

    def reset(self):
        self.tp = 0                          # Reset true positives
        self.fp = 0                          # Reset false positives
        self.fn = 0                          # Reset false negatives
        self.correct_class = 0               # Reset correctly classified objects
        self.total = 0                       # Reset total number of objects
        self.tot_iou = 0                     # Reset total IoU

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_boxes, pred_labels, pred_scores = pred.values()
            target_boxes, target_labels = target.values()
            # Calculate IoU between predicted and target boxes
            iou = box_iou(pred_boxes, target_boxes)
            n_preds = pred_boxes.shape[0]
            n_targets = target_boxes.shape[0]

            if n_preds == 0:
                self.tp += 0        # No true positives
                self.fp += 0        # No false positives
                self.fn += n_targets   # All targets are false negatives
                self.tot_iou += 0  # No IoU for unmatched predictions

            else:
                # Apply the Hungarian algorithm
                pred_idx, target_idx = linear_sum_assignment(-iou)
                # Get matched IoUs
                matched_ious = iou[pred_idx, target_idx]
                # Count true positives
                tp = torch.sum(matched_ious > self.iou_threshold)
                fp = n_preds - tp                                    # Count false positives
                fn = n_targets - tp                                  # Count false negatives

                self.tp += tp        # Accumulate true positives
                self.fp += fp        # Accumulate false positives
                self.fn += fn        # Accumulate false negatives
                self.total += n_targets   # Accumulate total number of objects
                # Accumulate matched IoU values
                self.tot_iou += torch.sum(matched_ious)

            self.total += n_targets   # Accumulate total number of objects

    def get_final_metrics(self):
        metric_dict = {}
        metric_dict['precision'] = (
            self.tp / (self.tp + self.fp)).item()   # Calculate precision
        metric_dict['recall'] = (
            self.tp / (self.tp + self.fn)).item()      # Calculate recall
        # Calculate F1 score
        metric_dict['F1'] = (
            2*self.tp / (2*self.tp + self.fn + self.fp)).item()
        # Calculate average IoU
        metric_dict['IoU'] = (self.tot_iou / self.total).item()
        pprint(metric_dict)
        return metric_dict
