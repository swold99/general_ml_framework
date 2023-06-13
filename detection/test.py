from time import time

import torch
from torchvision import transforms
from tqdm import tqdm

from base_classes import Evaluator, Trainer
from custom_transforms import Identity, TrivialAugmentSWOLD
from metrics import DetectionMeter
from networks.create_model import create_detection_model
from utils import load_model_state, move_to, show_detection_imgs, collate_fn
from create_dataset import create_dataset
from torchvision.ops import nms


class DetectionEvaluator(Evaluator):


    def __init__(self, savename, data_folder, params, transform_params) -> None:
        self.score_threshold = params['score_threshold']
        self.iou_threshold = params['iou_threshold']
        super().__init__(savename, data_folder, params, transform_params)

    def task_metrics(self):
        return DetectionMeter(iou_threshold=self.iou_threshold)

    def compose_transforms(self, transform_params, label_is_space_invariant=True):
        # Composes all wanted transforms into a single transform.
        trivial_augment = transform_params['trivial_augment']
        resize = transform_params['resize']
        input_tsfrm = transforms.Compose([transforms.ToTensor()])
        target_tsfrm = Identity()

        if resize is not None:
            input_tsfrm = transforms.Compose(
                [input_tsfrm, transforms.Resize(resize, antialias=True)])
        if trivial_augment:
            input_tsfrm = transforms.Compose(
                [input_tsfrm, TrivialAugmentSWOLD(label_is_space_invariant=label_is_space_invariant)])

        tsfrm = {'input': input_tsfrm, 'target': target_tsfrm}
        return tsfrm

    def preprocess_data(self, inputs, targets):
        tmp = []
        for element in targets:
            annotation = element['annotation']
            objects = annotation['object']
            scaling = [float(annotation['size']['height'])/self.im_size[1],
                       float(annotation['size']['width'])/self.im_size[0]]
            N = len(objects)
            boxes = torch.empty((N, 4), dtype=torch.float)
            labels = torch.empty((N), dtype=torch.int64)
            for i, object in enumerate(objects):
                bndbox = object['bndbox']
                box = [bndbox['xmin'], bndbox['ymin'],
                       bndbox['xmax'], bndbox['ymax']]
                boxes[i, :] = torch.FloatTensor(
                    [float(val)/scaling[1-(j % 2)] for j, val in enumerate(box)])
                labels[i] = self.classes.index(object['name'])
            tmp.append({'boxes': boxes, 'labels': labels})

        return move_to(list(inputs), self.device), move_to(tmp, self.device)
    
    def dataloader_factory(self, params):
        image_dataset = create_dataset(params['use_datasets'], params['quicktest'],
                                       'test', self.tsfrm)
        self.dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], collate_fn=collate_fn)

    def model_factory(self):
        model = create_detection_model(
            self.network, self.device, self.num_classes)
        name = self.savename + ".pth"
        self.model = load_model_state(model, name)

    def show_images(self, inputs, targets, preds):
        show_detection_imgs(inputs, targets, preds)

    def forward_pass(self, input, targets):
        return self.model(input)

    def process_model_out(self, outputs, device):
        outputs = move_to(outputs, device)
        cleaned_output = []
        for output in outputs:
            pred_boxes, pred_labels, pred_scores = output.values()
            good_pred_idx = pred_scores > self.score_threshold
            pred_boxes, pred_scores, pred_labels = self.remove_bad_boxes(pred_boxes, pred_scores, pred_labels, good_pred_idx)

            keep = nms(pred_boxes, pred_scores, iou_threshold=0.5)
            pred_boxes, pred_scores, pred_labels = self.remove_bad_boxes(pred_boxes, pred_scores, pred_labels, keep)

            cleaned_output.append({'boxes': pred_boxes, 'labels':pred_labels,'scores':pred_scores,})
        return cleaned_output

    def test_loop(self):
        self.model.eval()
        self.metrics.reset()
        self.losses.reset()
        prog_bar = tqdm(self.dataloader)
        times = []

        for batch_idx, item in enumerate(prog_bar):
            inputs, targets = item

            inputs, targets = self.preprocess_data(inputs, targets)

            with torch.no_grad():
                t1 = time()
                outputs = self.forward_pass(inputs, targets)
                t2 = time()
                times.append(t2-t1)
                preds = self.process_model_out(outputs, device='cpu')
                targets = move_to(targets, 'cpu')
                inputs = move_to(inputs, 'cpu')
            if self.show_test_imgs:
                self.show_images(inputs, preds, targets)

            self.metrics.update(preds, targets)

        metric_dict = self.metrics.get_final_metrics()
        print("Average inference time: ", torch.mean(
            torch.tensor(times)/self.batch_size).item(), "s")

        return metric_dict
    
    def remove_bad_boxes(self, boxes, scores, labels, keep_idx):
        boxes = boxes[keep_idx, :]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
        return boxes, scores, labels
