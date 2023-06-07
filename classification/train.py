import os
import sys

from networks.create_model import create_classification_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_classses import Trainer, Evaluator
from metrics import ClassificationMeter

class ClassificationTrainer(Trainer):
    def model_factory(self):
        return create_classification_model()
    
    def init_metrics(self):
        super().init_metrics()
        self.train_f1_list = []
        self.val_f1_list = []


    def task_metrics(self):
        return ClassificationMeter()
    
    def print_metrics(self):
        accuracy = self.metrics['accuracy']
        print(f'accuracy: {accuracy:.4f}')

    
class ClassificationEvaluator(Evaluator):
    def task_metrics(self):
            return ClassificationMeter()
    
