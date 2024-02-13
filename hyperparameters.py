import argparse

from transformers import Trainer
from transformers import TrainingArguments
from transformers import TrainerCallback

# implmenting several types of lr_scheduler
class CustomlLRscheduler(TrainerCallback):
      '''
      implementing a linear learning rate decade
      '''
      def __init__(self, warmup_step, total_step)->None:
          '''
          :param warmup_step:
          :param total_step:
          '''
          assert warmup_step<total_step
          self.warmup_step=warmup_step
          self.total_step=total_step

          # define the inintailizer of the parent class
          super(CustomlLRscheduler, self).__init__()

      def on_epoch_begin(self):
          '''
          :return:
          '''

