import argparse

from transformers import Trainer
from transformers import TrainingArguments
from transformers import TrainerCallback

# implmenting several types of lr_scheduler
class CustomlLRscheduler(TrainerCallback):
      def __init__(self):
          pass