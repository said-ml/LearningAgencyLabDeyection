import tensorflow as tf
import pandas as pd
from time import time
from zipfile import ZipFile
from scipy.special import softmax

import csv
import numba
from tqdm import tqdm as progress
from tqdm import trange
from datasets import Dataset

from typing import*

import zipfile
from configurations import measure_time
import json
import argparse
from itertools import chain
import pandas as pd
from pathlib import Path

from transformers import AutoTokenizer as tokenizer
from transformers import AutoModelForTokenClassification as model
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification as DataCollator
#from datasets import Dataset
import numpy as np

#getting the data path
zip_file_path=r'c:/users/lg/downloads/compressed/pii-detection-removal-from-educational-data.zip' # change it to fit your path
json_file='train.json'
test_path='test.json'
#getting the paths of config, model.weights metadata...etc of the model pretrainined model DeBERTaV3

config_path=r'C:\Users\LG\Downloads\config.json'
model_weights_path=r'C:\Users\LG\Downloads\model.weights.h5'
metadata_path=r'C:\Users\LG\Downloads\metadata.json'
tokenizer_path=r'\Users\LG\Downloads\tokenizer'


with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(json_file) as json_file:
        data = json.load(json_file)
        data=pd.DataFrame(data)
# getting the traing data from directory
#js_data=pd.read_json(train_data_path)

class processing_data:
      def __init__(self, data=data)->None:
          self.data=data

      @measure_time
      def drop_zero_labels(self):
          print(f'{len(self.data)}is the data size before the  cleaning process')
          def check_zeros(data):
              return any(np.array(data["labels"]) != "O")

          clean_data=self.data[self.data.apply(lambda data:check_zeros(data), axis=1)]
          zero_data=self.data[self.data.apply(lambda data:not check_zeros(data), axis=1)]


          #print(f'{len(clean_data)}is the data significant samples')
          #print(f'{len(zero_data)}is the samples who has no interest')
          #print(json.dumps(filtered_data, indent=4))
          return clean_data

      def add_data(self, data2):
          # clean the data from zeros and concatenated with original data

          assert self.data.shape==data2.shape
          data=np.concatenate(self.data, data2)
          return data



@numba.jit(nonpython=False)
def tokenize(example, tokenizer, label2id, max_length):

    # rebuild text from tokens
    text = []
    labels = []

    for t, l, ws in zip(
        example["tokens"], example["provided_labels"], example["trailing_whitespace"]
    ):
        text.append(t)
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")

    # actual tokenization
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {**tokenized, "labels": token_labels, "length": length}


def predictions():
        probs = []
        for model_path in ['/kaggle/input/pii-data-detection-models/deberta3base_1024']:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
            training_args = TrainingArguments(output_dir=CFG.infer_dir,
                                              per_device_eval_batch_size=1,
                                              report_to="none")
            trainer = Trainer(model=model,
                              args=training_args,
                              data_collator=data_collator,
                              tokenizer=tokenizer)
            test_tokenizer = TestTokenizer(tokenizer)
            tokenized_ds = ds.map(lambda example: test_tokenizer.tokenize(example), num_proc=CFG.num_proc)

            # Apply `softmax` to convert into a float (0~1) that sum up to a total of 1
            prob = softmax(trainer.predict(tokenized_ds).predictions, axis=-1)
            probs.append(prob)
            del model, trainer
            clear_memory()

        # Post-processing predictions by normalize the probs
        predictions = 0.0
        for prob in probs:
            predictions += prob
        predictions /= len(probs)

        return predictions
def submission_csv(predictions):
    '''
    the submission file of the format:
    row_id,   document,  token,  label
    0,        7,         9,      B-NAME_STUDENT
    1,        7,         10,     I-NAME_STUDENT
    2,        10,         0,     B-NAME_STUDENT
    :param data: test data
    :return: submission.csv
    '''
    # create the dataframe from predictions
    df = pd.DataFrame({
        "document": predictions[document],
        "token": predictions[token],
        "label": predictions[label],
        "token_str": predictions[token_str]
    })
    # add Ids of the rows
    df[row_id]=list(range(len(df)))

    #print(df.head())

    submussion_csv=df[["row_id", "document", "token", "label"]].to_csv("submission.csv", index=False)
    return submussion_csv









if __name__=="__main__":
    #print(data.head())
    process_data=processing_data()
    print(process_data.drop_zero_labels().head())








