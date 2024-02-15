import tensorflow as tf
import pandas as pd
from time import time
from zipfile import ZipFile
from scipy.special import softmax
from itertools import chain
from functools import partial

from datasets import Dataset
from  datasets import features


try:
    from seqeval.metrics import recall_score as recall
    from seqeval.metrics import precision_score as precision
    from seqeval.metrics import f1_score as f1

except ModuleNotFoundError:
    from sklearn.metrics import recall_score as recall
    from sklearn.metrics import precision_score as precision
    from sklearn.metrics import f1_score as f1


import csv
import numba
from tqdm import tqdm as progress
from tqdm import trange
from datasets import Dataset

from typing import*

import zipfile
from Configuration import measure_time
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


# set directory
output_dir='output'
TRAINING_MAX_LENGTH = 1024

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

tokenizer = tokenizer.from_pretrained(TRAINING_MODEL_PATH)

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

target = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]


ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [str(x["document"]) for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
})
ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH}, num_proc=3)

####################-----------------------#################
x=ds[0]
for t, l in zip(x['tokens'], x['provided_labels'] ):
    if l!=0:
        print((t, l))
    print("*"*10)
for t, l in zip(tokenizer.convert_ids_to_tokens(x['input_ids']), x['labels']):
    if id2label[1]!='0':
        print(t, id2label[1])

# Model training:
class MoelTraining:
      def __init__(self, model, tokenizer)->None:
          self.model=model
          self.tokenizer=tokenizer
          self.DataCollator=DataCollator


      # compute the model performance the metric
      def compute_metric(self, p, all_labels):
          predictions, labels=p
          predictions=np.argmax(predictions, axis=2)

          # Remove and ignore index (special tokens)
          true_predictions=[
              [all_labels[p] for (p, l) in zip(predictions, label) if l!=-100]
              for predictions, label in zip(predictions, labels)
          ]
          true_labels = [
              [all_labels[l] for (p, l) in zip(predictions, label) if l != -100]
              for predictions, label in zip(predictions, labels)
          ]

          recall=recall(true_labels, true_predictions)
          precision=precision(true_labels, true_predictions)
          f1_score=(1+5*5)*recall*precision/(5*5*precision+recall)

          results={
              'recall':recall,
              'precision':precision,
              'f1_score':f1_score
          }

          return results

model=model.from_pretrained(model_weights_path,
                            num_labels=len(all_labels),
                            id2label=id2label,
                            label2id=label2id,
                            ignore_mismatch_sizes=True,
                            )

collarator=DataCollator(tokenizer, pad_to_multiple_of=16)

# I actually chose to not use any validation set. This is only for the model I use for submission.
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    report_to="none",
    evaluation_strategy="no",
    do_eval=False,
    save_total_limit=1,
    logging_steps=20,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
)


def predictions():
        probs = []
        for model_path in [model_weights_path]:
            tokenizer =  tokenizer.from_pretrained(model_path)
            model = model.from_pretrained(model_path)
            data_collator = DataCollator(tokenizer, pad_to_multiple_of=16)
            training_args = TrainingArguments(output_dir=output_dir,
                                              per_device_eval_batch_size=1,
                                              report_to="none")
            trainer = Trainer(model=model,
                              args=training_args,
                              data_collator=data_collator,
                              tokenizer=tokenizer)
            test_tokenizer = tokenizer(tokenizer)
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







#############-----------------####################
trainer.save_model('debertavBbase_1024')
tokenizer.save_model('debertavBbase_1024')

if __name__=="__main__":
    #print(data.head())
    process_data=processing_data()
    print(process_data.drop_zero_labels().head())








