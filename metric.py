import os
import gc
import sys
import random
import json
import numpy as np
import pandas as pd


# ================================================================================================================================ #
def pii_fbeta_score(pred_df, gt_df, beta=5):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """
    #     pred_df = pred_df[pred_df.label!="O"].reset_index(drop=True)
    #     gt_df = gt_df[gt_df.label!="O"].reset_index(drop=True)

    df = pred_df.merge(gt_df, how="outer", on=["document", "token"], suffixes=("_pred", "_gt"))
    df["cm"] = ""

    df.loc[df.label_gt.isna(), "cm"] = "FP"
    df.loc[df.label_pred.isna(), "cm"] = "FN"

    df.loc[(df.label_gt.notna() & df.label_pred.notna()) & (df.label_gt != df.label_pred), "cm"] = "FNFP"  # CHANGED

    df.loc[
        (df.label_pred.notna()) & (df.label_gt.notna()) & (df.label_gt == df.label_pred), "cm"
    ] = "TP"

    FP = (df["cm"].isin({"FP", "FNFP"})).sum()
    FN = (df["cm"].isin({"FN", "FNFP"})).sum()
    TP = (df["cm"] == "TP").sum()
    s_micro = (1 + (beta ** 2)) * TP / (((1 + (beta ** 2)) * TP) + ((beta ** 2) * FN) + FP)

    dic_class = {}
    classes = [c.split('-')[-1] for c in gt_df['label'].unique()]
    for c in classes:
        dx = pred_df[pred_df.label.str.contains(c)].merge(gt_df[gt_df.label.str.contains(c)], how='outer',
                                                          on=['document', "token"], suffixes=('_pred', '_gt'))
        dx["cm1"] = ""

        dx.loc[dx.label_gt.isna(), "cm1"] = "FP"
        dx.loc[dx.label_pred.isna(), "cm1"] = "FN"

        dx.loc[
            (dx.label_gt.notna() & dx.label_pred.notna()) & (dx.label_gt != dx.label_pred), "cm1"] = "FNFP"  # CHANGED

        dx.loc[
            (dx.label_pred.notna()) & (dx.label_gt.notna()) & (dx.label_gt == dx.label_pred), "cm1"
        ] = "TP"

        FP = (dx["cm1"].isin({"FP", "FNFP"})).sum()
        FN = (dx["cm1"].isin({"FN", "FNFP"})).sum()
        TP = (dx["cm1"] == "TP").sum()
        s = (1 + (beta ** 2)) * TP / (((1 + (beta ** 2)) * TP) + ((beta ** 2) * FN) + FP)

        dic_class[c] = s

    return s_micro, dic_class
# =======================================

sample_df = pd.read_csv('/kaggle/input/pii-detection-removal-from-educational-data/sample_submission.csv')
sample_df.shape

sample_df.head()
pii_fbeta_score(sample_df, sample_df,beta=5)
pii_fbeta_score(sample_df, sample_df.sample(10),beta=5)
pii_fbeta_score(sample_df, sample_df.sample(5),beta=5)

# load training data and create reference dataframe ---
reference_data = json.load(open("/kaggle/input/pii-detection-removal-from-educational-data/train.json", "r"))

df = pd.DataFrame(reference_data)[['document', 'tokens', 'labels']].copy()
df = df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
df['token'] = df.groupby('document').cumcount()

label_list = df['label'].unique().tolist()

reference_df = df[df['label'] != 'O'].copy()
reference_df = reference_df.reset_index().rename(columns={'index': 'row_id'})
reference_df = reference_df[['row_id', 'document', 'token', 'label']].copy()

reference_df.head()

reference_df.shape

#make random predictions on a subset of tokens
rng = random.Random(42)
pred_df = reference_df.sample(frac=0.8).sort_values(by='document').reset_index(drop=True)
pred_df['label'] = pred_df['label'].apply(lambda x: rng.choice(label_list) if rng.random() >= 0.5 else x)
pred_df['row_id'] = list(range(len(pred_df)))
pred_df = pred_df[['row_id', 'document', 'token', 'label']].copy()

pred_df.head()

from collections import defaultdict
from typing import Dict


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_metrics(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """

    references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
    predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1]  # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:]  # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:]  # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    return {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "ents_f5": totals.f5,
        "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items() if k != 'O'},
    }