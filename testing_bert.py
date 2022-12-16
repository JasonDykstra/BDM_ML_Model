import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import string, re
from typing import *
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag_sents, pos_tag
import os
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from multiprocessing.pool import RUN
from random import random
from tkinter.tix import MAX
from unittest.util import _MAX_LENGTH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from wordcloud import WordCloud
import time
import re
import nltk
import csv
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag_sents, pos_tag
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import gensim
import gensim.downloader as gensim_api
import torch
from transformers import pipeline
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import ast

# Goal: Train an NLP model to predict whether or not a review will be positive (3+ stars) or negative (1-2 stars)


# Import data
reviews_df = pd.read_csv('data/amazon_reviews_us_Camera_v1_00.tsv', sep='\t', on_bad_lines='skip', header=0, nrows=20000)

def preprocess_data(df):
    df = df[["star_rating", "review_body"]]

    # Balance data

    ones_df = df[df["star_rating"] == 1]
    twos_df = df[df["star_rating"] == 2]
    threes_df = df[df["star_rating"] == 3]
    fours_df = df[df["star_rating"] == 4]
    fives_df = df[df["star_rating"] == 5]

    min_len = min(ones_df.shape[0], twos_df.shape[0], threes_df.shape[0], fours_df.shape[0], fives_df.shape[0])

    # Sample size for "Bad" reviews (1, 2, and 3 stars)
    bad_sample_len = int((min_len)*(2/3))

    # Samele size for "Good" reviews (4 and 5 stars)
    good_sample_len = min_len

    ones_df = ones_df.sample(n=bad_sample_len)
    twos_df = twos_df.sample(n=bad_sample_len)
    threes_df = threes_df.sample(n=bad_sample_len)
    fours_df = fours_df.sample(n=good_sample_len)
    fives_df = fives_df.sample(n=good_sample_len)

    # Convert star ratings of 1-3 to 0 and 4-5 to 1 for "Good" and "Bad"
    ones_df = ones_df.assign(star_rating = 0)
    twos_df = twos_df.assign(star_rating = 0)
    threes_df = threes_df.assign(star_rating = 0)
    fours_df = fours_df.assign(star_rating = 1)
    fives_df = fives_df.assign(star_rating = 1)

    # Combine the results all back together and shuffle them up
    result_df = pd.concat([ones_df, twos_df, threes_df, fours_df, fives_df], ignore_index=True)
    result_df = result_df.sample(frac=1)

    # Convert all reviews to strings
    result_df["review_body"] = result_df["review_body"].astype(str)
    

    return result_df


MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, do_lower_case=True)
bert_finetuning_df = preprocess_data(reviews_df)

# 80-20 train test split
text = bert_finetuning_df["review_body"]
labels = bert_finetuning_df["star_rating"]
X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.2)
X_train = X_train.reset_index(drop=True)
X_train.to_csv("output/X_train2.csv")
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Tokenize text
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=MAX_LEN)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=MAX_LEN)


class GroupDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = GroupDataset(X_train, y_train)
test_dataset = GroupDataset(X_test, y_test)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}


training_arguments = TrainingArguments(
    output_dir='./output',
    num_train_epochs=1,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    warmup_steps=100,
    logging_dir='./logs',
    load_best_model_at_end=True,
    logging_steps=200,
    save_steps=200,
    evaluation_strategy='steps',
)

trainer = Trainer(
    model=bert_model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Fine tuning
try:
    trainer.train()
except:
    pass

model_path = 'senitment-analysis-bert-base-uncased'



# Save model and tokenizer
bert_model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print("Model and tokenizer saved!")

# LOAD THE FINE-TUNED BERT MODEL AND TOKENIZER FROM LOCAL DRIVE
# Uncomment the two lines below to save model and tokenizer
# bert_model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizerFast.from_pretrained(model_path)


def get_prediction(text, convert_to_labels=False):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    # perform inference to our model
    outputs = bert_model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        1: "negative",
        2: "negative",
        3: "negative",
        4: "positive",
        5: "positive" 
    }
    if convert_to_labels:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())
    

curr = 0
curr_percent = 0
bert_predictions = []
for doc in bert_finetuning_df['review_body']:

    prediction = get_prediction(doc)
    bert_predictions.append(prediction)
    curr += 1

    if curr % int(bert_finetuning_df.shape[0]/100) == 0:
        curr_percent += 1
        print(f"{curr_percent}%")


bert_predictions = pd.DataFrame(bert_predictions, columns = ['Pred'])
bert_predictions['Real'] = bert_finetuning_df['star_rating']


# Evaluate BERT model (same as above)
y_pred = bert_predictions['Pred']
y_test = bert_predictions['Real']

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
# tn, fp, fn, tp = cm.ravel()
# p = tp/(tp+fp)
# r = tp/(tp+fn)
print("BERT results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(f"Precision: {p}")
# print(f"Recall: {r}")
# print(f"F1 Score: {(2*p*r)/(p+r)}")