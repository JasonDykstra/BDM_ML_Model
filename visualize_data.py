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

import json
import ast

data_df = pd.read_csv('data/amazon_reviews_us_Camera_v1_00.tsv', sep='\t', on_bad_lines='skip', header=0)

star_rating_distribution = []
star_rating_distribution.append(len(data_df[data_df["star_rating"] == 1]))
star_rating_distribution.append(len(data_df[data_df["star_rating"] == 2]))
star_rating_distribution.append(len(data_df[data_df["star_rating"] == 3]))
star_rating_distribution.append(len(data_df[data_df["star_rating"] == 4]))
star_rating_distribution.append(len(data_df[data_df["star_rating"] == 5]))

star_ratings = [1, 2, 3, 4, 5]

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(1, 1, 1)

sns.barplot(star_ratings, star_rating_distribution, ax=ax)

plt.show()