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

# Binary Classification
# Review Headers + Body 
# Currently just camera data



# Import data
reviews_df = pd.read_csv('data/amazon_reviews_us_Camera_v1_00.tsv', sep='\t', on_bad_lines='skip', header=0, nrows=500000)

def preprocess_data(df):
    df = df[["star_rating", "review_headline", "review_body"]]

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

    return result_df


reviews_df = preprocess_data(reviews_df)


# review_id,product_id,product_title,product_category,star_rating,helpful_votes,total_votes,verified_purchase,review_headline,review_body,review_date


def clean_data(df, columns: List[str]):
    """Clean up data by removing special characters and making text lowercase"""

    for col in columns:

        # Make sure col is strings
        df[col] = df[col].astype(str)
    
        # Make sure all text is lowercase
        df[col] = df[col].str.lower()

        # Get rid of special characters
        numArticles = len(df[col])
        for i in range(numArticles):
            df[col][i] = re.sub('[^A-Za-z0-9]+', ' ', df[col][i])


    return df


print("Cleaning data...")
reviews_df = clean_data(reviews_df, ["review_body", "review_headline"])


# def tokenize_text(data) -> List[str]:
#     # Tokenize text
#     df = data.copy()

#     df['text_tokenized'] = df['review_body'].apply(word_tokenize)

#     # Get all tokens from true data
#     token_list = []
#     for sublist in df['text_tokenized']:
#         for token in sublist:
#             token_list.append(token)
    
#     return token_list

# Copy df
reviews_df_ML = reviews_df.copy()

# Create random state
random_state = 10

# Extract labels
print("Extracting Labels...")
ratings = reviews_df_ML['star_rating'].values
reviews_df_ML.drop(['star_rating'], axis=1, inplace=True)

# Split data into train-test (80-20 split)
print("Performing train test split...")
X_train, X_test, y_train, y_test = train_test_split(reviews_df_ML, ratings, test_size=0.2, stratify=ratings, random_state=random_state)
# Convert X_train and X_test to series
X_train = X_train.iloc[:,0]
X_test = X_test.iloc[:,0]

# Features
# Bag of words
vec = CountVectorizer()
X_train_BoW = vec.fit_transform(X_train)
X_test_BoW = vec.transform(X_test)

# TF-IDF
print("Performing TF-IDF...")
vec = TfidfVectorizer()
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

# Random forest classifier + TF-IDF
model = RandomForestClassifier(random_state=random_state)
# model = LogisticRegression(random_state=random_state)
# model = MultinomialNB()

print(f"Training Random forest classifier + TF-IDF...")
model.fit(X_train_BoW, y_train)

# Get model predictions
y_pred = model.predict(X_test_BoW)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f'Confusion Matrix for Random Forest Classifier + TF-IDF')
plt.show()

# Get binary classification statistics (acc, precision, recall)
tn, fp, fn, tp = cm.ravel()
tn, fp, fn, tp
accuracy = accuracy_score(y_test, y_pred)
precision = tp/(tp+fp)
precision = precision_score(y_test, y_pred, average='micro')
recall = tp/(tp+fn)
print(f"=========== Random Forest Classifier + TF-IDF ===========")
print(f"Accuraccy: {accuracy}")
print(f"Precision Score: {precision}")
print(f"Recall Score: {recall}", end="\n\n")

# Write results to csv
flag = "a"
if not os.path.exists("output/model_data.csv"): flag = "w"
    
with open("output/model_data.csv", flag) as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Random Forest", "TF-IDF", precision, accuracy])