'''annotating all abortion speeches as pro-life or pro-choice'''

import os
import pickle
from optparse import OptionParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

from models import build_neural_network

def main():
    data = pd.read_parquet("../data/abortion_speeches.parquet")
    train = pd.read_csv("../data/annotated_data.csv")

    print("flattening tokens...")
    corpus = []
    for _, row in data.iterrows():
        flat_tokens = [token.lower() for sent in row['tokens'] for token in sent]
        corpus.append(' '.join(flat_tokens))

    print("loading precomputed TF-IDF scores...")
    tfidf_scores = pd.read_csv("../data/tfidf_scores.csv", header=0)
    vocab = tfidf_scores["token"].tolist()

    print("vectorizing corpus with precomputed vocabulary...")
    tfidf_matrix = TfidfVectorizer(vocabulary=vocab).fit_transform(corpus)
    
    tf_idf_data = pd.concat([data[['id', 'congress']], pd.DataFrame(tfidf_matrix.toarray())], axis=1)

    print("making predictions...")
    model_path = "../models/tf_idf1-nn_two_relu.pkl"
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    tf_idf_data['labeled_class'] = model.predict(tf_idf_data.drop(columns=['id', 'congress']))

    # checking incorrect predictions on train (should be 0%)
    print("counting mismatches...")
    merged_data = pd.merge(tf_idf_data, train[['id', 'labeled_class']], on='id', how='inner')
    mismatches = merged_data['labeled_class_x'] != merged_data['labeled_class_y']
    mismatch_count = mismatches.sum()
    total_count = len(merged_data)
    mismatch_percentage = (mismatch_count / total_count) * 100

    if mismatch_percentage > 0:
        raise ValueError(f"mismatch percentage is not 0%! it is {mismatch_percentage:.2f}%.")
    
    final_df = tf_idf_data[['id', 'congress', 'labeled_class']]
    
    # saving known labels for train and predicted labels for the rest of the data
    final_df.to_csv("../data/all_predictions.csv", index=False)

if __name__ == '__main__':
    main()
