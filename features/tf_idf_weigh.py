'''creating vector features: weighted average of word vectors, weighted by tf-idf scores'''

import pandas as pd
import numpy as np
import ast
from optparse import OptionParser
from gensim.models import Word2Vec

def compute_weighted_avg_vector(tokens, model, tfidf_df, speech_id):
    vectors = []
    weights = []
    
    for t in tokens:
        tfidf_score = tfidf_df.loc[tfidf_df['id'] == speech_id].iloc[0].get(t, 0)
        if tfidf_score > 0:
            vectors.append(model.wv[t] * tfidf_score)
            weights.append(tfidf_score)
    
    return np.sum(vectors, axis=0) / np.sum(weights)

def main():
    parser = OptionParser(usage="%prog")
    
    parser.add_option('--infile', type=str, default='../data/annotated_data.csv',
                      help='input file with speeches: default=%default')
    parser.add_option('--tfidf-file', type=str, default='../data/tf_idf1.csv',
                      help='input file with tf-idf scores: default=%default')
    parser.add_option('--custom-model-file', type=str, default='../models/custom_word2vec.model',
                      help='input file with word2vec model: default=%default')
    
    (options, args) = parser.parse_args()
    infile = options.infile
    tfidf_file = options.tfidf_file
    model_file = options.custom_model_file
    
    data = pd.read_csv(infile)
    flattened_tokens = [[word for sublist in ast.literal_eval(speech) for word in sublist]
                        for speech in data['tokens'].tolist()]
    
    model = Word2Vec.load(model_file)
    tfidf_df = pd.read_csv(tfidf_file)
    
    weighted_vectors = []
    for i, tokens_in_speech in enumerate(flattened_tokens):
        weighted_vectors.append(compute_weighted_avg_vector(tokens_in_speech, model, tfidf_df, data.iloc[i]['id']))
    
    data['vector'] = weighted_vectors
    data[['id', 'vector', 'labeled_class']].to_csv('../data/custom_train.csv', index=False)

if __name__ == '__main__':
    main()
