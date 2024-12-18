'''creating tf-idf features for the data (unigram thru 5-grams)'''

import os
import json
from optparse import OptionParser
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='../data/annotated_data.csv',
                      help='input file: default=%default')
    parser.add_option('--outfile', type=str, default='../data/tf_idf{n}.csv',
                      help='output file for data with tf-idf features: default=%default')

    (options, args) = parser.parse_args()

    infile = options.infile
    outfile = options.outfile
    scores_outfile = options.scores_outfile

    data = pd.read_csv(infile)
    
    corpus = []
    for _, row in data.iterrows():
        tokens = ast.literal_eval(row['tokens'])
        flat_tokens = [token.lower() for sent in tokens for token in sent]
        corpus.append(' '.join(flat_tokens))

    for n in range(1, 6):
        print(f"creating {n}-gram features..")
        vectorizer = TfidfVectorizer(ngram_range=(1, n))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()
            
        pd.concat([data[['id', 'labeled_class']], pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)], axis=1).to_csv(outfile.format(n=n), index=False)

if __name__ == '__main__':
    main()
