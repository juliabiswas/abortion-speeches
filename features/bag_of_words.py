'''creating bag of words features for the data'''

import os
import json
from optparse import OptionParser
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='../data/annotated_data.csv',
                      help='input file: default=%default')
    parser.add_option('--outfile', type=str, default='../data/bag_of_words',
                      help='output file prefix: default=%default')
    
    (options, args) = parser.parse_args()

    infile = options.infile
    outfile_prefix = options.outfile

    data = pd.read_csv(infile)
    
    for n in range(1, 6):
        bag_of_words = []
        
        for _, row in data.iterrows():
            row_bow = {}
            
            for sent in ast.literal_eval(row['tokens']):
                for ngram in [' '.join(ngram) for ngram in zip(*[[token.lower() for token in sent][i:] for i in range(n)])]:
                    row_bow[ngram] = row_bow.get(ngram, 0) + 1

            row_data = {'id': row['id'], 'labeled_class': row['labeled_class']}
            row_data.update(row_bow)
            
            bag_of_words.append(row_data)
        
        bag_of_words_df = pd.DataFrame(bag_of_words)
        bag_of_words_df.fillna(0, inplace=True)

        outfile = f"{outfile_prefix}{n}.csv"
        bag_of_words_df.to_csv(outfile, index=False)
        print(f"bag of {n}-grams saved to {outfile}")

if __name__ == '__main__':
    main()
