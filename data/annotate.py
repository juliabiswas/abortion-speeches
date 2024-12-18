'''annotating abortion-related speeches as pro-life or pro-choice'''

import os
import json
from optparse import OptionParser
import pandas as pd

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--infile', type=str, default='../data/train.parquet',
                      help='input file: default=%default')
    parser.add_option('--outfile', type=str, default='../data/annotated_data.csv',
                      help='output file: default=%default')
    (options, args) = parser.parse_args()

    infile = options.infile
    outfile = options.outfile
    
    data = pd.read_parquet(infile)
    data['labeled_class'] = None

    for i, row in data.iterrows():
        print(f"\n{row['tokens']}")
        annotation = input("0 for pro-life, 1 for pro-choice: ")

        while annotation not in ['0', '1']:
            print("invalid input. please enter 0 or 1.")
            annotation = input("0 for pro-life, 1 for pro-choice: ")
        
        data.at[i, 'labeled_class'] = int(annotation)

    data.to_csv(outfile, index=False)
            
    print(f"labeled data saved to {outfile}")
    
if __name__ == '__main__':
    main()

