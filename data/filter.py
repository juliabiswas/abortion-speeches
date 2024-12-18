'''filtering for abortion-related speeches'''

import os
import json
from optparse import OptionParser
import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from query_terms import queries

BATCH_SIZE = 50000

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--dir', type=str, default='../data',
                      help='input directory: default=%default')
    parser.add_option('--abortion-outfile', type=str, default='../data/abortion_speeches.parquet',
                      help='abortion output file: default=%default, must have same directory as other output files')
    parser.add_option('--all-speeches-outfile', type=str, default='../data/all_speeches.parquet',
                      help='all_speeches output file: default=%default, must have same directory as other output files')

    (options, args) = parser.parse_args()

    input_dir = options.dir
    abortion_outfile = options.abortion_outfile
    all_speeches_outfile = options.all_speeches_outfile

    outdir = os.path.split(abortion_outfile)[0]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    files = []
    for congress in range(43, 119): #43rd to 118th congress (inclusive)
        files.append(os.path.join(input_dir, 'speeches_' + str(congress).zfill(3) + '.jsonlist'))
    files.sort()

    filtered_count = 0
    total_count = 0
    
    extension = ".parquet"
    file_index = 0
    
    all = pd.DataFrame(columns=['congress', 'id', 'sents', 'tokens'])
    abortion = pd.DataFrame(columns=['congress', 'id', 'sents', 'tokens'])
    
    #filter
    for infile in files:
        print(f"processing {infile}...")
        basename = infile.split('/')[-1].split('.')[0]
        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        congress = int(lines[0]['infile'].split('_')[-1])

        for i, line in enumerate(tqdm(lines)):
            tokenized_sents = line['tokens']
            new_row = pd.DataFrame([{'congress': congress, 'id': line['id'], 'sents': line['sents'], 'tokens': tokenized_sents}])
            all = pd.concat([all, new_row], ignore_index=True)
            
            match = False
            
            for tokens in tokenized_sents:
                if does_not_match_excluded(tokens, queries) and match_tokens(tokens, queries):
                    print(tokens)
                    match = True
                    break
                    
            if match:
                filtered_count += 1
                abortion = pd.concat([abortion, new_row], ignore_index=True)

            total_count += 1
            
            if all.shape[0] >= BATCH_SIZE:
                all.to_parquet(all_speeches_outfile[:-len(extension)] + f'{file_index}' + extension, index=False, compression='snappy')
                all = all.drop(all.index)
                file_index+=1
                print(f"saved file {file_index}")
                
    if not all.empty:
        all.to_parquet(all_speeches_outfile[:-len(extension)] + f'{file_index}' + extension, index=False, compression='snappy')
        print(f"saved file {file_index}")
    
    abortion.to_parquet(abortion_outfile, index=False, compression='snappy')
                
    print(f"filtered {filtered_count} total abortion-related speeches, out of {total_count} total speeches")
    
def does_not_match_excluded(tokens, query_terms):
    """
    determines if a set of tokens has any of the "excluded" query terms
    
    params
    - tokens: a list of tokens
    - query_terms: a set of query terms from query_terms.py
    
    returns true if there are no matches; otherwise, false
    """
    prefix_lengths = [3, 6]

    for i in prefix_lengths:
        prefixes = set([t[:i] for t in tokens])
        if len(prefixes.intersection(query_terms[f'e{i}'])) > 0:
            return False
        
    return True
    
def match_tokens(tokens, query_terms):
    """
    determines if a set of tokens matches a set of query terms
    
    params
    - tokens: a list of tokens
    - query_terms: a set of query terms from query_terms.py
    
    returns true if there's a match; otherwise, false
    """

    tokens = [t.lower() for t in tokens]

    # progressively compare tokens to prefixes
    prefix_lengths = [8, 9, 11, 13]

    for i in prefix_lengths:
        prefixes = set([t[:i] for t in tokens])
        if len(prefixes.intersection(query_terms[f'p{i}'])) > 0:
            return True
            
    # look for exact bigram matches
    bigrams = [tokens[i-1] + ' ' + tokens[i] for i in range(1, len(tokens))]
    overlap = list(set(bigrams).intersection(query_terms['exact_bigrams']))
    if len(overlap) > 0:
        return True
    
    # look for exact trigram matches
    trigrams = [tokens[i-2] + ' ' + tokens[i-1] + ' ' + tokens[i] for i in range(2, len(tokens))]
    overlap = list(set(trigrams).intersection(query_terms['exact_trigrams']))
    if len(overlap) > 0:
        return True
        
    # look for exact quadgram matches
    quadgrams = [tokens[i-3] + ' ' + tokens[i-2] + ' ' + tokens[i-1] + ' ' + tokens[i] for i in range(3, len(tokens))]
    overlap = list(set(trigrams).intersection(query_terms['exact_quadgrams']))
    if len(overlap) > 0:
        return True

    return False

if __name__ == '__main__':
    main()

