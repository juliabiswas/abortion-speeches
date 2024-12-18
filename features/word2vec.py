'''developing custom word2vec model on full corpus of congressional speeches'''

import pandas as pd
import pyarrow.parquet as pq
from gensim.models import Word2Vec

def read_from_file_in_chunks(file_path, chunk_size):
    dataset = pq.ParquetDataset(file_path)
    
    num_rows = sum(p.count_rows() for p in dataset.fragments)
        
    for start_row in range(0, num_rows, chunk_size):
        end_row = min(start_row + chunk_size, num_rows)
        table = dataset.read().slice(start_row, end_row - start_row)
        
        data = table.to_pandas()
        print(f"reading {start_row} to {end_row} rows from {file_path}...")
        
        flattened_tokens = [[word for sublist in speech for word in sublist] for speech in data['tokens'].tolist()]
        
        yield flattened_tokens

def main():
    file_paths = [f'../data/all_speeches{i}.parquet' for i in range(307)]
    model = Word2Vec(vector_size=100, window=5, min_count=5, sg=0, workers=4, negative=10, epochs=10)
    
    sentences_accumulated = []
    target_rows = 500000
    total_rows_accumulated = 0
    
    vocabulary_initialized = False
    
    for file_path in file_paths:
        chunk_iter = read_from_file_in_chunks(file_path, target_rows)
        
        for chunk in chunk_iter:
            sentences_accumulated.extend(chunk)
            total_rows_accumulated += len(chunk)
            
            if total_rows_accumulated >= target_rows:
                print(f"accumulated {total_rows_accumulated} rows, updating model...")
                
                if not vocabulary_initialized:
                    model.build_vocab(sentences_accumulated, update=False)
                    vocabulary_initialized = True
                else:
                    model.build_vocab(sentences_accumulated, update=True)
                
                model.train(sentences_accumulated, total_examples=model.corpus_count, epochs=model.epochs)
                
                sentences_accumulated = []
                total_rows_accumulated = 0
    
    if sentences_accumulated:
        print(f"final batch of {total_rows_accumulated} rows, updating model...")
        model.build_vocab(sentences_accumulated, update=True)
        model.train(sentences_accumulated, total_examples=model.corpus_count, epochs=model.epochs)
    
    outfile = '../models/custom_word2vec.model'
    model.save(outfile)
    print(f"model saved to {outfile}")

if __name__ == '__main__':
    main()
