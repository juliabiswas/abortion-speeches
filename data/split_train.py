'''splits off a subset of abortion speeches to serve as training set'''

import pandas as pd

abortion = pd.read_parquet('../data/abortion_speeches.parquet')
train = abortion[abortion.index % (len(abortion) // 102) == 0]
train.to_parquet('../data/train.parquet', index=False, compression='snappy')
