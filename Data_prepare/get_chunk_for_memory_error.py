# Author kuhung(https://github.com/kuhung)
# Create time: 2017/05/31

import pandas as pd
reader = pd.read_csv('../data/weibo_dc_parse2015_link_filter.txt', iterator=True,sep='\t',header=None)
try:
    df = reader.get_chunk(100000)
    df.columns = ['weiboID','fans']
    df['fans'] =  df['fans'].astype(str).apply(lambda x: (len(x)-1)/7)
    
except StopIteration:
    print "Iteration is stopped."
    
    
loop = True
chunkSize = 100000
chunks = []
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)        
        chunks.append(chunk)
        
        chunk.columns = ['authorID','fans']
        chunk['fans'] =  chunk['fans'].astype(str).apply(lambda x: (len(x)-1)/7)
    except StopIteration:
        loop = False
        print "Iteration is stopped."
df = pd.concat(chunks, ignore_index=True)

df.to_csv('../data/fansCount.csv',index=False)
