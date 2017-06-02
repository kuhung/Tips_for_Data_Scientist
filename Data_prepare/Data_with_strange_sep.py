# Author：kuhung （https://github.com/kuhung）
# Create time：2017/06/02
# Source: https://github.com/kuhung/data_fun/blob/master/weibo_forward_EDA.ipynb



with open('../data/trainRepost.txt', 'r') as f:  
    data = f.readlines()  
    trainRepost = []
    for line in data:  
        line=line.strip('\r\n')
        trainRepost.append(line.split('\x01'))       

del data
