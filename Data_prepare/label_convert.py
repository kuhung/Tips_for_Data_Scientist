


import pandas as pd
import os

"
For label in the form of file name
"
train_path = 'K:\\Cancer\\img\\train'
ID = []
Type = []
for count in range (1,4):
    path = 'K:\\Cancer\\img\\train\\Type_%s'%count
    for file in os.listdir(path):
        if 'DS' not in file:
            ID.append(file)
            Type.append('Type_%s'%count)
    train_label = pd.DataFrame({'id':ID,'label':Type})

train_label.to_csv('train_label.csv',index=False)

"
For label muti col
"
stg1 = pd.read_csv('solution_stg1_release.csv')
stg1['Type_2'] = stg1['Type_2'].apply(lambda x: x*2)
stg1['Type_3'] = stg1['Type_3'].apply(lambda x: x*3)
stg1['label'] = stg1['Type_1']+stg1['Type_2']+stg1['Type_3']

def label_convert(x):
    if x==1:
        return 'Type_1'
    if x==2:
        return 'Type_2'
    if x==3:
        return 'Type_3'

stg1['label'] = stg1['label'].apply(lambda x: label_convert(x) )
stg1.drop(['Type_1','Type_2','Type_3'],axis=1,inplace=True)
stg1.columns = ['id','label']
stg1.to_csv('test_stg1_label.csv',index=False)
