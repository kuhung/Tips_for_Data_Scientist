# Author：kuhung （https://github.com/kuhung）
# Create time：2017/06/05
# According to：https://stackoverflow.com/questions/15755057/using-cumsum-in-pandas-on-group

data1=pd.DataFrame({'Dir':['E','E','W','W','E','W','W','E'],
                    'Bool':['Y','N','Y','N','Y','N','Y','N'], 
                    'Data':[4,5,6,7,8,9,10,11]}, 
                   index=pd.DatetimeIndex(['12/30/2000','12/30/2000','12/30/2000','1/2/2001','1/3/2001','1/3/2001','12/30/2000','12/30/2000']))

# Have this day
data1.groupby(['Bool', 'Dir']).apply(lambda x: x['Data'].cumsum())

# Not this day                   
data1.groupby(['Bool', 'Dir']).apply(lambda x: x['Data'].cumsum()-x['Data'])

------------------
Apply：

# Author: kuhung (https://github.com/kuhung)
# Created on: 2017/06/05



def count_cumsum(df,label_groupby,label_taregt = 'label',flag = True):
    
    '''
    df : DataFrame
    label_groupby: The feature you concern.
    label_target: Default is 'label'
    
    if flag == True:
        then  do count
    else:
        do sum    
    # key code
    count = count.groupby(['%s'%label_groupby]).apply(lambda x : x['label'].cumsum()-x['label'])
    '''
    
    if flag == True:
        count = df.groupby(['%s'%label_groupby,'clickDate'],as_index = False)['label'].count()
    else:
        count = df.groupby(['%s'%label_groupby,'clickDate'],as_index = False)['label'].sum()
        
    count = count.set_index('clickDate')
    
    # key code
    count = count.groupby(['%s'%label_groupby]).apply(lambda x : x['label'].cumsum()-x['label'])
    
    count.to_csv('../temp/count.csv')
    count = pd.read_csv('../temp/count.csv',header=None)
    
    if flag == True:
        count.columns = ['%s'%label_groupby,'clickDate','%s_count_c'%label_groupby]
    else:
        count.columns = ['%s'%label_groupby,'clickDate','%s_sum_c'%label_groupby]
        
    return count
