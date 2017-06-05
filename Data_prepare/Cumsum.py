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


