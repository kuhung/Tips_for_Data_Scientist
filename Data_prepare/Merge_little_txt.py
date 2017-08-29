# Author kuhung(https://github.com/kuhung)
# Create time: 2017/06/08

# 更简单的是利用linux的命令行 
# cat file1 file > file3

gist = pd.DataFrame()
for file in os.listdir(gist_path):
    path = os.path.join(gist_path,file)
    test = pd.read_csv(path,header=None)
    test['id'] = file[:32]
    gist=pd.concat([gist, test],ignore_index=True)

gist.to_csv('../gist_sample.csv',index=False,header=None)
