






gist = pd.DataFrame()
for file in os.listdir(gist_path):
    path = os.path.join(gist_path,file)
    test = pd.read_csv(path,header=None)
    test['id'] = file[:32]
    gist=pd.concat([gist, test],ignore_index=True)

gist.to_csv('../gist_sample.csv',index=False,header=None)
