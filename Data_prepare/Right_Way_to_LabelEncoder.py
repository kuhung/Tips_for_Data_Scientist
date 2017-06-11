## Source: https://www.kaggle.com/vrishank97/lightgbm-lb-0-56/code
## Created on: 2017/06/11



for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
