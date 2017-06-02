# Author：kuhung （https://github.com/kuhung）
# Create time：2017/06/02
# Source: https://github.com/kuhung/Student-Grants/blob/master/kuhung/base_line/feature-2-time.py


card_train_test.time = pd.to_datetime(card_train_test.time, format='%Y/%m/%d %H:%M:%S')
