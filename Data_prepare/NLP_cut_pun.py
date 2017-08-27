# Author kuhung(https://github.com/kuhung)
# Create time: 2017/08/27

## 去除中文文本中标点符号的函数
def cut_pun(contnet):
    contnet = contnet.replace(u"\r\n",u",")
    unused_words=u" \t\r\n，。：；“‘”【】『』|=+-——（）*&……%￥#@！~·《》？/?<>.;:'\"[]{}_)(^$!`,"
    for char in unused_words:
        contnet = contnet.replace(char,u'')
    return contnet
