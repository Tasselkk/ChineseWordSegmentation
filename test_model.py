from CWS.trainning import word_to_ix
from CWS.data_process import prepare_sequence
from CWS.BiLSTM_CRF import BiLSTM_CRF
import torch
import numpy

net=torch.load('cws.model')
net.eval()
stri="改善人民生活水平，建设社会主义政治经济。"
precheck_sent = prepare_sequence(stri, word_to_ix)
label=net(precheck_sent)[1]
#print(net(precheck_sent))
cws=[]




#print(label)
for i in range(len(label)):
    cws.extend(stri[i])
    if label[i]==2 or label==3:
        cws.append('/')
#print(cws)
str=''
for i in cws:
    str=str+i
print('==========Chinese Word Segmentation=========\n')
print('输入未分词语句：\n')
print(stri+'\n')
print('分词结果：\n')
print(str+'\n')
print('====================Done!===================\n')
        
    
