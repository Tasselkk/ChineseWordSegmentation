from CWS.data_process import read_file,tag_to_ix
from CWS.config import *
from CWS.BiLSTM_CRF import *
import torch
from torch import nn
from torch import optim

_,content,label=read_file(filename)

def train_data(content,label):
    train_data=[]
    for i in range(len(label)):
        train_data.append((content[i],label[i]))
    return train_data
data=train_data(content,label)

word_to_ix = {}
for sentence, tags in data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
#optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
optimizer=optim.Adam(model.parameters(),lr=1e-3)
#训练
'''
for epoch in range(epochs):
    for sentence, tags in data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)

        loss.backward()
        optimizer.step()
    if epoch%10==0:
        print('epoch/epochs:{}/{},loss:{:.6f}'.format(epoch+1,epochs,loss.data[0]))

#保存
torch.save(model,'cws.model')
torch.save(model.state_dict(),'cws_all.model')
'''