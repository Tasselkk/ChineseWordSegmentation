#conding=utf-8
import re
import torch
START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"B": 0, "M": 1, "E": 2,"S":3, START_TAG: 4, STOP_TAG: 5}

def prepare_sequence(seq, to_ix): #seq是字序列，idxs是字序列对应的向量，to_ix是字和序号的字典
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

#将句子转换为字序列
def get_word(sentence):
    word_list = []
    sentence = ''.join(sentence.split(' '))
    for i in sentence:
        word_list.append(i)
    return word_list

#将句子转换为BMES序列
def get_str(sentence):
    output_str = []
    sentence = re.sub('  ', ' ', sentence) #发现有些句子里面，有两格空格在一起
    list = sentence.split(' ')
    for i in range(len(list)):
        if len(list[i]) == 1:
            output_str.append('S')
        elif len(list[i]) == 2:
            output_str.append('B')
            output_str.append('E')
        else:
            M_num = len(list[i]) - 2
            output_str.append('B')
            output_str.extend('M'* M_num)
            output_str.append('E')
    return output_str

def read_file(filename):
    word, content, label = [], [], []
    text = open(filename, 'r', encoding='utf-8')
    for eachline in text:
        eachline = eachline.strip('\n')
        eachline = eachline.strip(' ')
        word_list = get_word(eachline)
        letter_list = get_str(eachline)
        word.extend(word_list)
        content.append(word_list)
        label.append(letter_list)
    return word, content, label
text,content,label=read_file('word.txt')

#print(text)



