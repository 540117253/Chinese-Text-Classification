import jieba as jb
import re
import pandas as pd
import collections
import numpy as np
import os
import pickle as pkl



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 屏蔽TensorFlow的通知信息和警告信息



# DataSet Directory
DataSet_Name = 'toutiao_cat_data.txt'
Input_DataSet_File = './data/'+DataSet_Name 


maxlen = 20 # 每条样本所保留的最大单词数

'''
    清除无用字符，并且切割词语
'''
def clean_cut(s):
    fil = re.compile(u'[^a-zA-Z\d\u4e00-\u9fa5]+', re.UNICODE)
    s = fil.sub('', s)
    s = ' '.join(jb.cut(s))
    return s


'''
    建立整个数据集的字典
'''
def build_vocabulary(words, n_words):
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


'''
    对进行截断或补0处理，确保每条样本的长度为maxlen
'''
def str_idx(corpus, dic, maxlen, UNK = 3):
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            X[i, -1 - no] = dic.get(k, UNK)
    return X



dataSet = pd.read_csv(Input_DataSet_File,encoding='utf-8', sep='_!_',header=None, names=['id','code','code_name','content','keyword'])

dataSet['content'] = dataSet['content'].astype('str').apply(clean_cut) #[' '.join(jb.cut(i)) for i in dataSet.loc[:,'content'].astype('str')]



'''
    tokenize label
'''
code_set = list(set(dataSet['code']))
code2label = dict()
for i in range(len(code_set)):
    code2label[code_set[i]]=i
def tokenize_label(code):
    return code2label[code]

dataSet['label'] = dataSet['code'].apply(tokenize_label)


'''
    construct the vocabulary of the whole dataSet
'''
concat = ' '.join(dataSet.loc[:,'content']).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_vocabulary(concat, vocabulary_size)

# 保存字典dictionary
with open('data/dictionary.txt', 'w', encoding='utf-8') as f:
    for k in dictionary.keys():
        f.write(str(k) + '\t' + str(dictionary.get(k)) + '\n')


'''
    序列化每条样本，并返回dataFrame格式
'''
processed_data = {
    "content" : list(map(lambda x : list(x), str_idx(dataSet.loc[:,'content'], dictionary,maxlen))),
    "label": dataSet['label'],
    "id": dataSet['id']
   }
token_data = pd.DataFrame(processed_data)#将字典转换成为dataFrame

cols=['id','content','label']
token_data=token_data.loc[:,cols]


f = open("data/token_data", 'wb')
pkl.dump(token_data, f)
f.close()

print('data process completed')