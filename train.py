import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from sklearn import metrics
from Models import CNN, Multi_LSTM, Bi_RNN, Bi_RNN_Attention




os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 屏蔽TensorFlow的通知信息和警告信息

# 预训练词向量路径
Pretrained_WordVectors_Name = 'sgns.baidubaike.bigram-char'
Pretrained_WordVectors_Name_Path = './data/' + Pretrained_WordVectors_Name


# 选择模型
# model_Name = 'multi_lstm'
# model_Name = 'bi_rnn'
# model_Name = 'cnn'
model_Name = 'bi_rnn_attention'



# 各个模型的共享参数
pretrained_wordVector = False # 是否使用预训练词向量
label_num = 15 # 该数据集中的类别数量
learning_rate = 1e-3 # 优化器的初始化学习率
batch_size = 1024
droput_rate = 0.5 # the rate of dropout for neurons
embedded_size = 300 # 词向量维度


# 基于RNN的模型参数
rnn_output_dim = 50 # the output dimensions of the rnn
num_layers = 3 # the number of rnn layers in model 'Multi_LSTM'
attention_size = 25


# 基于CNN的模型参数
filter_sizes = list(map(np.int32, "3".split(","))) # the size of each kernel in cnn
num_filters = 100 # the number of kernels in cnn



'''
   def create_embedding_matrix(vocabulary, dim): 
   
   功能：
       根据指定的维度dim及词汇表（词汇表是一个dictionary，key=单词，values=单词的编号），来构建词向量矩阵。
       如果存在某个词的预训练词向量，则该词的词向量初始化为预训练的词向量。否则，该词的词向量随机初始化。
   
   变量：
       embeddings_index：是一个dictionary，key=单词，values=单词对应的向量
       embedding_matrix的格式：例如编号为 i 的单词对应的词向量为embedding_matrix的第 i 行向量。
'''
def create_embedding_matrix(dictionary, dim):
    
    print("正在构建{dim}维词向量矩阵.....".format(dim=dim))
    
    # 根据glove文件，构建词向量索引。embeddings_index是一个dictionary，key=单词，values=单词对应的向量
    embeddings_index = {}
    with open(Pretrained_WordVectors_Name_Path, encoding="utf-8") as f:
        next(f) # read from the second line
        for line in f:
            try:
                values = line.split()
                word = values[0]  # 单词
                coefs = np.asarray(values[1:], dtype='float32')  # 单词对应的向量
                embeddings_index[word] = coefs  # 单词及对应的向量
            except:
                a=1

    # 初始化词向量矩阵embedding_matrix
    embedding_matrix = np.random.uniform(-1.0, 1.0, (len(dictionary)+1, dim)) # dictionary 序列化过的单词下标最小值为1，因此词向量矩阵的长度要+1
    
    # 根据dictionary和embeddings_index，对embedding_matrix赋值
    for word, i in dictionary.items(): # i 是从1开始，因此embedding_matrix第0行为零向量
        embedding_vector = embeddings_index.get(word)  # 根据词向量字典获取该单词对应的词向量
        if embedding_vector is not None: # 如果glove有该单词的词向量，则写入词向量矩阵中
            embedding_matrix[i] = embedding_vector
    
    print("完成构建{dim}维词向量矩阵!".format(dim=dim))

    return embedding_matrix



'''
    划分数据集为trainSet, validSet, testSet
'''
with open("data/token_data", 'rb') as f:
    token_data = pkl.load(f)
shuffle_token_data = token_data.loc[np.random.permutation(np.arange(len(token_data)))]
train_size = int(len(token_data)*0.8)
valid_size = int(len(token_data)*0.1)
test_size = int(len(token_data)*0.1)
train = shuffle_token_data[0:train_size]
valid = shuffle_token_data[train_size:(train_size+valid_size)]
test = shuffle_token_data[-test_size:]

'''
    读入数据集的字典
'''
dictionary = dict()
f = open('data/dictionary.txt', 'r', encoding='UTF-8')
lines = f.readlines()
for line in lines:
    line = line.strip()
    temp = line.split("\t")
    dictionary[temp[0]] = int(temp[1])
f.close()

'''
    初始化模型
'''
tf.reset_default_graph()
sess = tf.InteractiveSession()

maxlen = len(test.iloc[0]['content']) # 每个样本的序列化后所保留的单词数

if model_Name == 'multi_lstm':
    model = Multi_LSTM( rnn_output_dim, num_layers, embedded_size, len(dictionary)+1, maxlen,label_num, learning_rate)
elif model_Name == 'bi_rnn':
    model = Bi_RNN( rnn_output_dim, embedded_size, len(dictionary)+1, maxlen, label_num, learning_rate, rnn_type='gru')
elif model_Name == 'bi_rnn_attention':
    model = Bi_RNN_Attention( rnn_output_dim, embedded_size, len(dictionary)+1, maxlen, label_num, learning_rate, attention_size, rnn_type='gru')
elif model_Name == 'cnn':
    model = CNN( filter_sizes, num_filters, embedded_size, len(dictionary)+1, maxlen, label_num, learning_rate )

sess.run(tf.global_variables_initializer())

print('model_Name: {model_Name}'.format(model_Name=model_Name))



if pretrained_wordVector == True:
    embedding_matrix = create_embedding_matrix(dictionary=dictionary, dim=300) 
    sess.run(model.encoder_embeddings.assign(embedding_matrix))


'''
    train the model and early stop
'''
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0


while True:
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n'%(EPOCH))
        break
        
    train_acc, train_loss, valid_acc, val_loss = 0, 0, 0, 0

    model.droput_rate = droput_rate
    for i in range(0, (len(train) // batch_size) * batch_size, batch_size):
        batch_x = np.array(train.iloc[i:i+batch_size]['content'].tolist())
        acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer], 
                           feed_dict = {model.X : batch_x, 
                                        model.Y : train.iloc[i:i+batch_size]['label'].values})
        train_acc += acc
    
    model.droput_rate = 0.0
    for i in range(0, (len(valid) // batch_size) * batch_size, batch_size):
        batch_x = np.array(valid.iloc[i:i+batch_size]['content'].tolist())
        acc, loss = sess.run([model.accuracy, model.cost], 
                           feed_dict = {model.X : batch_x, 
                                        model.Y : valid.iloc[i:i+batch_size]['label'].values})
        valid_acc += acc
    

    train_acc /= (len(train) // batch_size)

    valid_acc /= (len(valid) // batch_size)
    
    if valid_acc > CURRENT_ACC:
        # print('epoch: %d, pass acc: %f, current acc: %f'%(EPOCH,CURRENT_ACC, valid_acc))
        CURRENT_ACC = valid_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1
        
    print('epoch: %d, training acc: %f, valid acc: %f\n' % ( EPOCH, train_acc, valid_acc))
    EPOCH += 1

print('best_valid acc:', CURRENT_ACC)

'''
    after the early stop, finally use the testDataSet to test the model
'''
test_acc = 0
model.droput_rate = 0.0
for i in range(0, (len(test) // batch_size) * batch_size, batch_size):
    batch_x = np.array(test.iloc[i:i+batch_size]['content'].tolist())
    acc, loss = sess.run([model.accuracy, model.cost], 
                       feed_dict = {model.X : batch_x, 
                                    model.Y : test.iloc[i:i+batch_size]['label'].values})
    test_acc += acc
    
test_acc /= (len(test) // batch_size)

print('test acc: %f '%( test_acc))


'''
    show the evaluation metric of the model in testDataSet
'''
logits = sess.run(model.logits, feed_dict={model.X: np.array(test.loc[:,'content'].tolist())})
print('model_Name:',model_Name)
print('evaluation metric in testDataSet')
print(metrics.classification_report(test.loc[:,'label'].values, np.argmax(logits,1), target_names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']))




