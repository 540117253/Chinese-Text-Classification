import tensorflow as tf
from sklearn import metrics
from tensorflow.python import keras


'''
    =========================================================================
                                Model Definition
    =========================================================================
'''
'''
    Text => Multi_Layer_LSTM => Fully_Connected => Softmax
'''
class Multi_LSTM:
    def __init__(self, rnn_output_dim, num_layers, embedded_size,
                 dict_size, maxlen, label_num, learning_rate):
        
        self.droput_rate = 0.5

        # print('model_Name:', 'Multi_LSTM')
        
        self.X = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
        self.Y = tf.placeholder(tf.int64, [None])
        
        self.encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1), trainable=False)
        encoder_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.X)
        
        rnn_cells = [keras.layers.LSTMCell(units=rnn_output_dim) for _ in range(num_layers)]
        outputs =keras.layers.RNN(rnn_cells,return_sequences=True, return_state=False)(encoder_embedded)
        outputs = tf.nn.dropout(outputs, keep_prob = self.droput_rate)
        
        self.logits = keras.layers.Dense(label_num, use_bias=True)(outputs[:,-1]) # 取出每条文本最后一个单词的隐藏层输出
        self.probability = tf.nn.softmax(self.logits, name='probability')
        
#         self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))
        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                    labels = self.Y, 
                                                                    logits = self.logits)
        self.cost = tf.reduce_mean(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        self.pre_y = tf.argmax(self.logits, 1, name='pre_y')
        correct_pred = tf.equal(self.pre_y, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''
    Text => Bidirectional GRU or LSTM => Fully_Connected => Softmax
'''
class Bi_RNN:
    def __init__(self, rnn_output_dim, embedded_size,
                 dict_size, maxlen, label_num, learning_rate, rnn_type):
        
        self.droput_rate = 0.5

        # print('model_Name:', 'Bi_RNN')

        '''
            Process the Reviews with Bi-RNN
        '''
        def bi_rnn(rnn_type, inputs, rnn_output_dim):
            if rnn_type == 'gru':
                h = keras.layers.Bidirectional(
                              keras.layers.GRU(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs) 

            elif rnn_type == 'lstm' :
                h = keras.layers.Bidirectional(
                              keras.layers.LSTM(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs)
            return h # shape= (None, u_n_words, 2*rnn_output_dim) or # shape(H_d) = (None, i_n_words, 2*rnn_output_dim)
        
        self.X = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
        self.Y = tf.placeholder(tf.int64, [None])
        
        self.encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1), trainable=False)
        encoder_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.X)

        outputs = bi_rnn(rnn_type = rnn_type, inputs = encoder_embedded, rnn_output_dim = rnn_output_dim)
        
        outputs = tf.nn.dropout(outputs, keep_prob = self.droput_rate)
 
        self.logits = keras.layers.Dense(label_num, use_bias=True)(outputs[:,-1]) # 取出每条文本最后一个单词的隐藏层输出
        self.probability = tf.nn.softmax(self.logits, name='probability')

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                    labels = self.Y, 
                                                                    logits = self.logits)
        self.cost = tf.reduce_mean(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        self.pre_y = tf.argmax(self.logits, 1, name='pre_y')
        correct_pred = tf.equal(self.pre_y, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


'''
    Text => Bidirectional GRU or LSTM => Word_Attention => Fully_Connected => Softmax
'''
class Bi_RNN_Attention:
    def __init__(self, rnn_output_dim, embedded_size,
                 dict_size, maxlen, label_num, learning_rate, attention_size, rnn_type):
        
        self.droput_rate = 0.5

        # print('model_Name:', 'Bi_RNN_Attention')

        '''
            Process the Reviews with Bi-RNN
        '''
        def bi_rnn(rnn_type, inputs, rnn_output_dim):
            if rnn_type == 'gru':
                h = keras.layers.Bidirectional(
                              keras.layers.GRU(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs) 

            elif rnn_type == 'lstm' :
                h = keras.layers.Bidirectional(
                              keras.layers.LSTM(rnn_output_dim,return_sequences=True,unroll=True),
                              merge_mode='concat'
                       )(inputs)
            return h # shape= (None, u_n_words, 2*rnn_output_dim) or # shape(H_d) = (None, i_n_words, 2*rnn_output_dim)
        
        self.X = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
        self.Y = tf.placeholder(tf.int64, [None])
        
        self.encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1), trainable=False)
        encoder_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.X)

        outputs = bi_rnn(rnn_type = rnn_type, inputs = encoder_embedded, rnn_output_dim = rnn_output_dim) # shape = [None, maxlen, rnn_output_dim] 
        
        outputs = tf.nn.dropout(outputs, keep_prob = self.droput_rate)

        '''
            Word Attention Layer
        '''
        attention_w = tf.get_variable("attention_v", [attention_size], tf.float32)
        query = keras.layers.Dense(attention_size)(tf.expand_dims(outputs[:,-1], 1)) # shape =[None, 1, attention_size]
        keys = keras.layers.Dense(attention_size)(outputs) # shape = [None, maxlen, attention_size]
        self.attention = tf.reduce_sum(attention_w * tf.tanh(keys + query), 2) # shape = [None, maxlen]
        self.attention = tf.nn.softmax(self.attention, name='attention')
        outputs = tf.squeeze(
                                tf.matmul(
                                    tf.transpose(outputs, [0, 2, 1]),tf.expand_dims(self.attention, 2)
                                ), # shape = [None, rnn_output_dim, 1]
                            2) # shape = [None, rnn_output_dim]

        self.logits = keras.layers.Dense(label_num, use_bias=True)(outputs)
        self.probability = tf.nn.softmax(self.logits, name='probability')

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                    labels = self.Y, 
                                                                    logits = self.logits)
        self.cost = tf.reduce_mean(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        self.pre_y = tf.argmax(self.logits, 1, name='pre_y')
        correct_pred = tf.equal(self.pre_y, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




'''
    Text => CNN => Fully_Connected => Softmax
'''
class CNN:
    def __init__(self, filter_sizes, num_filters, embedded_size,
                 dict_size, maxlen, label_num, learning_rate): 

        # print('model_Name:', 'CNN')

        self.droput_rate = 0.5
        
        '''
            Convulutional Neural Network
        '''
        def cnn (input_emb, filter_sizes, num_filters):
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                filter_shape = [filter_size, embedded_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")         
                conv = tf.nn.conv2d(
                    input_emb,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv") # shape(conv) = [None, sequence_length - filter_size + 1, 1, num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                word_num = input_emb.shape.as_list()[1]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, word_num - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool") # shape(pooled) = [None, 1, 1, num_filters]
                pooled_outputs.append(pooled)
            num_filters_total = num_filters * len(filter_sizes)

            h_pool = tf.concat(pooled_outputs,3) 

            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # shape = [None,num_filters_total] 

            cnn_fea = tf.nn.dropout(h_pool_flat, keep_prob =  self.droput_rate)

            return cnn_fea
        
        self.X = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
        self.Y = tf.placeholder(tf.int64, [None])
        
        self.encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1), trainable=False)
        encoder_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.X)
        
        # 由于conv2d需要一个四维的输入数据，因此需要手动添加一个维度。
        encoder_embedded = tf.expand_dims(encoder_embedded, -1) # shape(encoder_embedded) = [None, user_review_num*u_n_words, embedding_size, 1]

        outputs = cnn(input_emb = encoder_embedded, filter_sizes = filter_sizes, num_filters = num_filters)
 
        self.logits = keras.layers.Dense(label_num, use_bias=True)(outputs)
        self.probability = tf.nn.softmax(self.logits, name='probability')

        self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                                    labels = self.Y, 
                                                                    logits = self.logits)
        self.cost = tf.reduce_mean(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        self.pre_y = tf.argmax(self.logits, 1, name='pre_y')
        correct_pred = tf.equal(self.pre_y, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
