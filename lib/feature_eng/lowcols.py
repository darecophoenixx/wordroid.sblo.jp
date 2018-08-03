import numpy as np


class Seq(object):
    
    def __init__(self, doc_seq, batch_size=32, shaffle=False, state=None):
        self.doc_seq = doc_seq
        self.shaffle = shaffle
        self.state = state
        self.batch_size = batch_size
        
        self.product_list = list(self.doc_seq.word_dic.token2id.keys())
        self.user_list = np.array(list(self.doc_seq.doc_dic.token2id.keys()), dtype=str)

        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1
        
        self.initialize_it()
        
        # others
        self.num_class = max(self.doc_seq.word_dic.keys()) + 1
    
    def initialize_it(self):
        if self.shaffle:
            '''not implemented yet'''
            #random.seed(self.state)
            #random.shuffle(self.user_list)
        
        self.it = iter(range(0, len(self.user_list), self.batch_size))
        self.idx_next = self.it.__next__()
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = self.idx_next
        self.users_part = self.user_list[idx:((idx+self.batch_size) if idx+self.batch_size<len(self.user_list) else len(self.user_list))]
        res = self.getpart(self.users_part)
        try:
            self.idx_next = self.it.__next__()
        except StopIteration:
            self.initialize_it()
        return res
    
    def __getitem__(self, iuser):
        ret_user, ret_y = self.get_data(iuser)
        return ({'input_user': ret_user}, ret_y)
    
    def get_data(self, iuser):
        user_id = self.doc_seq.doc_dic.token2id[iuser]
        prods = self.doc_seq[user_id]
        prods_id = [self.doc_seq.word_dic.token2id[e1] for e1 in prods]
        cats = to_categorical(prods_id, num_classes=self.num_class)
        cat = cats.sum(axis=0)
        
        return (user_id, cat)
    
    def getpart(self, users_part):
        x_input_user = []
        y = []
        for iuser in users_part:
            x_train, y_train = self[iuser]
            x_input_user.append(x_train['input_user'])
            y.append(y_train.tolist())
        return ({
            'input_user': np.array(x_input_user),
            },
            np.array(y))

from keras_ex.GaussianKernel import GaussianKernel, GaussianKernel2, GaussianKernel3

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda, \
    Conv1D, Conv2D, Conv3D, \
    Conv2DTranspose, \
    AveragePooling1D, \
    MaxPooling1D, MaxPooling2D, MaxPooling3D, \
    GlobalAveragePooling1D, \
    GlobalMaxPooling1D, GlobalMaxPooling2D, \
    LocallyConnected1D, LocallyConnected2D, \
    concatenate, Flatten, Average, Activation, \
    RepeatVector, Permute, Reshape, Dot, \
    multiply, dot, add, \
    PReLU, \
    Bidirectional, TimeDistributed, \
    SpatialDropout1D, \
    BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K

def make_model(num_user=20, num_product=39, num_features=12,
                gamma=0.0, embeddings_val=0.5):

    user_embedding = Embedding(output_dim=num_features, input_dim=num_user,
                               embeddings_initializer=initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val),
                               embeddings_regularizer=regularizers.l2(gamma),
                               name='user_embedding', trainable=True)

    input_user = Input(shape=(1,), name='input_user')

    embed_user = Flatten()(user_embedding(input_user))

    model_user = Model(input_user, embed_user)
    
    #init_wgt = initializers.RandomUniform(minval=-embeddings_val, maxval=embeddings_val)((num_product, num_features))
    init_wgt = (np.random.random_sample((num_product, num_features)) - 0.5) * 2 * embeddings_val
    weights1 = [init_wgt, np.log(np.array([1./(2.*num_features*0.1)]))]
    layer_gk1 = GaussianKernel3(num_product, num_features, name='gkernel1', weights=weights1)
    oup = layer_gk1(embed_user)
    model_gk1 = Model(input_user, oup)
    
    
    model = Model(input_user, oup)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    models = {
        'model': model,
        'model_user': model_user,
        'model_gk1': model_gk1,
    }
    return models

