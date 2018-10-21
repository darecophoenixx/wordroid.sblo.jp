'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/HumanisticML/LICENSE.md
'''

import numpy as np
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

from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3


def _make_base_model0(inp_img, inp_cls,
                img_dim, cls_dim, num_lm,
                model_ae=None,
                wgt_embed=None, init_lm=None, init_lm2=None,
                r_state=None, name=None,
                fix_gk1=False, fix_embed=False, fix_ae=False):
    np.random.seed(r_state)
    '''==============================
    inputs
    =============================='''
    inp_200 = Input(shape=(num_lm,), name='input_200') # internal use
    
    '''==============================
    layers
    =============================='''
    if wgt_embed is None:
        wgt_embed = np.random.random_sample((cls_dim, img_dim))
    layer_dense_embed = Dense(img_dim, use_bias=False, name='embed', weights=[wgt_embed],
                              trainable=not fix_embed)
    
    '''Gaussian Kernel 1'''
    if init_lm is None:
        init_lm = np.random.random_sample((num_lm, img_dim))
    if fix_gk1:
        weights1 = [np.log(np.array([1./(2.*img_dim*0.1)]))]
        layer_gk1 = GaussianKernel2(init_lm, name='gkernel1_fix', weights=weights1)
    else:
        weights1 = [init_lm, np.log(np.array([1./(2.*img_dim*0.1)]))]
        layer_gk1 = GaussianKernel3(num_lm, img_dim, name='gkernel1', weights=weights1)
    
    '''Gaussian Kernel 2'''
    if init_lm2 is None:
        init_lm2 = np.random.random_sample((cls_dim, num_lm))
    weights2 = [init_lm2, np.log(np.array([1./(2.*num_lm*0.1)]))]
    layer_gk2 = GaussianKernel3(cls_dim, num_lm, weights=weights2, name='gkernel2')
    
    '''==============================
    models
    =============================='''
    oup_gk1 = layer_gk1(inp_img)
    model_gk1 = Model(inp_img, oup_gk1, name='model_gk1' if name is None else 'model{}_gk1'.format(name))
    
    oup_gk2 = layer_gk2(inp_200)
    model_classify = Model(inp_200, oup_gk2, name='model_classify' if name is None else 'model{}_classify'.format(name))
    
    model_embed = Model(inp_cls, layer_dense_embed(inp_cls), name='model_embed' if name is None else 'model{}_embed'.format(name))
    
    # ae
    if model_ae is None:
        oup_ae = Dense(img_dim, use_bias=True, name='ae', activation='sigmoid', trainable=not fix_ae)(inp_200)
        model_ae = Model(inp_200, oup_ae, name='model_ae' if name is None else 'model{}_ae'.format(name))
    
    # img > img
    oup_img_img = model_ae(oup_gk1)
    oup_img_img = Activation('linear', name='output_img')(oup_img_img)
    model_img_img = Model(inp_img, oup_img_img, name='model_img_img' if name is None else 'model{}_img_img'.format(name))
    
    # img > cls
    oup_img_cls = model_classify(oup_gk1)
    model_img_cls = Model(inp_img, oup_img_cls, name='model_img_cls' if name is None else 'model{}_img_cls'.format(name))
    
    # cls > img
    model_cls_img = Model(inp_cls, model_img_img(model_embed(inp_cls)), name='model_cls_img' if name is None else 'model{}_cls_img'.format(name))
    
    # cls > cls
    model_cls_cls = Model(inp_cls, model_img_cls(model_embed(inp_cls)), name='model_cls_cls' if name is None else 'model{}_cls_cls'.format(name))
    
    model_cls_img_cls = Model(inp_cls, model_img_cls(model_cls_img(inp_cls)), name='model_cls_img_cls' if name is None else 'model{}_cls_img_cls'.format(name))
    model_img_cls_img = Model(inp_img, model_cls_img(model_img_cls(inp_img)), name='model_img_cls_img' if name is None else 'model{}_img_cls_img'.format(name))
    
    
    
    return {
        'model_embed': model_embed,
        'model_ae': model_ae,
        'model_gk1': model_gk1,
        'model_classify': model_classify,
        
        'model_img_img': model_img_img,
        'model_img_cls': model_img_cls,
        'model_cls_img': model_cls_img,
        'model_cls_cls': model_cls_cls,
        
        'model_cls_img_cls': model_cls_img_cls,
        'model_img_cls_img': model_img_cls_img,
    }


def _make_model0(inp_img, inp_cls,
                img_dim, cls_dim, num_lm,
                model_ae=None,
                wgt_embed=None, init_lm=None, init_lm2=None,
                flag_img_cls_img=False, flag_cls_img_cls_x2=False, flag_cls_img_cls_img=False,
                flag_cls_cls=False,
                r_state=None, name=None,
                fix_gk1=False):
    '''==============================
    inputs
    =============================='''
    base_models = _make_base_model0(
        inp_img=inp_img, inp_cls=inp_cls,
        img_dim=img_dim, cls_dim=cls_dim, num_lm=num_lm,
        model_ae=model_ae,
        wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
        r_state=r_state, name=name, fix_gk1=fix_gk1)
    model_embed = base_models['model_embed']
    model_ae = base_models['model_ae']
    model_gk1 = base_models['model_gk1']
    model_classify = base_models['model_classify']
    
    model_img_img = base_models['model_img_img']
    model_img_cls = base_models['model_img_cls']
    model_cls_cls = base_models['model_cls_cls']
    model_cls_img = base_models['model_cls_img']
    
    model_cls_img_cls = base_models['model_cls_img_cls']
    model_img_cls_img = base_models['model_img_cls_img']
    
    
    '''==============================
    cost functions
    =============================='''
    def cost_cls(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred)
    def cost_ae(y_true, y_pred):
        return losses.mse(y_true, y_pred)
    def cost_zero(y_true, y_pred):
        return losses.mse(0, y_pred)
    
    
    
    
    
    '''=== img -> imgA ==='''
    path_img_imgA = model_img_img(inp_img)
    path_img_imgA = Activation('linear', name='path_img_imgA' if name is None else 'path{}_img_imgA'.format(name))(path_img_imgA)
    
    '''=== img -> cls ==='''
    path_img_cls = model_img_cls(inp_img)
    path_img_cls = Activation('linear', name='path_img_cls' if name is None else 'path{}_img_cls'.format(name))(path_img_cls)
    
    '''=== cls -> img ==='''
    path_cls_img = model_cls_img(inp_cls)
    path_cls_img = Activation('linear', name='path_cls_img' if name is None else 'path{}_cls_img'.format(name))(path_cls_img)
    
    '''=== cls -> cls ==='''
    path_cls_cls = model_cls_cls(inp_cls)
    path_cls_cls = Activation('linear', name='path_cls_cls' if name is None else 'path{}_cls_cls'.format(name))(path_cls_cls)
    
    '''=== cls -> img -> cls ==='''
    path_cls_img_cls = model_cls_img_cls(inp_cls)
    path_cls_img_cls = Activation('linear', name='path_cls_img_cls' if name is None else 'path{}_cls_img_cls'.format(name))(path_cls_img_cls)
    
    '''=== cls > img > cls | cls > img > cls ==='''
    path_cls_img_cls_x2 = model_cls_img_cls(model_cls_img_cls(inp_cls))
    path_cls_img_cls_x2 = Activation('linear', name='path_cls_img_cls_x2' if name is None else 'path{}_cls_img_cls_x2'.format(name))(path_cls_img_cls_x2)
    
    '''=== cls > img > cls | cls > img ==='''
    path_cls_img_cls_img = model_cls_img(model_cls_img_cls(inp_cls))
    path_cls_img_cls_img = Activation('linear', name='path_cls_img_cls_img' if name is None else 'path{}_cls_img_cls_img'.format(name))(path_cls_img_cls_img)
    
    '''=== img -> cls -> img ==='''
    path_img_cls_img = model_img_cls_img(inp_img)
    path_img_cls_img = Activation('linear', name='path_img_cls_img' if name is None else 'path{}_img_cls_img'.format(name))(path_img_cls_img)
    
    
    
    
    path_fit_imgA = Lambda(lambda x: x[0] - x[1], name='path_fit_imgA' if name is None else 'path{}_fit_imgA'.format(name))([
        path_img_imgA,
        path_cls_img
    ])
    
    '''no use yet'''
    path_fit_imgclsimg = Lambda(lambda x: x[0] - x[1], name='path_fit_imgclsimg' if name is None else 'path{}_fit_imgclsimg'.format(name))([
        path_img_cls_img,
        path_cls_img
    ])
    
    
    
    
    '''=== concat ==='''
    model_output_list = [path_cls_img, path_cls_img_cls, path_fit_imgA]
    loss_dic = {
        'path_cls_img': cost_ae,
        'path_cls_img_cls': cost_cls,
        'path_fit_imgA': cost_zero,
    }
    if flag_img_cls_img:
        model_output_list.append(path_img_cls_img)
        loss_dic['path_img_cls_img'] = cost_ae
    if flag_cls_img_cls_x2:
        model_output_list.append(path_cls_img_cls_x2)
        loss_dic['path_cls_img_cls_x2'] = cost_cls
    if flag_cls_img_cls_img:
        model_output_list.append(path_cls_img_cls_img)
        loss_dic['path_cls_img_cls_img'] = cost_ae
    if flag_cls_cls:
        model_output_list.append(path_cls_cls)
        loss_dic['path_cls_cls'] = cost_cls
    
    model = Model([inp_img, inp_cls],
                  model_output_list)
    if name is None:
        model.compile(loss=loss_dic,
                      metrics=['accuracy'],
                      optimizer='adam')
    
    return {
        'model': model,
        'model_embed': model_embed,
        'model_ae': model_ae,
        'model_gk1': model_gk1,
        'model_classify': model_classify,
        
        'model_img_img': model_img_img,
        'model_img_cls': model_img_cls,
        'model_cls_img': model_cls_img,
        'model_cls_cls': model_cls_cls,
        
        'model_cls_img_cls': model_cls_img_cls,
        'model_img_cls_img': model_img_cls_img,
        
        'path': {
            'path_img_imgA': path_img_imgA,
            'path_img_cls': path_img_cls,
            'path_cls_img': path_cls_img,
            'path_cls_cls': path_cls_cls,
            'path_cls_img_cls': path_cls_img_cls,
            'path_cls_img_cls_x2': path_cls_img_cls_x2,
            'path_cls_img_cls_img': path_cls_img_cls_img,
            'path_img_cls_img': path_img_cls_img,
            'path_fit_imgA': path_fit_imgA,
            'path_fit_imgclsimg': path_fit_imgclsimg,
        }
    }


'''
you can input 'model_convert'
'''
def _make_model1(inp, inp_cls,
                img_dim, cls_dim, num_lm,
                model_convert=None,
                model_ae=None,
                wgt_embed=None, init_lm=None, init_lm2=None,
                flag_img_cls_img=False, flag_cls_img_cls_x2=False, flag_cls_img_cls_img=False,
                flag_cls_cls=False, flag_cls_img_imgE=False,
                loss_wgt={'path_fit_imgA': 1.0},
                r_state=None, no_path_fit_cls_img=False,
                fix_gk1=False, fix_embed=False, fix_ae=False):
    '''==============================
    inputs
    =============================='''
    inp_img = Input(shape=(img_dim,), name='input_img') # internal use
    
    base_models = _make_base_model0(
        inp_img=inp_img, inp_cls=inp_cls,
        img_dim=img_dim, cls_dim=cls_dim, num_lm=num_lm,
        model_ae=model_ae,
        wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
        r_state=r_state,
        fix_gk1=fix_gk1, fix_embed=fix_embed, fix_ae=fix_ae)
    model_embed = base_models['model_embed']
    model_ae = base_models['model_ae']
    model_gk1 = base_models['model_gk1']
    model_classify = base_models['model_classify']
    
    model_img_img = base_models['model_img_img']
    model_img_cls = base_models['model_img_cls']
    model_cls_cls = base_models['model_cls_cls']
    model_cls_img = base_models['model_cls_img']
    
    model_cls_img_cls = base_models['model_cls_img_cls']
    model_img_cls_img = base_models['model_img_cls_img']
    
    
    np.random.seed(r_state)
    
    if model_convert is None:
        oup_img = inp
    else:
        oup_img = model_convert(inp)
    
    '''==============================
    cost functions
    =============================='''
    def cost_cls(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred)
    def cost_ae(y_true, y_pred):
        return losses.mse(y_true, y_pred)
    def cost_zero(y_true, y_pred):
        return losses.mse(0, y_pred)
    
    
    
    path_imgE = model_embed(inp_cls)
    
    '''=== img -> imgA ==='''
    path_img_imgA = model_img_img(oup_img)
    path_img_imgA = Activation('linear', name='path_img_imgA')(path_img_imgA)
    
    '''=== img -> cls ==='''
    path_img_cls = model_img_cls(oup_img)
    path_img_cls = Activation('linear', name='path_img_cls')(path_img_cls)
    
    '''=== cls -> img ==='''
    path_cls_img = model_cls_img(inp_cls)
    path_cls_img = Activation('linear', name='path_cls_img')(path_cls_img)
    
    '''=== cls -> cls ==='''
    path_cls_cls = model_cls_cls(inp_cls)
    path_cls_cls = Activation('linear', name='path_cls_cls')(path_cls_cls)
    
    '''=== cls -> img -> cls ==='''
    path_cls_img_cls = model_cls_img_cls(inp_cls)
    path_cls_img_cls = Activation('linear', name='path_cls_img_cls')(path_cls_img_cls)
    
    '''=== cls > img > cls | cls > img > cls ==='''
    path_cls_img_cls_x2 = model_cls_img_cls(model_cls_img_cls(inp_cls))
    path_cls_img_cls_x2 = Activation('linear', name='path_cls_img_cls_x2')(path_cls_img_cls_x2)
    
    '''=== cls > img > cls | cls > img ==='''
    path_cls_img_cls_img = model_cls_img(model_cls_img_cls(inp_cls))
    path_cls_img_cls_img = Activation('linear', name='path_cls_img_cls_img')(path_cls_img_cls_img)
    
    '''=== img -> cls -> img ==='''
    path_img_cls_img = model_img_cls_img(oup_img)
    path_img_cls_img = Activation('linear', name='path_img_cls_img')(path_img_cls_img)
    
    path_img_img_cls = model_img_cls(model_img_img(oup_img))
    path_img_img_cls = Activation('linear', name='path_img_img_cls')(path_img_img_cls)
    
    path_fit_imgA = Lambda(lambda x: x[0] - x[1], name='path_fit_imgA')([
        path_img_imgA,
        path_cls_img
    ])
    path_fit_cls_img = Lambda(lambda x: x[0] - x[1], name='path_fit_cls_img')([
        path_cls_img,
        oup_img
    ])
    path_fit_cls_img_imgE = Lambda(lambda x: x[0] - x[1], name='path_fit_cls_img_imgE')([
        path_cls_img,
        path_imgE
    ])
#    path_fit_img_cls_img = Lambda(lambda x: x[0] - x[1], name='path_fit_img_cls_img')([
#        path_img_cls_img,
#        oup_img
#    ])
#    path_fit_cls_img_cls_img = Lambda(lambda x: x[0] - x[1], name='path_fit_cls_img_cls_img')([
#        path_cls_img_cls_img,
#        oup_img
#    ])
    path_fit_img_cls_img = Lambda(lambda x: x[0] - x[1], name='path_fit_img_cls_img')([
        path_img_cls_img,
        path_cls_img
    ])
    path_fit_cls_img_cls_img = Lambda(lambda x: x[0] - x[1], name='path_fit_cls_img_cls_img')([
        path_cls_img_cls_img,
        path_cls_img
    ])
    
    
    
    
    '''=== concat ==='''
    if no_path_fit_cls_img:
        model_output_list = [path_cls_img_cls, path_fit_imgA]
        loss_dic = {
            'path_cls_img_cls': cost_cls,
            'path_fit_imgA': cost_zero,
        }
    else:
        model_output_list = [path_fit_cls_img, path_cls_img_cls, path_fit_imgA]
        loss_dic = {
            'path_fit_cls_img': cost_zero,
            'path_cls_img_cls': cost_cls,
            'path_fit_imgA': cost_zero,
        }
    if flag_img_cls_img:
        model_output_list.append(path_fit_img_cls_img)
        loss_dic['path_fit_img_cls_img'] = cost_zero
    if flag_cls_img_cls_x2:
        model_output_list.append(path_cls_img_cls_x2)
        loss_dic['path_cls_img_cls_x2'] = cost_cls
    if flag_cls_img_cls_img:
        model_output_list.append(path_fit_cls_img_cls_img)
        loss_dic['path_fit_cls_img_cls_img'] = cost_zero
    if flag_cls_cls:
        model_output_list.append(path_cls_cls)
        loss_dic['path_cls_cls'] = cost_cls
    if flag_cls_img_imgE:
        model_output_list.append(path_fit_cls_img_imgE)
        loss_dic['path_fit_cls_img_imgE'] = cost_zero
    
    model = Model([inp, inp_cls],
                  model_output_list)
    model.compile(loss=loss_dic,
                  loss_weights=loss_wgt,
                  metrics=['categorical_accuracy', 'binary_accuracy'],
                  optimizer='adam')
    
    ### model_pre
    model_output_list_pre = list(model_output_list)
    model_output_list_pre.append(path_img_cls)
    loss_dic_pre = dict(loss_dic)
    loss_dic_pre['path_img_cls'] = cost_cls
    model_pre = Model([inp, inp_cls],
                      model_output_list_pre)
    model_pre.compile(loss=loss_dic_pre,
                      loss_weights=loss_wgt,
                      metrics=['categorical_accuracy', 'binary_accuracy'],
                      optimizer='adam')
    
    ### model_pre01
    model_output_list_pre01 = list(model_output_list)
    model_output_list_pre01.append(path_img_cls)
    model_output_list_pre01.append(path_img_img_cls)
    loss_dic_pre01 = dict(loss_dic)
    loss_dic_pre01['path_img_cls'] = cost_cls
    loss_dic_pre01['path_img_img_cls'] = cost_cls
    model_pre01 = Model([inp, inp_cls],
                      model_output_list_pre01)
    model_pre01.compile(loss=loss_dic_pre01,
                      loss_weights=loss_wgt,
                      metrics=['categorical_accuracy', 'binary_accuracy'],
                      optimizer='adam')
    
    model_0 = Model(inp, path_img_cls)
    model_0.compile(loss='binary_crossentropy',
                    metrics=['categorical_accuracy', 'binary_accuracy'],
                    optimizer='adam')
    
    model_1 = Model(inp, path_img_img_cls)
    model_1.compile(loss='binary_crossentropy',
                    metrics=['categorical_accuracy', 'binary_accuracy'],
                    optimizer='adam')
    
    model_01 = Model(inp, [path_img_cls, path_img_img_cls])
    model_01.compile(loss='binary_crossentropy',
                    metrics=['categorical_accuracy', 'binary_accuracy'],
                    optimizer='adam')
    
    return {
        'model': model,
        'model_pre': model_pre, # means model_pre0
        'model_pre01': model_pre01,
        'model_0': model_0,
        'model_1': model_1,
        'model_01': model_01,
        'model_convert': model_convert,
        
        'model_embed': model_embed,
        'model_ae': model_ae,
        'model_gk1': model_gk1,
        'model_classify': model_classify,
        
        'model_img_img': model_img_img,
        'model_img_cls': model_img_cls,
        'model_cls_img': model_cls_img,
        'model_cls_cls': model_cls_cls,
        
        'model_cls_img_cls': model_cls_img_cls,
        'model_img_cls_img': model_img_cls_img,
        
        'path': {
            'path_img_imgA': path_img_imgA,
            'path_img_cls': path_img_cls,
            'path_img_img_cls': path_img_img_cls,
            'path_cls_img': path_cls_img,
            'path_cls_cls': path_cls_cls,
            'path_cls_img_cls': path_cls_img_cls,
            'path_cls_img_cls_x2': path_cls_img_cls_x2,
            'path_cls_img_cls_img': path_cls_img_cls_img,
            'path_img_cls_img': path_img_cls_img,
            'path_fit_imgA': path_fit_imgA,
            'path_fit_cls_img': path_fit_cls_img,
            'path_fit_cls_img_cls_img': path_fit_cls_img_cls_img,
            'path_fit_img_cls_img': path_fit_img_cls_img,
        }
    }


def make_modelz(img_dim, cls_dim, num_lm,
                model_ae=None,
                wgt_embed=None, init_lm=None, init_lm2=None,
                flag_img_cls_img=False, flag_cls_img_cls_x2=False, flag_cls_img_cls_img=False,
                flag_cls_cls=False,
                r_state=None):
    '''==============================
    define inputs
    =============================='''
    inp_cls = Input(shape=(cls_dim,), name='input_cls')
    inp_img = Input(shape=(img_dim,), name='input_img')
    
    return _make_model0(inp_img=inp_img, inp_cls=inp_cls,
                        img_dim=img_dim, cls_dim=cls_dim, num_lm=num_lm,
                        model_ae=model_ae,
                        wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
                        flag_img_cls_img=flag_img_cls_img,
                        flag_cls_img_cls_x2=flag_cls_img_cls_x2,
                        flag_cls_img_cls_img=flag_cls_img_cls_img,
                        flag_cls_cls=flag_cls_cls,
                        r_state=r_state)


'''
you can input 'model_convert'
'''
def make_modely(input_dim,
                img_dim, cls_dim, num_lm,
                model_convert=None,
                model_ae=None,
                wgt_embed=None, init_lm=None, init_lm2=None,
                flag_img_cls_img=False, flag_cls_img_cls_x2=False, flag_cls_img_cls_img=False,
                flag_cls_cls=False,
                r_state=None, no_path_fit_cls_img=False,
                fix_gk1=False, fix_embed=False, fix_ae=False):
    '''
    model_convert:
      input -> model_convert -> img
    '''
    '''==============================
    define inputs
    =============================='''
    inp_cls = Input(shape=(cls_dim,), name='input_cls')
    inp = Input(shape=(input_dim,), name='input')
    
    return _make_model1(inp, inp_cls,
                img_dim=img_dim, cls_dim=cls_dim, num_lm=num_lm,
                model_convert=model_convert,
                model_ae=model_ae,
                wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
                flag_img_cls_img=flag_img_cls_img,
                flag_cls_img_cls_x2=flag_cls_img_cls_x2,
                flag_cls_img_cls_img=flag_cls_img_cls_img,
                flag_cls_cls=flag_cls_cls,
                r_state=r_state, no_path_fit_cls_img=no_path_fit_cls_img,
                fix_gk1=fix_gk1, fix_embed=fix_embed, fix_ae=fix_ae)

def make_modelx(input_dim,
                img_dim, cls_dim, num_lm,
                model_convert=None,
                model_ae=None,
                wgt_embed=None, init_lm=None, init_lm2=None,
                flag_img_cls_img=False, flag_cls_img_cls_x2=False, flag_cls_img_cls_img=False,
                flag_cls_cls=False, flag_cls_img_imgE=False,
                r_state=None, no_path_fit_cls_img=True,
                fix_gk1=False, fix_embed=False, fix_ae=False):
    '''
    model_convert:
      input -> model_convert -> img
    '''
    '''==============================
    define inputs
    =============================='''
    inp_cls = Input(shape=(cls_dim,), name='input_cls')
    inp = Input(shape=(input_dim,), name='input')
    
    return _make_model1(inp, inp_cls,
                img_dim=img_dim, cls_dim=cls_dim, num_lm=num_lm,
                model_convert=model_convert,
                model_ae=model_ae,
                wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
                flag_img_cls_img=flag_img_cls_img,
                flag_cls_img_cls_x2=flag_cls_img_cls_x2,
                flag_cls_img_cls_img=flag_cls_img_cls_img,
                flag_cls_cls=flag_cls_cls, flag_cls_img_imgE=flag_cls_img_imgE,
                r_state=r_state, no_path_fit_cls_img=no_path_fit_cls_img,
                fix_gk1=fix_gk1, fix_embed=fix_embed, fix_ae=fix_ae)
