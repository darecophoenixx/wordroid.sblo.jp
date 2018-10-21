'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/HumanisticML/LICENSE.md
'''
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from keras_ex.HumanisticML.models import make_modelx, make_modely, _make_model1


class HML(object):
    
    def __init__(self, x_train, y_train, y_cat_train, x_test=None, y_test=None, y_cat_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.y_cat_train = y_cat_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_cat_test = y_cat_test
        
        self.input_dim = self.x_train.shape[1]
        self.cls_dim = self.y_cat_train.shape[1]
    
    def clear(self):
        self.x_train_img = None
        self.x_test_img = None
        self._pred_gk1_train = None
        self._pred_gk1_test = None
        self._pred_imgA_train = None
        self._pred_imgA_test = None
        self._pred_imgA_gk1_train = None
        self._pred_imgA_gk1_test = None
        self._pred_cls_img_train = None
        self._pred_cls_img_test = None
        self._pred_cls_img_gk1_train = None
        self._pred_cls_img_gk1_test = None
        self._pred_cls_cls_train = None
        self._pred_cls_cls_test = None
        self._pred_cls_img_cls_train = None
        self._pred_cls_img_cls_test = None
        self._pred_img_cls_train = None
        self._pred_img_cls_test = None
        self._pred_img_img_cls_train = None
        self._pred_img_img_cls_test = None
        self._pred_img_cls_img_train = None
        self._pred_img_cls_img_test = None
    
    def make_model(self, img_dim, num_lm,
                   flag_img_cls_img=False,
                   flag_cls_img_cls_x2=False,
                   flag_cls_img_cls_img=False,
                   flag_cls_cls=False,
                   no_path_fit_cls_img=False,
                   model_convert=None,
                   model_ae=None,
                   wgt_embed=None, init_lm=None, init_lm2=None,
                   fix_gk1=False, fix_embed=False, fix_ae=False):
        self.img_dim = img_dim
        self.num_lm = num_lm
        
        models = make_modely(input_dim=self.input_dim,
                             img_dim=img_dim, cls_dim=self.cls_dim, num_lm=num_lm,
                             flag_img_cls_img=flag_img_cls_img,
                             flag_cls_img_cls_x2=flag_cls_img_cls_x2,
                             flag_cls_img_cls_img=flag_cls_img_cls_img,
                             flag_cls_cls=flag_cls_cls,
                             no_path_fit_cls_img=no_path_fit_cls_img,
                             model_convert=model_convert, model_ae=model_ae,
                             wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
                             fix_gk1=fix_gk1, fix_embed=fix_embed, fix_ae=fix_ae)
        self.models = models
        self.model = models['model']
    
    def _val_data(self):
        X_test_dummy = np.zeros((self.y_cat_test.shape[0], 1))
        val_data = (
            {'input': self.x_test, 'input_cls': self.y_cat_test},
            {
                'path_fit_cls_img': X_test_dummy,
                'path_cls_img_cls': self.y_cat_test,
                'path_fit_imgA': X_test_dummy,
                'path_fit_img_cls_img': X_test_dummy,
                'path_cls_cls': self.y_cat_test,
                'path_img_cls': self.y_cat_test,
                'path_img_img_cls': self.y_cat_test,
                'path_fit_cls_img_imgE': X_test_dummy,
            }
        )
        return val_data
    
    def _train_data(self):
        X_dummy = np.zeros((self.y_cat_train.shape[0], 1))
        train_data = (
            {'input': self.x_train, 'input_cls': self.y_cat_train},
            {
               'path_fit_cls_img': X_dummy,
               'path_cls_img_cls': self.y_cat_train,
               'path_fit_imgA': X_dummy,
               'path_fit_img_cls_img': X_dummy,
               'path_cls_cls': self.y_cat_train,
               'path_img_cls': self.y_cat_train,
               'path_img_img_cls': self.y_cat_train,
               'path_fit_cls_img_imgE': X_dummy,
            }
        )
        return train_data
    
    def evaluate_test(self):
        val_data = self._val_data()
        res = self.model.evaluate(val_data[0], val_data[1])
        return res
    
    def evaluate(self):
        train_data = self._train_data()
        res = self.model.evaluate(train_data[0], train_data[1])
        return res
    
    def _fit(self, model, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        X_dummy = np.zeros((self.y_cat_train.shape[0], 1))
        
        val_data = None
        if val_flag and self.x_test is not None:
            X_test_dummy = np.zeros((self.y_cat_test.shape[0], 1))
            val_data = self._val_data()
        train_data_x, train_data_y = self._train_data()
        res = model.fit(train_data_x, train_data_y,
                        validation_data=val_data,
                        verbose=verbose,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks)
        self.clear()
        return res
    
    def fit_0(self, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        model_0 = self.models['model_0']
        return self._fit(model_0,
                         epochs=epochs, verbose=verbose, batch_size=batch_size,
                         val_flag=val_flag, callbacks=callbacks)
    
    def fit_1(self, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        model_1 = self.models['model_1']
        return self._fit(model_1,
                         epochs=epochs, verbose=verbose, batch_size=batch_size,
                         val_flag=val_flag, callbacks=callbacks)
    
    def fit_01(self, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        model_01 = self.models['model_01']
        return self._fit(model_01,
                         epochs=epochs, verbose=verbose, batch_size=batch_size,
                         val_flag=val_flag, callbacks=callbacks)
    
    def fit_pre(self, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        model_pre = self.models['model_pre']
        return self._fit(model_pre,
                         epochs=epochs, verbose=verbose, batch_size=batch_size,
                         val_flag=val_flag, callbacks=callbacks)
    
    def fit_pre01(self, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        model_pre01 = self.models['model_pre01']
        return self._fit(model_pre01,
                         epochs=epochs, verbose=verbose, batch_size=batch_size,
                         val_flag=val_flag, callbacks=callbacks)
    
    def fit(self, epochs=3, verbose=1, batch_size=32, val_flag=False,
            callbacks=None):
        return self._fit(self.model,
                         epochs=epochs, verbose=verbose, batch_size=batch_size,
                         val_flag=val_flag, callbacks=callbacks)
    
    def _calc_img(self, batch_size=32, verbose=0):
        model_convert = self.get_model('convert')
        if model_convert is None:
            self.x_train_img = self.x_train.copy()
            try:
                self.x_test_img = self.x_test.copy()
            except:
                self.x_test_img = None
            return
        
        self.x_train_img = model_convert.predict(self.x_train, batch_size=batch_size, verbose=verbose)
        print(self.x_train_img.shape)
        
        self.x_test_img =None
        if self.x_test is not None:
            self.x_test_img = model_convert.predict(self.x_test, batch_size=batch_size, verbose=verbose)
            print(self.x_test_img.shape)
    
    def calc_pred_img_cls_img_train(self):
        if self._pred_img_cls_img_train is None:
            self._pred_img_cls_img_train = self.calc_img_cls_img(self.x_train_img, verbose=1)
        return self._pred_img_cls_img_train
    pred_img_cls_img_train = property(calc_pred_img_cls_img_train)
    
    def calc_pred_img_cls_img_test(self):
        if self._pred_img_cls_img_test is None:
            self._pred_img_cls_img_test = self.calc_img_cls_img(self.x_test_img, verbose=1)
        return self._pred_img_cls_img_test
    pred_img_cls_img_test = property(calc_pred_img_cls_img_test)
    
    def calc_pred_img_img_cls_train(self):
        if self._pred_img_img_cls_train is None:
            self._pred_img_img_cls_train = self.calc_img_cls(self.pred_imgA_train, verbose=1)
        return self._pred_img_img_cls_train
    pred_img_img_cls_train = property(calc_pred_img_img_cls_train)
    
    def calc_pred_img_img_cls_test(self):
        if self._pred_img_img_cls_test is None:
            self._pred_img_img_cls_test = self.calc_img_cls(self.pred_imgA_test, verbose=1)
        return self._pred_img_img_cls_test
    pred_img_img_cls_test = property(calc_pred_img_img_cls_test)
    
    def calc_pred_img_cls_train(self):
        if self._pred_img_cls_train is None:
            self._pred_img_cls_train = self.calc_img_cls(self.x_train_img, verbose=1)
        return self._pred_img_cls_train
    pred_img_cls_train = property(calc_pred_img_cls_train)
    
    def calc_pred_img_cls_test(self):
        if self._pred_img_cls_test is None:
            self._pred_img_cls_test = self.calc_img_cls(self.x_test_img, verbose=1)
        return self._pred_img_cls_test
    pred_img_cls_test = property(calc_pred_img_cls_test)
    
    def calc_pred_cls_img_cls_train(self):
        if self._pred_cls_img_cls_train is None:
            self._pred_cls_img_cls_train = self.calc_cls_img_cls(self.y_cat_train, verbose=1)
        return self._pred_cls_img_cls_train
    pred_cls_img_cls_train = property(calc_pred_cls_img_cls_train)
    
    def calc_pred_cls_img_cls_test(self):
        if self._pred_cls_img_cls_test is None:
            self._pred_cls_img_cls_test = self.calc_cls_img_cls(self.y_cat_test, verbose=1)
        return self._pred_cls_img_cls_test
    pred_cls_img_cls_test = property(calc_pred_cls_img_cls_test)
    
    def calc_pred_cls_cls_train(self):
        if self._pred_cls_cls_train is None:
            self._pred_cls_cls_train = self.calc_cls_cls(self.y_cat_train, verbose=1)
        return self._pred_cls_cls_train
    pred_cls_cls_train = property(calc_pred_cls_cls_train)
    
    def calc_pred_cls_cls_test(self):
        if self._pred_cls_cls_test is None:
            self._pred_cls_cls_test = self.calc_cls_cls(self.y_cat_test, verbose=1)
        return self._pred_cls_cls_test
    pred_cls_cls_test = property(calc_pred_cls_cls_test)
    
    def calc_pred_cls_img_gk1_train(self):
        if self._pred_cls_img_gk1_train is None:
            self._pred_cls_img_gk1_train = self.calc_gk1(self.pred_cls_img_train, verbose=1)
        return self._pred_cls_img_gk1_train
    pred_cls_img_gk1_train = property(calc_pred_cls_img_gk1_train)
    
    def calc_pred_cls_img_gk1_test(self):
        if self._pred_cls_img_gk1_test is None:
            self._pred_cls_img_gk1_test = self.calc_gk1(self.pred_cls_img_test, verbose=1)
        return self._pred_cls_img_gk1_test
    pred_cls_img_gk1_test = property(calc_pred_cls_img_gk1_test)
    
    def calc_pred_cls_img_train(self):
        if self._pred_cls_img_train is None:
            self._pred_cls_img_train = self.calc_cls_img(self.y_cat_train, verbose=1)
        return self._pred_cls_img_train
    pred_cls_img_train = property(calc_pred_cls_img_train)
    
    def calc_pred_cls_img_test(self):
        if self._pred_cls_img_test is None:
            self._pred_cls_img_test = self.calc_cls_img(self.y_cat_test, verbose=1)
        return self._pred_cls_img_test
    pred_cls_img_test = property(calc_pred_cls_img_test)
    
    def calc_pred_imgA_gk1_train(self):
        if self._pred_imgA_gk1_train is None:
            self._pred_imgA_gk1_train = self.calc_gk1(self.pred_imgA_train, verbose=1)
        return self._pred_imgA_gk1_train
    pred_imgA_gk1_train = property(calc_pred_imgA_gk1_train)
    
    def calc_pred_imgA_gk1_test(self):
        if self._pred_imgA_gk1_test is None:
            self._pred_imgA_gk1_test = self.calc_gk1(self.pred_imgA_test, verbose=1)
        return self._pred_imgA_gk1_test
    pred_imgA_gk1_test = property(calc_pred_imgA_gk1_test)
    
    def calc_pred_imgA_train(self):
        if self._pred_imgA_train is None:
            self._pred_imgA_train = self.calc_img_img(self.x_train_img, verbose=1)
        return self._pred_imgA_train
    pred_imgA_train = property(calc_pred_imgA_train)
    
    def calc_pred_imgA_test(self):
        if self._pred_imgA_test is None:
            self._pred_imgA_test = self.calc_img_img(self.x_test_img, verbose=1)
        return self._pred_imgA_test
    pred_imgA_test = property(calc_pred_imgA_test)
    
    def calc_pred_gk1_train(self):
        if self._pred_gk1_train is None:
            self._pred_gk1_train = self.calc_gk1(self.x_train_img, verbose=1)
        return self._pred_gk1_train
    pred_gk1_train = property(calc_pred_gk1_train)
    
    def calc_pred_gk1_test(self):
        if self._pred_gk1_test is None:
            self._pred_gk1_test = self.calc_gk1(self.x_test_img, verbose=1)
        return self._pred_gk1_test
    pred_gk1_test = property(calc_pred_gk1_test)
    
    
    
    
    def classification_report(self, y, pred_cls):
        print(classification_report(y, np.argmax(pred_cls, axis=1)))
        print(f1_score(y, np.argmax(pred_cls, axis=1), average='macro'))
        res = confusion_matrix(y, np.argmax(pred_cls, axis=1))
        print(res)
        return res
    
    def get_model(self, nm):
        return self.models.get('model_'+nm)
    
    def calc_convert(self, X, batch_size=32, verbose=0):
        model_convert = self.get_model('convert')
        res = model_convert.predict(X, batch_size=batch_size, verbose=verbose)
        print('calc_convert >>>', res.shape)
        return res
    
    def calc_img_img(self, img, batch_size=32, verbose=0):
        model = self.get_model('img_img')
        try:
            res = model.predict(img, batch_size=batch_size, verbose=verbose)
            print('calc_img_img >>>', res.shape)
            return res
        except:
            return None
    
    def calc_img_cls(self, img, batch_size=32, verbose=0):
        model = self.get_model('img_cls')
        try:
            res = model.predict(img, batch_size=batch_size, verbose=verbose)
            print('calc_img_cls >>>', res.shape)
            return res
        except:
            return None
    
    def calc_img_cls_img(self, img, batch_size=32, verbose=0):
        model = self.get_model('img_cls_img')
        try:
            res = model.predict(img, batch_size=batch_size, verbose=verbose)
            print('calc_img_cls_img >>>', res.shape)
            return res
        except:
            return None
    
    def calc_cls_img(self, cat, batch_size=32, verbose=0):
        model = self.get_model('cls_img')
        try:
            res = model.predict(cat, batch_size=batch_size, verbose=verbose)
            print('calc_cls_img >>>', res.shape)
            return res
        except:
            return None
    
    def calc_cls_cls(self, cat, batch_size=32, verbose=0):
        model = self.get_model('cls_cls')
        try:
            res = model.predict(cat, batch_size=batch_size, verbose=verbose)
            print('calc_cls_cls >>>', res.shape)
            return res
        except:
            return None
    
    def calc_cls_img_cls(self, cat, batch_size=32, verbose=0):
        model = self.get_model('cls_img_cls')
        try:
            res = model.predict(cat, batch_size=batch_size, verbose=verbose)
            print('calc_cls_img_cls >>>', res.shape)
            return res
        except:
            return None
    
    def calc_gk1(self, img, batch_size=32, verbose=0):
        model = self.get_model('gk1')
        try:
            res = model.predict(img, batch_size=batch_size, verbose=verbose)
            print('calc_gk1 >>>', res.shape)
            return res
        except:
            return None


class HMLx(HML):
    
    def make_model(self, img_dim, num_lm,
                   flag_img_cls_img=False,
                   flag_cls_img_cls_x2=False,
                   flag_cls_img_cls_img=False,
                   flag_cls_cls=False,
                   flag_cls_img_imgE=True,
                   model_convert=None,
                   model_ae=None,
                   wgt_embed=None, init_lm=None, init_lm2=None,
                   no_path_fit_cls_img=True,
                   fix_gk1=False, fix_embed=False, fix_ae=False):
        self.img_dim = img_dim
        self.num_lm = num_lm
        
        models = make_modelx(input_dim=self.input_dim,
                             img_dim=img_dim, cls_dim=self.cls_dim, num_lm=num_lm,
                             flag_img_cls_img=flag_img_cls_img,
                             flag_cls_img_cls_x2=flag_cls_img_cls_x2,
                             flag_cls_img_cls_img=flag_cls_img_cls_img,
                             flag_cls_cls=flag_cls_cls,
                             flag_cls_img_imgE=flag_cls_img_imgE,
                             model_convert=model_convert, model_ae=model_ae,
                             wgt_embed=wgt_embed, init_lm=init_lm, init_lm2=init_lm2,
                             no_path_fit_cls_img=no_path_fit_cls_img,
                             fix_gk1=fix_gk1, fix_embed=fix_embed, fix_ae=fix_ae)
        self.models = models
        self.model = models['model']


class Seq(object):
    '''
    datagen must be fitted
    x_train need 4 dimension
    '''
    
    def __init__(self, datagen, x_train, y_cat_train, batch_size=32,
                 shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png'):
        self.datagen = datagen
        self.x_train = x_train
        self.y_cat_train = y_cat_train
        self.batch_size = batch_size
        
        '''estimate self length'''
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1
        
        self.initialize_it()
        
        self.gen = self.datagen.flow(x_train, y_cat_train,
                                     batch_size=batch_size,
                                     shuffle=shuffle, seed=seed,
                                     save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)
    
    def initialize_it(self):
        self.it = iter(range(0, len(self.x_train), self.batch_size))
        self.idx_next = self.it.__next__()
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        res_x, res_y = next(self.gen)
        res_x = res_x.copy().reshape((len(res_x), -1))
        x_dummy = np.zeros((len(res_x), 1))
        x = {'input': res_x, 'input_cls': res_y}
        y = {
            'path_fit_cls_img': x_dummy,
            'path_cls_img_cls': res_y,
            'path_fit_imgA': x_dummy,
            'path_fit_img_cls_img': x_dummy,
            'path_cls_cls': res_y,
            'path_img_cls': res_y,
            'path_img_img_cls': res_y,
            'path_fit_cls_img_imgE': x_dummy,
        }
        return (x, y)


