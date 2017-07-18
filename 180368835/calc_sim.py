# -*- encoding: utf-8 -*-
'''
python2.7
'''
import datetime
from SimpleXMLRPCServer import SimpleXMLRPCServer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

import gensim

modelfile = '/ssd2/var2/gensim/wikidump/sentences013.d/sentences013.model100'
modelfile_bin = '/ssd2/var2/gensim/wikidump/sentences013.d/sentences013.bin.model100'

def main():
    if False:
        '''
        バイナリ形式で、保存しなおします
        数分かかります．ご辛抱を
        '''
        model = gensim.models.word2vec.Word2Vec.load_word2vec_format(modelfile)
        model.save(modelfile_bin)
        return
    
    model = gensim.models.word2vec.Word2Vec.load(modelfile_bin)
    #model.init_sims(replace=True)
    #res = server.most_similar([u'ビッグデータ'], [], 20)
    res = model.most_similar([u'人工知能'], [], 20)
    
    for irow in res:
        print irow[0], irow[1]
    
    return

if __name__ == '__main__':
    now = datetime.datetime.now()
    main()
    print datetime.datetime.now() - now
