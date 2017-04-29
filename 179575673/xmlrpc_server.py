# -*- encoding: utf-8 -*-
'''
python2.7
'''
from SimpleXMLRPCServer import SimpleXMLRPCServer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

import gensim

modelfile = '/ssd2/var2/gensim/wikidump/sentences013.d/sentences013.model100'

def main():
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format(modelfile)
    model.init_sims(replace=True)
    s = SimpleXMLRPCServer(('0.0.0.0', 9001), allow_none=True)
    s.register_instance(model, allow_dotted_names=True)
    print s.server_address
    s.serve_forever()

if __name__ == '__main__':
    main()
