# -*- encoding: utf-8 -*-
'''
python2.7
'''
import re
import itertools
import datetime

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import gensim

class MySentences(object):
    RE_AT = re.compile(u'^@')
    
    def __init__(self, fname, start=0, stop=None, step=1):
        self.fname = fname
        self.start = start
        self.stop = stop
        self.step = step
        self.counter = 0
    
    def __iter__(self):
        with open(self.fname) as fin:
            it = itertools.islice(fin, self.start, self.stop, self.step)
            for line in it:
                try:
                    lineunicode = unicode(line, encoding='utf8', errors='strict')
                except:
                    print lineunicode
                    #raise Exception()
                self.counter += 1
                ret = lineunicode.split()
                if self.RE_AT.match(ret[0]):
                    ret.pop(0)
                yield ret

sentencesfile = '/ssd2/var2/gensim/wikidump/sentences013'
modelfile =     '/ssd2/var2/gensim/wikidump/sentences013.d/sentences013.model100'

def main():
    sentences = MySentences(sentencesfile)
#    for iline in itertools.islice(sentences, 0, 25, 1):
#        print ' '.join(iline)
#    return
    
    print 'start training...'
    model = gensim.models.Word2Vec(sentences)
    model.save_word2vec_format(modelfile)
    return

if __name__ == '__main__':
    now = datetime.datetime.now()
    main()
    print datetime.datetime.now() - now
