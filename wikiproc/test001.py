# -*- encoding: utf-8 -*-
'''
python2.7
'''
import bz2
import itertools
import gensim

wikixml = '/ssd2/var2/gensim/wikidump/jawiki-20161020-pages-articles.xml.bz2'

def main():
    parseHandler = gensim.corpora.wikicorpus.extract_pages(bz2.BZ2File(wikixml), ['0'])
    for wikititle_uni, pageElemStr, page_id  in itertools.islice(parseHandler, 0, 3, 1):
        if len(pageElemStr) == 0:
            continue
        print page_id, wikititle_uni
        c = gensim.corpora.wikicorpus.filter_wiki(pageElemStr)
        print c

if __name__ == '__main__':
    main()
