# -*- encoding: utf-8 -*-
'''
python2.7
'''

'''
MeCabコストファイルを作成する
こういうやつ↓

・・・
この男子、人魚ひろいました。,0,0,0,名詞,一般,*,*,*,*,この男子、人魚ひろいました。,*,*,Wikipedia
この男子、人魚をひろいました。,0,0,0,名詞,一般,*,*,*,*,この男子、人魚をひろいました。,*,*,Wikipedia
この男子、宇宙人と戦えます。,0,0,0,名詞,一般,*,*,*,*,この男子、宇宙人と戦えます。,*,*,Wikipedia
この男子、石化に悩んでます。,0,0,0,名詞,一般,*,*,*,*,この男子、石化に悩んでます。,*,*,Wikipedia
この男子、魔法がお仕事です。,0,0,0,名詞,一般,*,*,*,*,この男子、魔法がお仕事です。,*,*,Wikipedia
・・・

後ろにoriginをセットしてある
'''
import sys
import sqlite3


db2_sqlite = '/ssd2/var2/gensim/wikidump/pagelinks_uniq_2.sqlite3'
'''↑pagelinksをユニークにしたテーブル ここからワードを引っ張ってくる'''
costfile_name = '/ssd2/var2/mecabproc/mecab_no_cost.test.eucjp'
#costfile_name = '/ssd2/var2/mecabproc/proc3/mecab_zero_cost.eucjp'
'''↑出力ファイル'''

def main():
    conn = sqlite3.connect(db2_sqlite, timeout=15)
    cur = conn.cursor()
    cur.execute('select zen, pl_title from pagelinksuniq')
    
    cnt = 0
    fw = open(costfile_name, 'w')
    for irow in cur:
        cnt += 1
        zen_unicode = irow[0]
        pl_title_uni = irow[1]
        
        if len(zen_unicode) < 2:
            continue
        mm1 = zen_unicode + u',,,' + u',名詞,一般,*,*,*,*,' + pl_title_uni + u',*,*,Wikipedia\n'
        #mm1 = zen_unicode + u',0,0,0' + u',名詞,一般,*,*,*,*,' + pl_title_uni + u',*,*,Wikipedia\n'
        try:
            fw.write(mm1.encode('euc-jp'))
        except UnicodeEncodeError:
            print zen_unicode
            print '^^^skip...'
        if cnt % 10000 == 0:
            sys.stdout.write(str(cnt)+'\r')
    fw.close()
    return


if __name__ == '__main__':
    main()
