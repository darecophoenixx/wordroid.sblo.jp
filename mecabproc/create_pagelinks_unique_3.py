# -*- encoding: utf-8 -*-
'''
python2.7
'''
import sys
import re
import datetime
import sqlite3
import itertools

import MySQLdb
import _mysql_exceptions
from MySQLdb.cursors import SSDictCursor
import zenhan

'''
sqliteに作成するテーブル
CREATE TABLE IF NOT EXISTS pagelinksuniq (
  `pl_title` TEXT NOT NULL DEFAULT '',
  `zen` TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (`pl_title`)
);
CREATE INDEX IF NOT EXISTS idx_pagelinksuniq ON pagelinksuniq (zen);
'''

db2_sqlite = '/ssd2/var2/gensim/wikidump/pagelinks_uniq_test.sqlite3'
sqlite_sql02 = '''insert into pagelinksuniq values(?, ?)'''

def conv(txt, unic=False):
    kZ = unicode(txt)
    kZ = zenhan.z2h(kZ)
    kZ = kZ.lower()
    kZ = zenhan.h2z(kZ)
    if unic:
        return kZ
    kZ = kZ.encode('utf8')
    return kZ

RE_UNDERBAR_BLACKET = re.compile(u'^(.+)_[(](.+?)[)]$')
def parse_underbar_blacket(txt):
    '''
    txt must be unicode
    '''
    res = RE_UNDERBAR_BLACKET.match(txt)
    if res is not None:
        return (res.group(1), res.group(2))
    return None

def main():
    '''
    MySQLのpagelinksとpageから
    sqlite3のpagelinks_uniqを作成
    '''
    from mysql_con import conn_mysql
    cur_mysql = conn_mysql.cursor()
    
    words = set()
    '''pageからワードを集める'''
    cur_mysql.execute('select page_title from page where page_namespace=0')
    cnt = 0
    for irow in itertools.islice(cur_mysql, None):
        cnt += 1
        iword = irow['page_title']
        words.add(iword)
        if cnt % 10000 == 0:
            print cnt, iword, len(words)
    
    '''pagelinksからワードを集める'''
    cur_mysql.execute('select * from pagelinks')
    cnt = 0
    for irow in itertools.islice(cur_mysql, None):
        cnt += 1
        iword = irow['pl_title']
        words.add(iword)
        if cnt % 10000 == 0:
            print cnt, iword
    cur_mysql.close()
    
    conn = sqlite3.connect(db2_sqlite, timeout=15)
    cur = conn.cursor()
    cnt = 0
    for iword in words:
        cnt += 1
        print iword
        try:
            iword_uni = unicode(iword)
            res = parse_underbar_blacket(iword_uni)
            if res:
                insert_pagelinksuniq(cur, res[0])
                insert_pagelinksuniq(cur, res[1])
            else:
                insert_pagelinksuniq(cur, iword_uni)
        except Exception, e:
            print [iword]
            print e
            pass
        if cnt % 10000 == 0:
            print cnt, iword
            conn.commit()
    conn.commit()

def insert_pagelinksuniq(cur, iwordunicode):
    try:
        iwordunicode_zen = conv(iwordunicode.replace(u'_', u' '), unic=True)
        cur.execute(sqlite_sql02, (iwordunicode, iwordunicode_zen))
        print iwordunicode
    except sqlite3.IntegrityError:
        pass

if __name__ == '__main__':
    now = datetime.datetime.now()
    main()
    print datetime.datetime.now() - now
