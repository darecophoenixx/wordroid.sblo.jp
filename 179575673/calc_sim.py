# -*- encoding: utf-8 -*-
'''
python2.7
'''
'''
XMLRPCServerにリクエストする
'''
import xmlrpclib

server = xmlrpclib.ServerProxy('http://localhost:9001', allow_none=True)

def main():
    #print server.get(u'ヒアルロン酸')
    res = server.most_similar([u'ヒアルロン酸'], [], 20)
    
    for irow in res:
        print irow[0], irow[1]
    return

if __name__ == '__main__':
    main()
