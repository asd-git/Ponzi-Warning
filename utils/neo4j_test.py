# -*- coding: utf-8 -*- 
# @Time : 2022/5/7 13:23 
# @Author : Jerry_Jin
# @contact: jerry.jin1114@gmail.com
# @File : neo4j_test.py
import pandas as pd
from py2neo import Graph

neo4j_G = Graph("http://10.12.11.154:7474", auth=("neo4j", "1997226"))

account = '0x5fb3d432bae33fcd418ede263d98d7440e7fa3ea'
# trans_cyber = "match (s:CA{name:'" + account + "'})-[r:`trans`]->(e) " \
#                                                    "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
#                                                    f"r.timestamp as timestamp order by timestamp"

call_cyber = "match (s)-[r:`call`]->(e:CA{name:'" + account + "'})" \
                                                               "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
                                                               f"r.timestamp as timestamp order by timestamp"
# trans_res = pd.DataFrame(neo4j_G.run(trans_cyber).data())
call_res = pd.DataFrame(neo4j_G.run(call_cyber).data())
print(call_res)