import networkx as nx
import pandas as pd
from py2neo import Graph

# database config
neo4j_G = Graph("http://10.12.11.179:7475", auth=("neo4j", "000000"))


def subgraph_extract(account, edge_nums, database):
    subgraph = nx.DiGraph()
    subgraph.add_node(account)  # treat target account as the first node

    trans_cyber = "match (s:CA{name:'" + account + "'})-[r:`trans_old`]->(e) " \
                                                   "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
                                                   f"r.timestamp as timestamp order by timestamp limit {edge_nums} "

    call_cyber = "match (s)-[r:`trans_old`]->(e:CA{name:'" + account + "'})" \
                                                                   "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
                                                                   f"r.timestamp as timestamp order by timestamp limit {edge_nums} "

    trans_res = pd.DataFrame(database.run(trans_cyber).data())
    call_res = pd.DataFrame(database.run(call_cyber).data())
    # print(trans_res)
    # print(call_res)
    df = pd.concat([call_res, trans_res])
    new_df = df.sort_values(by=['timestamp','from'],ascending=[True,False]).reset_index(drop=True)
    print(new_df.iloc[:edge_nums, :])


if __name__ == '__main__':
    account = '0x2cdb253c0e44a284f6174ae90b5ea247e6cf3649'
    edge_nums = 10
    database = neo4j_G
    subgraph_extract(account, edge_nums, database)