import argparse

import networkx as nx
import pandas as pd
from py2neo import Graph

# database config
neo4j_G = Graph("http://10.12.11.179:7475", auth=("neo4j", "000000"))


def subgraph_extract(account, edge_nums, database):
    subgraph = nx.DiGraph()
    subgraph.add_node(account)  # treat target account as the first node

    trans_cyber = "match (s:CA{name:'" + account + "'})-[r:`trans_old`]-(e) " \
                                                   "return startNode(r).name as from,endNode(r).name as to, r.value as value, " \
                                                   f"r.timestamp as timestamp order by timestamp limit {edge_nums} "


    trans_res = pd.DataFrame(database.run(trans_cyber).data())
    print(trans_res)



if __name__ == '__main__':

    # account = '0x2cdb253c0e44a284f6174ae90b5ea247e6cf3649'



    parser = argparse.ArgumentParser()
    """
    可调参
    """
    parser.add_argument('-a', '--account', type=str, default='0x5fb3d432bae33fcd418ede263d98d7440e7fa3ea')
    # [10~100]
    parser.add_argument('-edge', '--edge_nums', type=int, default=10)
    args = parser.parse_args()
    database = neo4j_G
    subgraph_extract(args.account, args.edge_nums, database)