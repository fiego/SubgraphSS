
import gc
import time
import numpy as np
import pandas as pd
import networkx as nx

## Neighbourhood-Based Node Level Embedding
from karateclub import DeepWalk    # KDD 2014
from karateclub import Node2Vec    # KDD 2016
# from karateclub import LINE      # 无
from karateclub import RandNE      # ICDM 2018
from karateclub import Diff2Vec    # CompleNet 2018 跑不动
from karateclub import NetMF       # WSDM 2018
# from karateclub import NodeSketch  # KDD 2019 无 model._embedding
# from karateclub import BoostNE     # ASONAM 2019 无 model._embedding
from karateclub import GLEE          # Journal of Complex Networks 2020
# 无 SDNE


# Graph Level Embedding 有问题
from karateclub import Graph2Vec   # MLGWorkshop 2017
from karateclub import GL2Vec      # ICONIP 2019


## Structural Node Level Embedding
# from karateclub import GraphWave
from karateclub import Role2Vec    # IJCAI StarAI 2018 跑不动

#%%
gc.collect()  # 回收内存
def save_list_to_npy(data_list, save_path):
    data = np.array(data_list)
    np.save(save_path, data)  # 保存为.npy格式
    gc.collect()  # 回收内存
    time.sleep(10)

def read_graph():
    edges = pd.read_csv("./data/DBpedia/GEs_Corpus/edges.csv", header=0, sep=",", encoding="utf-8", low_memory=False)
    print(edges.head())
    graph = nx.from_pandas_edgelist(edges, "index_category1", "index_category2")
    del edges
    return graph


def train_DeepWalk(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/DeepWalk.npy"
    print(save_path)
    model = DeepWalk()
    model.fit(graph)
    # X = model.get_memberships()
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_Node2Vec(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/Node2Vec.npy"
    print(save_path)
    model = Node2Vec()
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_Diff2Vec(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/Diff2Vec.npy"
    print(save_path)
    model = Diff2Vec()
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_RandNE(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/RandNE.npy"
    print(save_path)
    model = RandNE()
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_NetMF(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/NetMF.npy"
    print(save_path)
    model = NetMF()
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_GLEE(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/GLEE.npy"
    print(save_path)
    model = GLEE()
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_Role2Vec(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/Role2Vec.npy"
    print(save_path)
    model = Role2Vec()
    model.fit(graph)
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

##  有问题
def train_Graph2Vec(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/Graph2Vec.npy"
    print(save_path)
    graphs = [nx.newman_watts_strogatz_graph(50, 5, 0.3) for _ in range(100)]
    model = Graph2Vec()
    model.fit(graph)
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存

def train_GL2Vec(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/GL2Vec.npy"
    print(save_path)
    model = GL2Vec()
    model.fit(graph)
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存


#%%
from karateclub import TADW    #
def train_TADW(graph):
    save_path = f"./model/Embeddings/GEs/trained_model/TADW.npy"
    print(save_path)
    model = TADW()
    model.fit(graph)
    save_list_to_npy(model._embedding, save_path)  # 保存
