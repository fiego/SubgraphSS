
'''
https://github.com/benedekrozemberczki/karateclub
'''

#%% 切换至合适的路径
import os
print(os.getcwd())
root_path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 上上级目录: "../.." ；上级目录: ".."
os.chdir(root_path)
print(os.getcwd())

import GEs

#%%
if __name__=="__main__":
    graph = GEs.read_graph()

    # ["DeepWalk", "NetMF", "GLEE", "RandNE", "Node2Vec"]
    GEs.train_DeepWalk(graph)   #
    # GEs.train_Node2Vec(graph) #
    # GEs.train_RandNE(graph)   #
    # GEs.train_GLEE(graph)     #
    # GEs.train_NetMF(graph)    #

    # GEs.train_Diff2Vec(graph)  # 无向图
    # GEs.train_Role2Vec(graph)  # 无向图
    # GEs.train_Graph2Vec(graph) #
    # GEs.train_GL2Vec(graph)



