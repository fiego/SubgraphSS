import os
#%% 切换至合适的路径
print(os.getcwd())
root_path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # 上上级目录: "../.." ；上级目录: ".."
os.chdir(root_path)
print(os.getcwd())
#%%
import gc
import torch
import KGEs
#%%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gc.collect()
torch.cuda.empty_cache()
#%%
if __name__=="__main__":
    model = "PairRE"
    KEGs.KnowledgeGraphEmbeddings(model)

## dim = 100, batch_size=64
## SZU
## 2020.8.30 DBpedia
# K1 PairRE  2020
# K2 CrossE  2019
# K3 TuckER  2019
# K4 ProjE   2017
# K5 HolE    2016
# K6 TransH  2014

## no
# K6 TransR  2015
# K7 TransE  2013 no




## dim = 100, batch_size=128
## SZU
## 2020.7.30 DBpedia 8.21 zhongduan
## 2020.8.26 DBpedia
# K1 TransE  2013
# K2 TransH  2014
# K3 CrossE  2019
# K4 ProjE   2017
# K5 TuckER  2019
# K6 HolE  2016
# K7


## SCNU
## 2020.7.22 DBpedia
# K1 PairRE  2020
# K2 TransR  2015
# SimplE  2018 待跑
#
