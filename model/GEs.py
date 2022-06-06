
import json
import numpy as np
from utils import read
# from model.BaseModel import BaseModel
#%%
class GraphEmbeddings(object):  # object
    def __init__(self, mode_name = "Node2Vec"):  # GEs_name = "Node2Vec"
        super(GraphEmbeddings, self).__init__()
        self.mode_name = mode_name  # # DeepWalk NetMF GLEE RandNE Node2Vec
        self.categoryIndex2ID_dict = read.load_categoryIndex2ID_dict(f"./data/DBpedia/GEs_Corpus/category2id_index.csv")
        self.categoryIndex_embeddings = np.load(f"./model/Embeddings/GEs/trained_model/{self.mode_name}.npy").tolist()
        self.articleID_categoryIDs_dict = json.load(open(f"./data/DBpedia/articleID_categoryIDs.json", "r"))
        self.categoryIDs_childCategoryIDs_dict = json.load(open(f"./data/DBpedia/categoryIDs_childCategoryIDs.json", "r"))
        self.embedding_dim = len(self.categoryIndex_embeddings[0])

# %%
    def get_CategorySets_from_Article(self, articleID):
        ''' 通过article找category
        '''
        articleID = str(articleID)  ##
        if articleID in self.articleID_categoryIDs_dict.keys():
            categoryIDs_list = self.articleID_categoryIDs_dict[articleID]
            return set(categoryIDs_list)
        else:
            return None

#%% get_GraphEmbeddings
    def get_average_GEsCategorySets2Vector(self, Wa, categoryID_sets):
        ''' 输入：categoryID_sets, 输出：CategorySets节点的平均向量avg_GEsCategory2Vec.
        '''
        avg_GEsCategory2vec = np.zeros(self.embedding_dim, dtype=float).reshape(1, -1)  # 全0向量初始值
        num = 0
        for categoryID in categoryID_sets:
            GEsCategory2vec = self.get_child_Enhanced_GEsCategory2Vector(Wa, categoryID)  # 使用孩子邻居增强
            if np.any(GEsCategory2vec):  # 判断非0向量
                avg_GEsCategory2vec += GEsCategory2vec
                num += 1
        if num >= 1:
            avg_GEsCategory2vec = (avg_GEsCategory2vec / num).reshape(1, -1)  #.tolist()
            ## axis=0，计算每一列的均值; # axis = 1计算每一行的均值
            # avg_GEsCategory2vec = (avg_GEsCategory2vec.reshape(1, -1) - self.avg_vec.reshape(1, -1)) / self.std_vec.reshape(1, -1)  ##标准化欧氏距离
        return avg_GEsCategory2vec

    def get_GEsCategory2Vector(self, categoryID):
        ''' 输入：categoryID, 输出：category节点的向量GEsCategory2Vec.
        '''
        try:
            categoryIndex = self.categoryIndex2ID_dict[str(categoryID)]  # categoryID --> categoryIndex
            GEsCategory2vec = np.array(self.categoryIndex_embeddings[int(categoryIndex)])
        except:
            GEsCategory2vec = np.zeros(self.embedding_dim, dtype=float)  # 全0向量初始值
            # print(f'GEsCategory2Vector, No categoryID: {categoryID}')
        return GEsCategory2vec.reshape(1, -1)

    # %%
    def get_child_Enhanced_GEsCategory2Vector(self, Wa, categoryID):  # 使用孩子节点增强
        childEnh_GEsCategory2vec = self.get_GEsCategory2Vector(categoryID)  # 使用category节点向量 初始化
        if np.any(childEnh_GEsCategory2vec):    # 判断非0向量
            child_GEsCategory2vec = self.get_avg_first_child_GEsCategory2Vector(categoryID)  # 获取该节点(聚合到第1层)孩子节点的平均向量
            # child_GEsCategory2vec = self.get_avg_second_child_GEsCategory2Vector(Wa, categoryID)  # 获取该节点(聚合到第2层)孩子节点的平均向量
            if np.any(child_GEsCategory2vec):  # 判断非0向量
                childEnh_GEsCategory2vec = Wa * childEnh_GEsCategory2vec + (1-Wa) * child_GEsCategory2vec  #
        return childEnh_GEsCategory2vec.reshape(1, -1)

    # %%
    def get_avg_first_child_GEsCategory2Vector(self, categoryID):  # 获取该节点孩子节点的平均向量(第1层)
        child_GEsCategory2vec = np.zeros(self.embedding_dim, dtype=float).reshape(1, -1)  # 全0向量初始值
        try:
            childCategoryIDs_list = self.categoryIDs_childCategoryIDs_dict[str(categoryID)]  ## 使用孩子节点增强
            num = 0
            for cID in childCategoryIDs_list:
                child_GEsCategory2vec = self.get_GEsCategory2Vector(cID)  # func
                if np.any(child_GEsCategory2vec):  # 判断非0向量
                    child_GEsCategory2vec += child_GEsCategory2vec
                    num += 1
            if num >= 1:
                child_GEsCategory2vec = (child_GEsCategory2vec / num).reshape(1, -1)  # .tolist()
        except:
            pass
        return child_GEsCategory2vec

    def get_avg_second_child_GEsCategory2Vector(self, Wa, categoryID):  # 获取该节点孩子节点的平均向量(第2层)
        child_GEsCategory2vec1 = np.zeros(self.embedding_dim, dtype=float).reshape(1, -1)  # 全0向量初始值，第1层孩子节点
        try:
            childCategoryIDs_list1 = self.categoryIDs_childCategoryIDs_dict[str(categoryID)]  ## 第1层孩子节点
            num1 = 0  # 统计第1层子类数目
            for cID1 in childCategoryIDs_list1:
                num2 = 0  # 统计第2层子类数目
                GEsCategory2vec1 = self.get_GEsCategory2Vector(cID1)  # func
                avg_child_GEsCategory2vec2 = np.zeros(self.embedding_dim, dtype=float).reshape(1, -1)  # 全0向量初始值，第2层孩子节点
                try:
                    childCategoryIDs_list2 = self.categoryIDs_childCategoryIDs_dict[str(cID1)]  ## 第2层孩子节点
                    for cID2 in childCategoryIDs_list2:
                        GEsCategory2vec2 = self.get_GEsCategory2Vector(cID2)  # func
                        if np.any(GEsCategory2vec2):  # 判断非0向量
                            avg_child_GEsCategory2vec2 += GEsCategory2vec2
                            num2 += 1
                            
                    if num2 >= 1:
                        avg_child_GEsCategory2vec2 = (avg_child_GEsCategory2vec2 / num2).reshape(1, -1)  # .tolist()
                        
                    child_GEsCategory2vec2 = Wa*GEsCategory2vec1 + (1-Wa)*avg_child_GEsCategory2vec2 ##
                    
                    if np.any(child_GEsCategory2vec2):  # 判断非0向量
                        child_GEsCategory2vec1 += child_GEsCategory2vec2
                        num1 += 1
                except:
                    pass
                
            if num1 >= 1:
                child_GEsCategory2vec1 = (child_GEsCategory2vec1 / num1).reshape(1, -1)  # .tolist()
        except:
            pass
        return child_GEsCategory2vec1


