
import torch
import numpy as np
from gensim import matutils
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from model.TEs import TextEmbeddings
from model.GEs import GraphEmbeddings
from model.KGEs import KnowledgeGraphEmbeddings

#%% cosine_similarity
def measure_similarity(a_vec, b_vec, sim_type="L2squared", std_vec=None):
    sim = 0.0  # 0.0
    if sim_type == "L1":
        a_vec = torch.from_numpy(a_vec)
        b_vec = torch.from_numpy(b_vec)
        sim = torch.pairwise_distance(a_vec, b_vec, p=1).numpy()[0]
    elif sim_type == "L2":
        a_vec = torch.from_numpy(a_vec)
        b_vec = torch.from_numpy(b_vec)
        sim = (torch.pairwise_distance(a_vec, b_vec, p=2).numpy()[0])/len(a_vec)
        # sim = numpy.sqrt(numpy.sum(numpy.square(a_vec - a_vec)))  ## 或者 sim = numpy.linalg.norm(vec1 - vec2)
    elif sim_type == "L2squared":
        a_vec = torch.from_numpy(a_vec)
        b_vec = torch.from_numpy(b_vec)
        sim = torch.pow(torch.pairwise_distance(a_vec, b_vec, p=2), 2).numpy()[0]
    elif sim_type == "euclidean":
        # a_vec = normalize(a_vec, axis=1)
        # b_vec = normalize(b_vec, axis=1)
        if np.any(a_vec) and np.any(b_vec):  # 两个都为非0向量
            a_vec = matutils.unitvec(a_vec)
            b_vec = matutils.unitvec(b_vec)
            sim = (cdist(a_vec, b_vec, 'euclidean')[0][0])**2
            sim = 1.0/(1.0 + sim)  # 用欧式距离的倒数
        else:
            sim = 0.0  # 0.0
    elif sim_type == "seuclidean":
        if np.any(a_vec) and np.any(b_vec):  # 两个都为非0向量
            # a_vec = matutils.unitvec(a_vec)
            # b_vec = matutils.unitvec(b_vec)
            # X = np.vstack((a_vec, b_vec))
            # std_vec = np.std(X, axis=0, ddof=1)
            # sim = cdist(a_vec, b_vec, 'seuclidean', V=std_vec)[0][0]
            # sim = scipy.spatial.distance.pdist(X, metric='seuclidean', V=std_vec)[0]
            # X = np.vstack((a_vec, a_vec))
            # sk_vec = np.var(X, axis=0, ddof=1)
            # std_vec = np.var(X, axis=0, ddof=1)
            sim = np.sqrt(((a_vec - b_vec) ** 2 / std_vec).sum())
        else:
            sim = 0.0  # 0.0
    elif sim_type == "cosine":
        if np.any(a_vec) and np.any(b_vec):  # 两个都为非0向量
            a_vec = matutils.unitvec(a_vec)
            b_vec = matutils.unitvec(b_vec)
            sim = (1. - cdist(a_vec, b_vec, 'cosine'))[0][0]  # [-1,1] 会出现负的余弦值
        else:
            sim = 0.0  # 0.0
    else:
        pass
    return sim

'''
https://stats.stackexchange.com/questions/198810/interpreting-negative-cosine-similarity
minx = -1
maxx = 1
cos_sim(row1, row2)- minx)/(maxx-minx)
'''
#%% TextEmbeddings Similarity
class TEsSim(TextEmbeddings):
    def __init__(self):
        super(TEsSim, self).__init__()
        # self.Wa = Wa

    def sim_TEsArticle2Vector(self, Wa, articleID1, articleID2):
        sim = 0.0
        if articleID1 == articleID2:
            sim = 10.0
        else:
            ## 利用Article对应category的文本向量增强
            article2vec1 = self.get_average_TEsCategory_Enhanced_TEsArticle2Vector(Wa, articleID1)  # 1215 新增
            article2vec2 = self.get_average_TEsCategory_Enhanced_TEsArticle2Vector(Wa, articleID2)
            
            sim = measure_similarity(article2vec1, article2vec2, sim_type="euclidean")
        return sim

#%% KnowledgeGraphEmbeddings Similarity
class KGEsSim(KnowledgeGraphEmbeddings):
    def __init__(self, mode_name):
        super(KGEsSim, self).__init__(mode_name)
        self.KGEs_name = mode_name
        # self.Wa = Wa

    def sim_KGEsArticle2Vector(self, Wa, articleID1, articleID2):  ## 输入：articleID
        if (articleID1 == articleID2):
            sim = 10.0
        else:
            KGEsArticle2vec1 = self.get_Enhanced_KGEsArticle2Vector(Wa, articleID1)  # 0830修改
            KGEsArticle2vec2 = self.get_Enhanced_KGEsArticle2Vector(Wa, articleID2)
            sim = measure_similarity(KGEsArticle2vec1, KGEsArticle2vec2, sim_type="euclidean")  # 计算两个向量的相似度 cosine L2squared
        return sim

#%% GraphEmbeddings Similarity
class GEsSim(GraphEmbeddings):
    def __init__(self, mode_name):
        super(GEsSim, self).__init__(mode_name)
        self.mode_name = mode_name
        # self.Wa = Wa

    def sim_GEsArticle2Vector(self, Wa, articleID1, articleID2):  ## 输入：articleID
        sim = 0.0
        CatSets1 = self.get_CategorySets_from_Article(articleID1)  # get_CategorySets_from_category
        CatSets2 = self.get_CategorySets_from_Article(articleID2)
        if (articleID1 == articleID2):
            sim = 10.0
        elif (CatSets1 is None) or (CatSets2 is None):  # 有一个catogory为空，并集(or |)
            sim = 0.0  #
        else:
            avg_GEsCategory2vec1 = self.get_average_GEsCategorySets2Vector(Wa, CatSets1)  ##
            avg_GEsCategory2vec2 = self.get_average_GEsCategorySets2Vector(Wa, CatSets2)
            sim = measure_similarity(avg_GEsCategory2vec1, avg_GEsCategory2vec2, sim_type="euclidean")  # 计算两个向量的相似度
        return sim


