
import json
import numpy as np
# from utils import utils
# from model.BaseModel import BaseModel
# from model.TFIDF import TFIDF
from model.Word2Vec import Word2Vec
from model.fastText import fastText
from model.Doc2Vec import Doc2Vec
# from model.SentenceBERT import SentenceBERT
# from model.fastText import fastText

# %%
class TextEmbeddings(Doc2Vec):  # Word2Vec fastText Doc2Vec SentenceBERT TFIDF
    def __init__(self):
        super(TextEmbeddings, self).__init__()
        self.categoryID_articleIDs_dict = json.load(open(f"./data/DBpedia/categoryID_articleIDs.json", "r"))
        self.articleID_categoryIDs_dict = json.load(open(f"./data/DBpedia/articleID_categoryIDs.json", "r"))
        self.mappingbased_articleID_articleIDs_dict = json.load(open(f"./data/DBpedia/mappingbased_articleID_centerArticleIDs.json", "r"))  #中心

#%%  # 1215 新增 average_Category关系增强
    def get_average_TEsCategory_Enhanced_TEsArticle2Vector(self, Wa, articleID):
        ''' 输入：articleID, 获取article对应category下所有的文本向量的平均，加权组合。
        :param articleID:
        :return:
        '''
        Enhanced_TEsArticle2vec = self.get_TEsArticle2vector(articleID)  # 初始化,原始自身的文本向量
        avg_TEsCategoryID2Vector = np.zeros(self.vector_size, dtype=float).reshape(1, -1)
        try:
            categoryIDs_list = self.articleID_categoryIDs_dict[str(articleID)]
            num = 0
            for categoryID in categoryIDs_list:
                TEsCategoryID2Vector = self.get_TEsCategoryID2Vector(categoryID)  #
                
                if np.any(TEsCategoryID2Vector):  # 判断非0向量
                    avg_TEsCategoryID2Vector += TEsCategoryID2Vector
                    # print("TEsCategoryID2Vector")
                    num += 1
            if num >= 1:
                avg_TEsCategoryID2Vector = (avg_TEsCategoryID2Vector / num).reshape(1, -1)
        except:
            pass
        if np.any(avg_TEsCategoryID2Vector):  # 判断非0向量
            Enhanced_TEsArticle2vec = Wa * Enhanced_TEsArticle2vec + (1 - Wa) * avg_TEsCategoryID2Vector  ##
            
        return Enhanced_TEsArticle2vec


    def get_TEsCategoryID2Vector(self, categoryID):  ## 输入：categoryID, 获取该Category下所有Article的平均文本向量
        TEsCategoryID2Vector = np.zeros(self.vector_size, dtype=float).reshape(1, -1)  # 全0向量初始值
        try:
            articleIDs_list = self.categoryID_articleIDs_dict[str(categoryID)]
            num = 0
            for articleID in articleIDs_list:
                TEsArticle2vec = self.get_TEsArticle2vector(articleID)  # func
                
                if np.any(TEsArticle2vec):  # 判断非0向量
                    TEsCategoryID2Vector += TEsArticle2vec
                    num += 1
            if num >= 1:
                TEsCategoryID2Vector = (TEsCategoryID2Vector / num).reshape(1, -1)  # .tolist()
        except:
            pass
        return TEsCategoryID2Vector
        
