
import json
import torch
import numpy as np
from pykeen.triples import TriplesFactory
# from model.BaseModel import BaseModel
#%%
class KnowledgeGraphEmbeddings(object):  # object
    def __init__(self, mode_name="PairRE"):  # KGEs_name = "TransE"
        super(KnowledgeGraphEmbeddings, self).__init__()
        self.mode_name = mode_name     # TransE PairRE TransR
        self.training = TriplesFactory.from_path(path=f"./data/DBpedia/KGEs_Corpus/mappingbased_tsv/train.tsv",
                                                 create_inverse_triples=False)  # TriplesFactory
        self.KGEs_model = torch.load(f"./model/Embeddings/KGEs/trained_model/DBpedia_mappingbased_tsv/{self.mode_name}_100d/trained_model.pkl")
        self.mappingbased_articleID_articleIDs_dict = json.load(open(f"./data/DBpedia/mappingbased_articleID_centerArticleIDs.json", "r"))  # 中心
        self.entity_embeddings = self.KGEs_model.entity_representations[0](indices=None).detach().cpu().numpy()
        self.avg_vec = np.mean(self.entity_embeddings, axis=0)
        self.std_vec = np.std(self.entity_embeddings, axis=0, ddof=1)  #
        self.embedding_dim = 100  # 找不到函数，先手动 PairRE:无self.KGEs_model.embedding_dim

#%% get_KnowledgeGraphEmbeddings
    def get_Enhanced_KGEsArticle2Vector(self, Wa, articleID):  ## 输入：articleID
        Enhanced_KGEsArticle2vec = self.get_KGEsArticle2Vector(articleID)  # 初始化
        if np.any(Enhanced_KGEsArticle2vec):    # 判断非0向量
            avg_KGEsArticle2Vector = self.get_average_mappingbased_KGEsArticle2Vector(articleID)
            if np.any(avg_KGEsArticle2Vector):  # 判断非0向量
                Enhanced_KGEsArticle2vec = Wa*Enhanced_KGEsArticle2vec + (1-Wa)*avg_KGEsArticle2Vector  #
        ## axis=0，计算每一列的均值; # axis = 1计算每一行的均值
        std_Enhanced_KGEsArticle2vec = (Enhanced_KGEsArticle2vec.reshape(1, -1) - self.avg_vec.reshape(1, -1)) / self.std_vec.reshape(1, -1)  ##标准化欧氏距离
        return std_Enhanced_KGEsArticle2vec.reshape(1, -1)

    def get_KGEsArticle2Vector(self, articleID):  ## 输入：articleID
        KGEsArticle2vec = np.zeros(self.embedding_dim, dtype=float)  # 全0向量初始值
        try:
            list_idx = self.training.entities_to_ids([str(articleID)])
            # KGEsArticle2vec = self.entity_embeddings[list_idx]
            idx = torch.as_tensor(list_idx, device=self.KGEs_model.device)
            # KGEsArticle2vec = self.entity_embeddings(idx).detach()[0].cpu().numpy()  #
            KGEsArticle2vec = self.KGEs_model.entity_representations[0](idx).detach()[0].cpu().numpy()  # 0828修改
        except:
            pass
            # print(f'KnowledgeGraphEmbedding, No articleID: {articleID}')
        return KGEsArticle2vec.reshape(1, -1)

    def get_average_mappingbased_KGEsArticle2Vector(self, articleID):  ## 输入：articleID  # 0930新增
        avg_KGEsArticle2Vector = np.zeros(self.embedding_dim, dtype=float).reshape(1, -1)  # 全0向量初始值
        try:
            articleIDs_list = self.mappingbased_articleID_articleIDs_dict[str(articleID)]
            num = 0
            for aID in articleIDs_list:
                KGEsArticle2vec = self.get_KGEsArticle2Vector(aID)  # func
                if np.any(KGEsArticle2vec):  # 判断非0向量
                    avg_KGEsArticle2Vector += KGEsArticle2vec
                    num += 1
            if num >= 1:
                avg_KGEsArticle2Vector = (avg_KGEsArticle2Vector / num).reshape(1, -1)  #.tolist()
        except:
            pass
        return avg_KGEsArticle2Vector

