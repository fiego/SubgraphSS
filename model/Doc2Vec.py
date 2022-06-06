
import numpy as np
import gensim.models

#%%
class Doc2Vec(object):
    def __init__(self):
        super(Doc2Vec, self).__init__()
        self.mode_name = "DMPV"  # DMPV DBOW # DBOW效果更好
        # self.TEs_model = gensim.models.Doc2Vec.load(f"./model/Embeddings/TEs/trained_model/Doc2Vec/tokenize/{self.TEs_name}_512d_200e.model")
        self.TEs_model = gensim.models.Doc2Vec.load(f"./model/Embeddings/TEs/trained_model/Doc2Vec/{self.mode_name}_300d_200e.model")  #### Doc2Vec
        # self.TEs_model = gensim.models.Doc2Vec.load("./mode_data/Doc2Vec/DBOW_300d_200e.model")  ## Doc2Vec
        self.vector_size = self.TEs_model.vector_size

#%% get_TextEmbedding
    def get_TEsArticle2vector(self, articleID):
        '''Doc2Vec'''
        try:
            article2vec = self.TEs_model.dv.get_vector(str(articleID))  ##
        except:
            article2vec = np.zeros(self.vector_size, dtype=float)
            # print(f'{self.TEs_name}, No articleID: {articleID}')
        return article2vec.reshape(1, -1)

