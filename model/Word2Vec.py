
import json
import numpy as np
import gensim.models
from utils import data
from utils import preprocessing

#%%
class Word2Vec(object):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.mode_name = "Word2Vec"

        self.TEs_model = gensim.models.Word2Vec.load("./model/Embeddings/TEs/trained_model/Word2Vec/Word2Vec_300d_200e.model")   ## fastText
        self.vector_size = self.TEs_model.vector_size

        self.articleID_abstract_dict = data.load_articleID_abstract_dict('./data/DBpedia/long_abstracts_check.csv')
        # self.articleID_abstract_dict = utils.load_dict_from_pickle('./data/DBpedia/articleID_abstract.pickle')
        # self.mappingbased_articleID_articleIDs_dict = json.load(open(f"./data/DBpedia/mappingbased_articleID_articleIDs.json", "r"))

# %% get_TextEmbedding
    def get_TEsArticle2vector(self, articleID):
        '''Word2Vec'''
        try:
            doc_abstract = self.articleID_abstract_dict[str(articleID)]
            # print(f"articleID: {articleID},  doc_abstract: {doc_abstract}")
            article2vec = self.get_sentence_vector(doc_abstract)  ##
        except:
            article2vec = np.zeros(self.vector_size, dtype=float)
            # print(f'{self.TEs_name}, No articleID: {articleID}')
        return article2vec.reshape(1, -1)

    def get_sentence_vector(self, text):
        sentence_vector = np.zeros(self.vector_size, dtype=float)
        word_list = preprocessing.word_tokenize(text)  # 分词
        num = 0
        for word in word_list:
            try:
                sentence_vector += self.TEs_model.wv.get_vector(word) ## Word2Vec.wv[key]== Word2Vec.wv.get_vector(key)
                num += 1
            except:
                pass
        if num >= 1:
            # sentence_vector = sentence_vector / len(word_list)
            sentence_vector = sentence_vector / num
        return sentence_vector
