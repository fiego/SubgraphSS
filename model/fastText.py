
import fasttext
import json
import numpy as np
from utils import data
from utils import preprocessing

#%%
class fastText(object):
    def __init__(self):
        super(fastText, self).__init__()
        self.mode_name = "fastText"

        self.TEs_model = fasttext.load_model("./model/Embeddings/TEs/trained_model/fastText/fastText_300d_200e.model")   ## fastText
        self.vector_size = 300

        self.articleID_abstract_dict = data.load_articleID_abstract_dict('./data/DBpedia/long_abstracts_check.csv')
        # self.articleID_abstract_dict = utils.load_dict_from_pickle('./data/DBpedia/articleID_abstract.pickle')
        # self.mappingbased_articleID_articleIDs_dict = json.load(open(f"./data/DBpedia/mappingbased_articleID_articleIDs.json", "r"))

# %% get_TextEmbedding
    def get_TEsArticle2vector(self, articleID):
        '''fastText'''
        try:
            doc_abstract = self.articleID_abstract_dict[str(articleID)]
            article2vec = self.get_sentence_vector(doc_abstract)  ## self.get_sentence_vector self.TEs_model.get_sentence_vector
        except:
            article2vec = np.zeros(self.vector_size, dtype=float)
            # print(f'{self.TEs_name}, No articleID: {articleID}')
        return article2vec.reshape(1, -1)

    def get_sentence_vector(self, text):  # self.TEs_model.get_sentence_vector 已有
        sentence_vector = np.zeros(self.vector_size, dtype=float)
        word_list = preprocessing.word_tokenize(text)  # 分词
        num = 0
        for word in word_list:
            try:
                sentence_vector += self.TEs_model.get_word_vector(word) #
                num += 1
            except:
                pass
        if num >= 1:
            sentence_vector = sentence_vector / num  # 所有的词向量求平均
        return sentence_vector
