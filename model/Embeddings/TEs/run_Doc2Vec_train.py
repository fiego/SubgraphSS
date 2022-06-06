
import gensim
from model.Doc2Vec import Doc2Vec

if __name__ == '__main__':

    doc2vec = Doc2Vec()  ## 加载模型
    mode = doc2vec.train()

# %% laod
#     TEs_model = gensim.models.Doc2Vec.load("./checkpoint/Doc2Vec/SCNU_simple_0630//DBOW_300d_200e.model")

    # TEs_model = gensim.models.Doc2Vec.load("./checkpoint/Doc2Vec/DP_DBOW_300d_200e.model")

