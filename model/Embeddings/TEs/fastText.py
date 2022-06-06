
'''
https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb
https://fasttext.cc/docs/en/unsupervised-tutorial.html
'''

import fasttext
from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath

import gc
import multiprocessing
from .BaseModel import BaseModel

class ModelConfig(BaseModel):
    """配置参数"""
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.model_name = 'fastText'
        self.model = 'skipgram',  # 'cbow'
        self.vector_size = 300      # 向量维度 300 512
        self.minCount = 20
        # self.minn = 20,
        # self.maxn = 5,
        self.cores = multiprocessing.cpu_count()
        self.learn_rate = 0.001
        self.epochs = 500       # 语料库上的迭代次数(epoch)
        self.window_size = 8    # size of the context window

        self.train_corpus = './data/tokenize_abstracts.txt'                    # input train_corpus
        self.save_path = f"./checkpoint/fastText/{self.model_name}_{self.vector_size}d_{self.epochs}e.model"

        # self.train_corpus = './data/test_data/test_abstracts.txt'       # test_data: input train_corpus
        # self.save_path = "./checkpoint/test_model/test_fastText.model"  # test_model: output save_model

class fastText(ModelConfig):
    def __init__(self):
        super(fastText, self).__init__()
        # self.model_name = Doc2Vec.__name__

    def train(self):
        gc.collect()  ## 清理内存
        print(self.train_corpus)

        print("app started...")
        print(f'num of cores is {self.cores}')

#%% 1 Using fastText
        # In practice, we observe that skipgram models works better with subword information than cbow.
        model = fasttext.train_unsupervised(self.train_corpus,
                model='cbow',    # cbow skipgram
                # minn=self.minn,
                # maxn=self.maxn,
                minCount=self.minCount,
                ws=self.window_size,
                lr=self.learn_rate,
                dim=self.vector_size,
                epoch=self.epochs,
                thread=self.cores)

        model.save_model(self.save_path)  ## 模型保存

        print(model)

#%% 2 Using Gensim's implementation of fastText
        ''' 'num of cores is 56'
        2021-04-22 23:07:03,179 : INFO : FastText lifecycle event {'params': 'FastText(vocab=0, vector_size=300, alpha=0.025)', 'datetime': '2021-04-22T23:07:03.179442', 'gensim': '4.0.1', 'python': '3.8.8 (default, Apr 13 2021, 19:58:26) \n[GCC 7.3.0]', 'platform': 'Linux-5.8.0-43-generic-x86_64-with-glibc2.10', 'event': 'created'}
        '''
'''
        model = FastText(vector_size=self.vector_size)

        # build the vocabulary
        model.build_vocab(corpus_file=self.train_corpus)

        # train the model
        model.train(corpus_file=self.train_corpus,
                    epochs=self.epochs,
                    total_examples=model.corpus_count,
                    total_words=model.corpus_total_words,)

        print(model)
'''
#%%





