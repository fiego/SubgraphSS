
'''
使用Python中的NLTK和spaCy删除停用词与文本标准化: https://blog.csdn.net/fendouaini/article/details/100645088
NLP入门-- 文本预处理Pre-processing: https://zhuanlan.zhihu.com/p/53277723
'''

import re
import inflect
import unicodedata
import nltk.corpus         #停用词
import nltk.tokenize     #分词
import string
import nltk.stem
# from spacy.lang.en import English
# from spacy.lang.en.stop_words import STOP_WORDS  # 使用spacy去除停用词
import gensim.parsing.preprocessing  # 使用Gensim去除停用词,直接在未分词原始文本上进行
# %%

def word_tokenize(text):
    ''' Return a tokenized copy of *text*, using NLTK's recommended word tokenizer.
    '''
    # text = gensim.parsing.preprocessing.remove_stopwords(text)  # 直接在原文本去除停用词，停用词较多，速度较慢，不采用
    word_tokens = gensim.utils.simple_preprocess(doc=text)        # 分词速度较快
    # word_tokens = nltk.tokenize.word_tokenize(text.lower(), language="english")  # sent_tokenize(text) 按句子分割；word_tokenize(sentence) 分词; WhitespaceTokenizer(), 空格符号分割
    # word_tokens = judge_pure_english(word_tokens)     # 判断是否是英文
    word_tokens = remove_stopwords(word_tokens)       # 去除停用词
    # word_tokens = stemming(word_tokens)             # 词干提取 – Stemming
    return word_tokens


def normalize(word_tokens):
    word_tokens = remove_non_ascii(word_tokens)
    word_tokens = remove_punctuation(word_tokens)
    word_tokens = replace_numbers(word_tokens)
    word_tokens = remove_stopwords(word_tokens)
    # word_tokens = stemming(word_tokens)            # 词干提取 – Stemming
    # word_tokens = lemmatisation(word_tokens)       # 词形还原 – Lemmatisation
    return word_tokens

def is_all_eng(strs):
    ''' 判断字符串是否是纯英文
    '''
    for i in strs:
        if i not in string.ascii_letters:  # ascii_letters = ascii_lowercase + ascii_uppercase
            return False
    return True


def judge_pure_english(word_tokens):
    '''  判断word_list里面是否存在非英文元素
    '''
    words = []
    for word in word_tokens:
        if is_all_eng(word):
            words.append(word)
    return words


STOPWORDS = set(nltk.corpus.stopwords.words('english'))   # nltk.corpus.stopwords: 179
STOPWORDS.update(gensim.parsing.preprocessing.STOPWORDS)  # STOPWORDS(frozenset): 337

def remove_stopwords(word_tokens):
    ''' 去除停用词
    '''
    # STOPWORDS = get_stopwords('./data/dict/stopwords_EN.txt')  # 停用词列表，根据需要手动增添
    # print('stop_words:', stop_words)
    filtered_word_tokens = []
    for word in word_tokens:
        if word not in STOPWORDS:
            filtered_word_tokens.append(word)
    # print('filtered_word_tokens:', filtered_word_tokens)
    return filtered_word_tokens

def get_stopwords(path):
    ''' 获取停用词列表
    '''
    stopwords = [x.strip() for x in open(path, encoding='utf8').readlines()]
    return stopwords


def remove_punctuation(word_tokens):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in word_tokens:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(word_tokens):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in word_tokens:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_non_ascii(word_tokens):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in word_tokens:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def stemming(word_tokens):
    '''  词干提取 – Stemming
         词干提取是去除单词的前后缀得到词根的过程。
    '''
    ps = nltk.stem.porter.PorterStemmer()
    stem_words = [ps.stem(word) for word in word_tokens]
    # print('stem_words:', stem_words)
    return stem_words


def lemmatisation(word_tokens):
    ''' 词形还原 – Lemmatisation
        词形还原是基于词典，将单词的复杂形态转变成最基础的形态。
    '''
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    # pos是词性: ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
    lemma_words = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in word_tokens]
    print('lemma_words:', lemma_words)
    return lemma_words

