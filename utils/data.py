
import csv
import gensim
import scipy.stats
import pandas as pd
from itertools import islice
import utils.preprocessing as preprocessing
from gensim.models.doc2vec import TaggedLineDocument
#%%
csv.field_size_limit(13107200)  # _csv.Error: field larger than field limit (131072)
def read_corpus(path, tokens_only=False):
    with open(path, mode='r', encoding='utf8') as fp:
        reader = csv.reader(fp)
        # print(type(reader))
        for row in islice(reader, 1, None):  # 跳开行首
            tag = str(row[0])  # 0列 tag:id <class 'str'>
            doc = row[2]  # 1列 doc:abstract <class 'str'> 根据实际修改
            # words = doc.split(" ")  # 使用 tokenize_abstracts.csv 的时候打开（使用空格分词）
            words = preprocessing.word_tokenize(doc)  ## 自定义的分词
            # words = gensim.utils.simple_preprocess(doc)  ## 使用simple_preprocess进行预处理，切词
            if tokens_only:
                yield words
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(words, [tag])

# train_corpus = list(read_corpus(lee_train_file))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
#%%

def read_large_csv(path):
    ## 读取大的csv文件
    fp = open(file=path, mode='r')  ## 用open打开
    df_chunk = pd.read_csv(path, chunksize=10000, low_memory=False)  # 结合 pd.read_csv()
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    csv_df = pd.concat(res_chunk)
    fp.close()  # close
    return csv_df

def load_articleID_abstract_dict(path):
    csv_df = pd.read_csv(path, usecols=["id", "abstract"], sep=",",
                               error_bad_lines=False, header=0, low_memory=False)
    ## 根据实际的数据修改列
    ids = csv_df.iloc[:, 0].astype(str)         # 取第0列, ID
    # pages = csv_df.iloc[:, 1].astype(str)     # 取第1列, page
    abstracts = csv_df.iloc[:, 1].astype(str)   # 取第2列, text/abstract  0827修改
    articleID_abstract_dict = dict(zip(ids, abstracts))
    return articleID_abstract_dict

#%% 相同 10.0； 错误 0.0
def get_MaxMinList(data_list):
    data_list = list(set(data_list))
    try:  # 删除特定值
        data_list.remove(10.0)   ## 删除指定的值
    except:
        pass

    try:
        data_list.remove(0.0)    ## 删除指定的值
    except:
        pass
    Max = max(data_list) + 0.001
    Min = min(data_list) - 0.001
    return Max, Min

def replace_MaxMinValueList(data_list):
    set_value = 10.0  # 指定的值（相同概念的值）
    new_data_list = []
    xMax, xMin = get_MaxMinList(data_list)  ##
    for i in data_list:
        if i == set_value:
            new_data_list.append(xMax)  # set_value
        else:
            new_data_list.append(i)
    return new_data_list

def MaxMinNormalization(data_list):
    xMax, xMin = get_MaxMinList(data_list)  ##
    xMax = xMax + 0.001
    xMin = xMin - 0.001
    data_list = replace_MaxMinValueList(data_list)  ## 替换
    new_data_list = []
    for x in data_list:
        if x == 0.0:
            x = 0.0
        else:
            x = (x - xMin) / (xMax - xMin)  # 归一化
        new_data_list.append(x)
    return new_data_list

# %% compute_pearsonr
def compute_pearsonr(mode_name, Wa, benchmark_path, result_path):
    benchmark_df = pd.read_csv(benchmark_path, header=0, keep_default_na=False)  #
    result_df = pd.read_csv(result_path, header=0, keep_default_na=False)  #
    # print(f"Num: {len(benchmark_df)}; {benchmark_path}")
    # result_df["YesNo"] = benchmark_df["YesNo"]
    # result_df = result_df.loc[(result_df.YesNo == "Y")]  # 筛选出benchmark中 YesNo="Y"
    
    score0, score1 = [], []
    for i, row in result_df.iterrows():
        s0 = row["Score"]
        s1 = row[mode_name]  #
        if s1 != 0.0:
            score0.append(s0)
            score1.append(s1)
    
    score1 = MaxMinNormalization(score1)  ## 计算pearsonr时做归一化 1212修改
    
    pearsonr = scipy.stats.pearsonr(score0, score1)
    print(f'{mode_name} pearsonr: {pearsonr}')
    # spearmanr = scipy.stats.spearmanr(score0, score1)
    # print(f'{mode_name} spearmanr: {spearmanr}')
    
    print(f"##: {mode_name}, {Wa}, {pearsonr[0]}")
    return Wa, pearsonr[0]