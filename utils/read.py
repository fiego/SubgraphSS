
from itertools import islice
import pandas as pd

def read_large_csv(path):
    ## 读取大的csv文件
    fp = open(file=path, mode='r')  ## 用open打开
    df_chunk = pd.read_csv(path, chunksize=10000)  # 结合 pd.read_csv()
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    csv_df = pd.concat(res_chunk)
    fp.close()  # close
    return csv_df

def read_docs(path):
    # Load data
    csv_df = read_large_csv(path)
    ## 根据实际的数据修改列
    docs = csv_df.iloc[:, 1].astype(str)  # 取第1列, text
    return docs

def read_word_tokens_txt(path):
    fp = open(path, mode='r', encoding='utf-8')
    # csv_reader = csv.reader(fp)  # 使用csv.reader读取csvfile中的文件
    word_tokens_list = []
    for line in islice(fp, 0, None):       # 根据需要是否跳开第一行
        word_tokens = line.strip("\n")   # <class 'list'>   .split(' ')
        # print(word_tokens)
        word_tokens_list.append(word_tokens)
    fp.close()
    return word_tokens_list

def read_txt(path):
    with open(path, mode='r', encoding='utf-8') as fp:
        word_tokens_list = fp.readlines()
        return word_tokens_list

def load_categoryIndex2ID_dict(path):
    # Load data
    csv_df = pd.read_csv(path, usecols=["categoryIndex", "categoryID"],  # categoryPage
                               sep=",", error_bad_lines=False, header=0, low_memory=False)
    ## 根据实际的数据修改列
    categoryIndex = csv_df.iloc[:, 0].astype(str)  # 取第1列, categoryIndex, value
    categoryID = csv_df.iloc[:, 1].astype(str)     # 取第1列, categoryID, key
    categoryIndex2ID_dict = dict(zip(categoryID, categoryIndex))
    return categoryIndex2ID_dict
