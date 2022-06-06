import pickle
import pandas as pd
import csv
# from model.similarity import Similarity

#%%
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

def save_dict_to_pickle(path, data_dict):
    print('To save the dictionary into a file: ', path)
    # json.dump(data_dict, open(path, mode='w', encoding='utf-8'))
    with open(path, mode='wb') as fp:
        pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_from_pickle(path):
    print('To read data from the file: ', path)
    # data_dict = json.load(open(path, mode='r', encoding='utf-8'))
    with open(path, mode='rb') as fp:
        data_dict = pickle.load(fp)
        return data_dict

def save_dict_to_csv(path, columns_name, data_dict):
    fp = open(path, mode='w', encoding='utf-8', newline='')  # 写入文件
    fp_writer = csv.writer(fp)  # 构建csv写入对象
    fp_writer.writerow(columns_name)  # 写入列表头
    for key, value in data_dict.items():
        item = [key, value]
        fp_writer.writerow(item)
    fp.close()
