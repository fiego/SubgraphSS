

import csv
from utils import utils
import utils.utils.preprocessing

def write_csv_to_txt(save_path, csv_path):
    ''' 。
    '''
    # Load data
    csv_df = utils.read.read_large_csv(csv_path)
    ## 根据实际的数据修改列
    abstracts = csv_df.iloc[:, 1].astype(str)  # 取第2列, text/abstract
    fp = open(save_path, 'w', encoding='utf-8')     # 写入 TXT
    for text in abstracts:
        fp.write(text + '\n')
    fp.close()


def write_doc_segment(save_path, csv_path):
    ''' 把分词后word_tokens写入文件。
    '''
    # Load data
    csv_df = utils.read.read_large_csv(csv_path)  # _check
    ## 根据实际的数据修改列
    ids = csv_df.iloc[:, 0].astype(int)        # 取第0列, id
    pages = csv_df.iloc[:, 1].astype(str)      # 取第1列, page
    abstracts = csv_df.iloc[:, 2].astype(str)  # 取第2列, text/abstract
    # docs = read_docs(path)

    fp = open(save_path, 'w', encoding='utf-8')     # 写入 TXT
    # fp_writer = csv.writer(fp, delimiter=" ")  # 构建csv写入对象 \t ,
    # fp_writer.writerow(columns_name)           # TXT,不需要表头

    fp_test = open(f'{save_path}.csv', 'w', encoding='utf-8', newline='')  # 写入 CSV，命名为：**.csv
    fp_writer_test = csv.writer(fp_test, delimiter=",")
    fp_writer_test.writerow(['index', 'id', 'page', 'abstract'])   # CSV,写入列表头

    count = 1
    index = 0
    for id,page,text in zip(ids, pages, abstracts):
        word_tokens = utils.preprocessing.word_tokenize(text)
        words = ' '.join(word_tokens)
        # print(words)
        if len(words) >= 3:  # word_tokens 个数
            fp.write(words+'\n')
            item = [index, id, page, words]
            fp_writer_test.writerow(item)
            index += 1  # 编新的序号
        else:
            print(f"{count}: word_tokens in abstract_text is null. id:{id}, {text}")
            count += 1
    fp.close()
    fp_test.close()


