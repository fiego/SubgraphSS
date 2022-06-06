
import os
import torch,gc
import scipy.stats
import pandas as pd
# from tqdm import tqdm
from model.Similarity import GEsSim

#%%
print(os.getcwd())
gc.collect()
torch.cuda.empty_cache()

#%%
def read_optimal_setting(mode_name, benchmark):
    optimal_path = f"./results/PCC/PCC_{mode_name}_1215.csv"
    optimal_df = pd.read_csv(optimal_path, header=0, usecols=[f'{benchmark}', 'Wa'])
    optimal_df = optimal_df[(optimal_df['Wa'] >= 0) & (optimal_df['Wa'] < 1)]  # Wa选择范围设置
    maxValue_df = optimal_df[optimal_df[benchmark] == optimal_df[benchmark].max()]  # 挑选最优的PCC
    optimal_PCC = maxValue_df[benchmark].values[0]
    optimal_Wa = maxValue_df["Wa"].values[0]
    return optimal_Wa, optimal_PCC

#%% 选择数据集
if __name__ == '__main__':
    # 模型名字
    TEs_list = ["Word2Vec", "fastText", "DMPV", "DBOW"]
    KGEs_list = ["PairRE", "TransR", "TransE"]
    GEs_list1 = ["RandNE1s", "Node2Vec1s", "GLEE1s", "NetMF1s", "DeepWalk1s"]  #
    GEs_list2 = ["RandNE2s", "Node2Vec2s", "GLEE2s", "NetMF2s", "DeepWalk2s"]  #
    
    mode_name_list = TEs_list + KGEs_list + GEs_list1 + GEs_list2

    # 选择数据集
    benchmarks = ["RG65", "MC30", "WS353", "WS203Sim", "SimLex666"]

    for benchmark in benchmarks:
        benchmark_path = f"./data/Word_Similarity_Dataset/{benchmark}/{benchmark}_transform.csv"
        save_optimal_path = f"./results/PCC/Optimal/Optimal_{benchmark}_1215.csv"  ## 数据集保存的路径
        benchmark_result_df = pd.read_csv(benchmark_path, header=0, keep_default_na=False)
        
        for mode_name in mode_name_list:
            optimal_Wa, optimal_PCC = read_optimal_setting(mode_name, benchmark)
            
            optimal_result_path = f"./results/{benchmark}/{mode_name}_{optimal_Wa}_{benchmark}_1215.csv"
            
            optimal_result_df = pd.read_csv(optimal_result_path, header=0, keep_default_na=False)
            
            benchmark_result_df[mode_name] = optimal_result_df.iloc[:, -1]  # 每个模型结果最后一列拼接到benchmark_result_df
            
        benchmark_result_df.to_csv(save_optimal_path, index=None, encoding="utf-8")  # 获取到所有每个数据集所有模型的相似度
            
