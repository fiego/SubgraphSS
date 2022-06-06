
import os
import torch,gc
# import scipy.stats
import pandas as pd
from tqdm import tqdm
from utils.data import compute_pearsonr, MaxMinNormalization
from model.Similarity import GEsSim
#%%
print(os.getcwd())
gc.collect()
torch.cuda.empty_cache()

#%% GEsSim
def run_GEsSim_Article(GEs, Wa, benchmark_path, benchmark_result_path):
    benchmark_df = pd.read_csv(benchmark_path, header=0, keep_default_na=False) #

    sim_GEsArt_list = []
    for i, row in benchmark_df.iterrows():
        aID1 = str(row["articleID1"])
        aID2 = str(row["articleID2"])

        if len(aID1) > 0 and len(aID2) > 0:
            sim_GEsArticle = GEs.sim_GEsArticle2Vector(Wa, aID1, aID2)
            sim_GEsArt_list.append(sim_GEsArticle)
        else:
            sim_GEsArticle = 0.0
            sim_GEsArt_list.append(sim_GEsArticle)

    benchmark_df[GEs.mode_name] = sim_GEsArt_list  ##
    benchmark_df.to_csv(benchmark_result_path, index=None)  # 生成的结果文件

#%% 选择数据集
if __name__ == '__main__':
    # 模型名字
    # mode_name_list = ["Node2Vec"]
    mode_name_list = ["DeepWalk", "GLEE", "NetMF", "Node2Vec", "RandNE"]   ## 设置 GEs_name , GLEE 效果差; NetMF 效果差;

    # 选择数据集
    # benchmarks = ["WS353"]  # RG65 MC30 WS353 WS203Sim SimLex666
    benchmarks = ["RG65", "MC30", "WS353", "WS203Sim", "SimLex666"]
    
    for mode_name in tqdm(mode_name_list):
        Wa_list = []
        df = pd.DataFrame()
        GEs = GEsSim(mode_name)
        GEs.mode_name = f"{GEs.mode_name}1s"  # 1s 2s
        for benchmark in benchmarks:
            Wa_list = []
            PCC_list = []
            for Wa in [x/10 for x in list(range(0, 11, 1))]:
                benchmark_path = f"./data/Word_Similarity_Dataset/{benchmark}/{benchmark}_transform.csv"
                result_path = f"./results/{benchmark}/{GEs.mode_name}_{Wa}_{benchmark}_1215.csv"  ## 数据集保存的路径
                print(f"{GEs.mode_name}, {benchmark}, Wa={Wa}")

                run_GEsSim_Article(GEs, Wa, benchmark_path, result_path)
                Wa, PCC = compute_pearsonr(GEs.mode_name, Wa, benchmark_path, result_path)  # 打印结果

                Wa_list.append(Wa)
                PCC_list.append(PCC)

            df[benchmark] = PCC_list
        df["Wa"] = Wa_list

        save_path = f"./results/PCC/PCC_{GEs.mode_name}_1215.csv"  #
        df.to_csv(save_path, index=None, encoding="utf-8")
        

