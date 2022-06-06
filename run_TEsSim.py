
import os
import torch,gc
import pandas as pd
from tqdm import tqdm
from utils.data import compute_pearsonr, MaxMinNormalization
from model.Similarity import TEsSim
#%%
print(os.getcwd())
gc.collect()
torch.cuda.empty_cache()

#%% TEsSim
def run_TEsSim_Article(TEs, Wa, benchmark_path, benchmark_result_path):
    benchmark_df = pd.read_csv(benchmark_path, header=0, keep_default_na=False) #

    sim_TEsArticle_list = []
    for i, row in benchmark_df.iterrows():
        aID1 = str(row["articleID1"])
        aID2 = str(row["articleID2"])

        if len(aID1) > 0 and len(aID2) > 0:  # 非空
            sim_TEsArticle = TEs.sim_TEsArticle2Vector(Wa, aID1, aID2)  # 计算相似度
            sim_TEsArticle_list.append(sim_TEsArticle)
        else:
            sim_TEsArticle = 0.0
            sim_TEsArticle_list.append(sim_TEsArticle)

    benchmark_df[TEs.mode_name] = sim_TEsArticle_list
    benchmark_df.to_csv(benchmark_result_path, index=None)  # 生成的结果文件    # print(benchmark_df)

if __name__ == '__main__':
    # 选择数据集
    benchmarks = ["WS353"]  # RG65 MC30 WS353 WS203Sim SimLex666
    # benchmarks = ["RG65", "MC30", "WS353", "WS203Sim", "SimLex666"]
    
    Wa_list = []
    df = pd.DataFrame()
    TEs = TEsSim()  # 类实例化
    for benchmark in tqdm(benchmarks):
        Wa_list = []
        PCC_list = []
        for Wa in [x/10 for x in list(range(0, 11, 1))]:
            benchmark_path = f"./data/Word_Similarity_Dataset/{benchmark}/{benchmark}_transform.csv"
            result_path = f"./results/{benchmark}/{TEs.mode_name}_{Wa}_{benchmark}_1215.csv"  ## 数据集保存的路径
            print(f"{TEs.mode_name}, {benchmark}, Wa={Wa}")

            run_TEsSim_Article(TEs, Wa, benchmark_path, result_path)
            Wa, PCC = compute_pearsonr(TEs.mode_name, Wa, benchmark_path, result_path)  # 打印结果
            
            Wa_list.append(Wa)
            PCC_list.append(PCC)

        df[benchmark] = PCC_list
    df["Wa"] = Wa_list
    
    save_PCC_path = f"./results/PCC/PCC_{TEs.mode_name}_1215.csv"
    df.to_csv(save_PCC_path, index=None, encoding="utf-8")
    
