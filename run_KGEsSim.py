
import os
import torch,gc
import pandas as pd
from tqdm import tqdm
from utils.data import compute_pearsonr, MaxMinNormalization
from model.Similarity import KGEsSim
#%%
print(os.getcwd())
gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use multiple gpus 0,1

#%% KGEsSim
def run_KGEsSim_Article(KGEs, Wa, benchmark_path, benchmark_result_path):
    benchmark_df = pd.read_csv(benchmark_path, header=0, keep_default_na=False) #
    sim_KGEsArt_list = []
    for i, row in benchmark_df.iterrows():
        aID1 = str(row["articleID1"])
        aID2 = str(row["articleID2"])
        if len(aID1) > 0 and len(aID2) > 0:
            sim_KGEsArticle = KGEs.sim_KGEsArticle2Vector(Wa, aID1, aID2)
            sim_KGEsArt_list.append(sim_KGEsArticle)
        else:
            sim_KGEsArticle = 0.0
            sim_KGEsArt_list.append(sim_KGEsArticle)

    benchmark_df[KGEs.mode_name] = sim_KGEsArt_list  #
    benchmark_df.to_csv(benchmark_result_path, index=None)  # 生成的结果文件

#%%
if __name__ == '__main__':
    # 模型名字
    mode_name_list = ["PairRE"]
    # mode_name_list = ["TransE", "PairRE", "TransR"]   ## 设置 GEs_name

    # 选择数据集
    benchmarks = ["WS353"]  # RG65 MC30 WS353 WS203Sim SimLex666
    # benchmarks = ["RG65", "MC30", "WS353", "WS203Sim", "SimLex666"]

    for mode_name in tqdm(mode_name_list):
        Wa_list = []
        df = pd.DataFrame()
        KGEs = KGEsSim(mode_name)  ## 类实例化
        for benchmark in benchmarks:
            Wa_list = []
            PCC_list = []
            for Wa in [x/10 for x in list(range(0, 11, 1))]:
                benchmark_path = f"./data/Word_Similarity_Dataset/{benchmark}/{benchmark}_transform.csv"
                benchmark_result_path = f"./results/{benchmark}/{KGEs.mode_name}_{Wa}_{benchmark}_1215.csv"  ## 数据集保存的路径
                print(f"{KGEs.mode_name}, {benchmark}, Wa={Wa}")

                run_KGEsSim_Article(KGEs, Wa, benchmark_path, benchmark_result_path)
                Wa, PCC = compute_pearsonr(KGEs.mode_name, Wa, benchmark_path, benchmark_result_path)  # 打印结果

                Wa_list.append(Wa)
                PCC_list.append(PCC)
                        
            df[benchmark] = PCC_list
        df["Wa"] = Wa_list
        
        save_path = f"./results/PCC/PCC_{KGEs.mode_name}_1215.csv"
        df.to_csv(save_path, index=None, encoding="utf-8")
        
    