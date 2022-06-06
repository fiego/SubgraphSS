
import numpy as np
import scipy.stats
import pandas as pd
from utils.data import MaxMinNormalization

#%% generate_mixed_optimal_PCC
def generate_mixed_optimal_PCC():
    TEs_list = ["fastText", "DMPV"]        # "fastText", "DMPV" "DBOW", "Word2Vec"
    KGEs_list = ["PairRE", "TransR"]       # "TransE"
    GEs_list = ["RandNE2s", "Node2Vec2s"]  # "GLEE", "NetMF", "DeepWalk"
    benchmarks = ["RG65", "MC30", "WS353", "WS203Sim", "SimLex666"]

    for TEs in TEs_list:
        for KGEs in KGEs_list:
            for GEs in GEs_list:
                mixedPCC_df = pd.DataFrame()
                for benchmark in benchmarks:
                    benchmark_path = f"./data/Word_Similarity_Dataset/{benchmark}/{benchmark}_transform.csv"
                    benchmark_our_path = f"./results/PCC/Optimal/Optimal_{benchmark}_1215.csv"
        
                    benchmark_df = pd.read_csv(benchmark_path, header=0, keep_default_na=False)  #
                    benchmark_our_df = pd.read_csv(benchmark_our_path, header=0, keep_default_na=False)  #
                    
                    benchmark_our_df["YesNo"] = benchmark_df["YesNo"]
                    benchmark_our_df = benchmark_our_df.loc[(benchmark_our_df.YesNo == "Y")]  # 筛选出benchmark中 YesNo="Y"
                    print(f"Num: {len(benchmark_df)}; {benchmark_our_path}")
    
                    score = benchmark_our_df["Score"].astype(float)
                    TEsSim_list = MaxMinNormalization(benchmark_our_df[TEs].astype(float))    # 归一化
                    KGEsSim_list = MaxMinNormalization(benchmark_our_df[KGEs].astype(float))
                    GEsSim_list = MaxMinNormalization(benchmark_our_df[GEs].astype(float))
                    
                    Wa_list, Wb_list, Wc_list, mixed_PCC_list = [], [], [], []
                    for Wa in [x for x in list(range(1, 10, 1))]:
                        for Wb in [x for x in list(range(1, 10, 1))]:
                            for Wc in [x for x in list(range(1, 10, 1))]:
                                if Wa + Wb + Wc == 10:
                                    mixed_score = [Wa*i/10 + Wb*j/10 + Wc*k/10 for i,j,k in zip(TEsSim_list, KGEsSim_list, GEsSim_list)]
                                   
                                    Wa_list.append(Wa/10)
                                    Wb_list.append(Wb/10)
                                    Wc_list.append(Wc/10)
                                    mixed_PCC_list.append(scipy.stats.pearsonr(score, mixed_score)[0])
                                   
                    mixed_score_mean = np.mean([TEsSim_list, KGEsSim_list, GEsSim_list], axis=0).tolist()  # 是否还需要再次归一化？
                    mixed_score_max = np.max([TEsSim_list, KGEsSim_list, GEsSim_list], axis=0).tolist()  # 是否还需要再次归一化？
                    
                    Wa_list.append(0)
                    Wb_list.append(0)
                    Wc_list.append(0)
                    mixed_PCC_list.append(scipy.stats.pearsonr(score, mixed_score_mean)[0])
                    
                    Wa_list.append(1)
                    Wb_list.append(1)
                    Wc_list.append(1)
                    mixed_PCC_list.append(scipy.stats.pearsonr(score, mixed_score_max)[0])

                    mixedPCC_df["Wa"] = Wa_list  #
                    mixedPCC_df["Wb"] = Wb_list
                    mixedPCC_df["Wc"] = Wc_list
                    mixedPCC_df[benchmark] = mixed_PCC_list
                    
                mixed_path = f"./results/Mixed/{TEs}_{KGEs}_{GEs}_1215.csv"
                mixedPCC_df.to_csv(mixed_path, index=False, encoding="utf-8")  # 导入index
                
#%%
if __name__ == '__main__':
    # 选择数据集
    generate_mixed_optimal_PCC()
    