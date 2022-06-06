
import pandas as pd
from tqdm import tqdm
from utils.data import compute_pearsonr
# %%
if __name__ == '__main__':
	# 模型名字
	TEs_list = ["Word2Vec", "fastText", "DMPV", "DBOW"]
	KGEs_list = ["PairRE", "TransR", "TransE"]
	GEs_list1 = ["RandNE1s", "Node2Vec1s", "GLEE1s", "NetMF1s", "DeepWalk1s"]  #
	GEs_list2 = ["RandNE2s", "Node2Vec2s", "GLEE2s", "NetMF2s", "DeepWalk2s"]  #
	
	mode_name_list = TEs_list + KGEs_list + GEs_list1 + GEs_list2
	# mode_name_list = ["PairRE", "Node2Vec2s", "fastText"]  # debug
	
	# 选择数据集
	benchmarks = ["RG65", "MC30", "WS353", "WS203Sim", "SimLex666"]
	# benchmarks = ["SimLex666"]
	
	for mode_name in tqdm(mode_name_list):
		Wa_list = []
		df = pd.DataFrame()
		for benchmark in benchmarks:
			Wa_list = []
			PCC_list = []
			for Wa in [x/10 for x in list(range(0, 11, 1))]:
				benchmark_path = f"./data/Word_Similarity_Dataset/{benchmark}/{benchmark}_transform.csv"
				result_path = f"./results/{benchmark}/{mode_name}_{Wa}_{benchmark}_1215.csv"  ## 数据集保存的路径
				print(f"{mode_name}, {benchmark}, Wa={Wa}")
				
				Wa, PCC = compute_pearsonr(mode_name, Wa, benchmark_path, result_path)  # 计算PCC
				
				Wa_list.append(Wa)
				PCC_list.append(PCC)
			
			df[benchmark] = PCC_list
		df["Wa"] = Wa_list
		
		save_path = f"./results/PCC/PCC_{mode_name}_1215.csv"
		df.to_csv(save_path, index=None, encoding="utf-8")
