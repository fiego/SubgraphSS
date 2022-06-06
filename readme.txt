

The main steps of SubgraphSS are as follows:

1、Data Preparation
DBpedia
we provide a list of files used in DBpedia(http://downloads.dbpedia.org/wiki-archive/downloads-2016-10.html).
Then, we extract data from page_ids_en.ttl, long_abstracts_en.ttl (abstract), mappingbased_objects_en.ttl (mappingbased), skos_categories_en.ttl (skos) and article_categories_en.ttl files.

Datasets(RG65, MC30, WS353, WS203-Sim, and SimLex-999)
Note that some words in the benchmarks do not correspond to concepts in DBpedia.
In that case, we map each benchmark word pair to its standard form. To eliminate the
disambiguation, we use the same approach as studies.
Lastly, RG65, MC30, WS353, WS203-Sim, and SimLex-999 have 53, 26, 274, 171, and 603 pairs of words, respectively.


2、Embedded Vector Model Training
After preprocessing DBpedia data, we use DBpedia data for model training.
Libraries and parameters are listed before presenting the results of all methods.
The libraries used only need to be consistent with the version of model training and model loading.

Abstract Embedding
We use Gensim(https://radimrehurek.com/gensim/) to train all abstracts of concepts in DBpedia, each vector size is set to 300, and other parameters keep the default.
Reference model.Embeddings.TEs.run_fastText_train.py

Structure Embedding
In terms of structure embedding, we used PyKEEN(https://github.com/pykeen/pykeen) to train the notion graph, with the vector size of each concept set to 100
and the other parameters left alone.
Reference model.Embeddings.KGEs.run_KGEs_train.py

Category Embedding
We use Karate Club(https://github.com/benedekrozemberczki/karateclub) for category embedding, each category’s vector size is set to 128, and no other parameters are altered.
Reference model.Embeddings.GEs.run_GEs_train.py


3、Semantic Similarity Calculation

Abstract Similarity
Reference run_TEsSim.py

Structure Similarity
Reference run_KGEsSim.py

Category Similarity
Reference run_GEsSim.py


4、Mixed Computation
Reference run_optimal.py
Reference generate_mixed_PCC_optimal_Weight.py