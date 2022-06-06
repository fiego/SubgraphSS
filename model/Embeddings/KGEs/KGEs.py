
# https://github.com/pykeen/pykeen
# https://pykeen.readthedocs.io/en/stable/tutorial/first_steps.html
# https://pykeen.readthedocs.io/en/latest/tutorial/running_hpo.html

import logging
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
logging.basicConfig(level=logging.DEBUG)

#%% Parameter Setting  已训练 dim=100, batch_size=128
dim = 100        # 100
epochs = 100
batch_size = 64  # 128 64

#%% Dataset
corpus = "DBpedia"        # DBpedia DP_DBpedia
triple_tsv = "mappingbased_tsv"  # mappingbased_tsv skos_tsv Nations

## KGEs
TRAIN_PATH = f"./data/DBpedia/KGEs_Corpus/mappingbased_tsv/train.tsv"
TEST_PATH = f"./data/DBpedia/KGEs_Corpus/mappingbased_tsv/test.tsv"
VALIDATE_PATH = f"./data/DBpedia/KGEs_Corpus/mappingbased_tsv/valid.tsv"

## Nations
# TRAIN_PATH = f"./data/DBpedia/KGEs_Corpus/Nations/train.tsv"
# TEST_PATH = f"./data/DBpedia/KGEs_Corpus/Nations/test.tsv"
# VALIDATE_PATH = f"./data/DBpedia/KGEs_Corpus/Nations/valid.tsv"

#%% TriplesFactory: training, testing, validation
training = TriplesFactory.from_path(
    path=TRAIN_PATH,
    create_inverse_triples=False)

testing = TriplesFactory.from_path(
    path=TEST_PATH,
    create_inverse_triples=False,
    entity_to_id=training.entity_to_id,     ## training
    relation_to_id=training.relation_to_id)

# validation = TriplesFactory.from_path(
#     path=VALIDATE_PATH,
#     create_inverse_triples=False,
#     entity_to_id=training.entity_to_id,     ## training
#     relation_to_id=training.relation_to_id)

#%%
def KnowledgeGraphEmbeddings(model):
    save_path = f"./model/Embeddings/KGEs/trained_model/{corpus}_{triple_tsv}/{model}_{dim}d"
    print(f"save_path:{save_path}")
    pipeline_result = pipeline(
        # dataset='Nations',    # test_dataset
        training=training,      # TRAIN_PATH
        testing=testing,        # TEST_PATH
        # validation=validation,  # VALIDATE_PATH 好像不需要
        model=model,
        model_kwargs=dict(embedding_dim=dim),
        # optimizer_kwargs=dict(lr=0.005),
        training_kwargs=dict(use_tqdm_batch=False,
                             batch_size=batch_size,
                             num_epochs=epochs,
                             # checkpoint_on_failure=True,
                             checkpoint_frequency=66,
                             checkpoint_directory=save_path,
                             checkpoint_name=f"{model}_{dim}d.pt"),
        # evaluation_kwargs=dict(do_time_consuming_checks=False,
        #                        use_tqdm=False),
                               # batch_size=batch_size),
        ## https://pykeen.readthedocs.io/en/stable/reference/stoppers.html?highlight=stopper_kwargs
        # stopper='early',
        # stopper_kwargs=dict(frequency=10, patience=5, relative_delta=0.002),
        random_seed=1688,
        device='gpu'  # cpu
        # automatic_memory_optimization=True
    )
    pipeline_result.save_to_directory(save_path)  ##


#%% test
# if __name__=="__main__":
#     KnowledgeGraphEmbeddings("transE")

