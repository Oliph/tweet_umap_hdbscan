"""
Config file for clusters creation and search
"""
from itertools import chain

folder_name = "01-01-20_31-12-20"
# folder_name = "data"
data_filename = "raw_data.json"
corpus_filename = "corpus.txt"
preprocess_filename = "preprocess_df.ftr"
sent_emb_filename = "sent_emb.npy"
grid_search_results = "grid_search_hdbscan.csv"
num_cpu = 5

bert_model_name = "distiluse-base-multilingual-cased-v2"

list_n_dim = [2, 5, 10, 25]
list_cluster_size = list(
    chain(
        # range(3, 20, 1),
        range(10, 50, 5),
        range(50, 100, 10),
        range(100, 501, 50),
    )
)
list_min_size = list(
    chain(
        range(1, 5),
        range(25, 50, 5),
        range(50, 100, 10),
        # range(100, 201, 50),
    )
)


umap_n_neighbours = 15
umap_min_dist = 0.1
