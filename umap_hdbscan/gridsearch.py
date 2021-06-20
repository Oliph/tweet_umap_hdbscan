#!/usr/bin/env python
# coding: utf-8

__author__ = "Olivier R Philippe"
"""
Grid search implementation for HDBSCAN.
Perform clustering using different min_cluster_size and mim_sample and number of dimension using UMAP

It outputs the results in a csv file that can be used to rerun this script and skip the already run clusters.
Used the relative_validity_ measure to select the best cluster methods
Take advantage of the cached implementation to play with the cluster size and use multiprocessing to spam several
clustering at the same time
"""

# General import
import os
import re
import csv
import time
import shutil

# import pickle
import logging
import multiprocessing

from pathlib import Path
from itertools import chain
from collections import Counter

import torch
import joblib

# Data imports
import umap
import hdbscan
import numpy as np

import config_cluster

# import umap.umap_ as umap
from tqdm import tqdm
from numpy import load, save
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def check_cuda():
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        logging.info("There are %d GPU(s) available." % torch.cuda.device_count())

        logging.info("We will use the GPU:", torch.cuda.get_device_name(0))

    else:
        logging.info("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def create_sent_embeddings(bert_model_name, sent_emb_path, corpus_path):

    model = SentenceTransformer(bert_model_name)
    device = check_cuda()

    with open(corpus_path) as f:
        corpus = [line.rstrip() for line in f]

    corpus_embeddings = model.encode(
        corpus, convert_to_tensor=True, show_progress_bar=True
    )

    sent_emb = np.array(corpus_embeddings.cpu())

    np.save(sent_emb_path, sent_emb, allow_pickle=False)
    return sent_emb


def write_report(queue, grid_search_results_path, csv_columns):
    """ """
    report_file = Path(grid_search_results_path)
    if report_file.is_file():
        pass
    else:

        with open(grid_search_results_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
    while True:
        report = queue.get()
        if report == "END":
            break
        with open(grid_search_results_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(report)
        f.close()


def get_stats_cluster(data, cluster, relative_validity=True):
    """ """
    # Get the total
    labels = cluster.labels_
    total = len(labels)
    if relative_validity is True:
        validity_index_score = cluster.relative_validity_
    else:
        validity_index_score = hdbscan.validity.validity_index(
            X=cluster._raw_data.astype("double"), labels=cluster.labels_
        )

    # Get the string to be sure to remove the label '-1' and not the position index -1
    count_labels = Counter([str(x) for x in labels])
    # Get the noise
    noise = count_labels["-1"]
    # Remove the noise to be able to get the others infos
    del count_labels["-1"]
    # Top 10
    top_10 = sum([x[1] for x in count_labels.most_common(10)])
    # First cluster
    top_1 = [x[1] for x in count_labels.most_common(1)][0]
    # Get the number of clusters
    n_clusters = len(list(count_labels))

    return {
        "validity_index_score": validity_index_score,
        "noise": noise,
        "top_10": top_10,
        "top_1": top_1,
        "n_clusters": n_clusters,
        "total": total,
    }


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def create_umap_reduction(
    data,
    dim_red_path,
    n_dim,
    umap_n_neighbours,
    umap_min_dist,
    low_memory=True,
    record=True,
):

    model_dim_red_path = "{}/umap_{}.pkl".format(dim_red_path, n_dim)
    try:
        logging.info("Try to load existing model to reduce in {} dim".format(n_dim))
        model_dim_red = joblib.load(
            open(
                model_dim_red_path,
                "rb",
            )
        )
        logging.info("Success in loading model")
    except FileNotFoundError:
        logging.info("No model existing, fitting a new one")

        model_dim_red = umap.UMAP(
            low_memory=low_memory,
            n_neighbors=umap_n_neighbours,
            min_dist=umap_min_dist,
            n_components=n_dim,
            # metric="cosine",
            verbose=True,
        ).fit(data)
        logging.info("New model fitted, record it")

        if record is True:
            joblib.dump(
                model_dim_red,
                open(
                    model_dim_red_path,
                    "wb",
                ),
            )
    logging.info("Reducing the sent_emb to {} dim".format(n_dim))
    return model_dim_red.transform(data)


def getting_hdscan(
    cluster_path,
    data,
    n_dim,
    list_cluster_size,
    min_sample,
    queue_report=False,
    record=False,
    return_model=False,
    **kwargs_hdbscan
):

    start_time = time.time()

    memory_filename = os.path.join(
        cluster_path, "cache", "{}-{}".format(min_sample, n_dim)
    )
    # In case only one cluster size is given, transform it to a list
    if not isinstance(list_cluster_size, list):
        list_cluster_size = [int(list_cluster_size)]
    min_sample = int(min_sample)
    for cluster_size in list_cluster_size:

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cluster_size,
            min_samples=min_sample,
            gen_min_span_tree=True,
            allow_single_cluster=True,
            # metric="cosine",
            memory=memory_filename,
            **kwargs_hdbscan
        )

        try:
            clusterer.fit(data)
            dict_report = get_stats_cluster(data, clusterer)
            dict_report["min_cluster_size"] = cluster_size
            dict_report["min_sample"] = min_sample
            dict_report["n_dim"] = n_dim
            assert set(dict_report.keys()) == set(csv_columns)
            if queue_report is not False:
                queue_report.put(dict_report)

            if record is not False:
                if isinstance(record, bool):
                    filename = os.path.join(
                        cluster_path,
                        "_cluster-size_{}_min-samples_{}_n-dim_{}.pkl".format(
                            cluster_size,
                            min_sample,
                            n_dim,
                        ),
                    )
                else:
                    filename = os.path.join(cluster_path, record)
                joblib.dump(
                    clusterer,
                    open(filename, "wb"),
                )

        except AssertionError:
            logging.error(
                "The columns from the report are not matching the columns from the csv file"
            )
            raise
        except Exception as e:
            logging.error("Exception raised : {}".format(e))
            raise

        end_time = time.time()
    logging.info(
        "Finished to fit in {:.2f} : Mim-sample {} - n-dim {}".format(
            end_time - start_time, min_sample, n_dim
        )
    )

    # logging.info("Cleaning the cache folders")
    shutil.rmtree(os.path.join(memory_filename))

    if return_model is True:
        return clusterer


def create_clusters_to_do(list_n_dim, list_min_size, list_cluster_size):
    """
    Create a dictionary containing all the different clusters to do from the lists
    of dimensions, min_size, cluster_size.
    The dimension being the first key, it allows to process all the test with that UMAP dimension
    reduction and not having several UMAP loaded on the same time
    It is important to have the cluster_size as the last key from the embedded dictionary
    to be able to use the cache option given by HDBSCAN to speed up the process.

    :params:
        list_n_dim list(): list of different dimension to test
        list_min_size list(): list of all the min_size to test
        list_cluster_size list(): list of all the cluster size to test

    :return:
        return_dict dict(): dictionary containing all the combination
    """

    return_dict = dict()
    for n_dim in list_n_dim:
        for min_sample in list_min_size:
            for min_cluster_size in list_cluster_size:
                return_dict.setdefault(int(n_dim), {}).setdefault(
                    int(min_sample), []
                ).append(int(min_cluster_size))
    return return_dict


def get_already_done(grid_search_results_path, all_clusters_to_do):
    """
    Use the recorded details from the grid_search_results_path to avoid rerun the same UMAP-HDBSCAN.
    Helps to avoid rerun several times the same models

    :params:
        grid_search_results_path str(): full path of the grid search results score file

        all_clusters_to_do dict(): dictionary containing the different combination of n_dim, min_sample
            and cluster size to be run

    :return:
        to_return dict(): the same dictionary passed in params but without the combination of n_dim, min_sample
            and cluster size to be run if no file is found
    """
    try:

        to_return = dict()

        with open(grid_search_results_path, "r") as f:
            csv_reader = csv.reader(f)
            # Get the list of rows, skipping the headers
            # The order of the values is within the headers_variable
            next(csv_reader)  # To avoid header
            for n_dim, min_sample, cluster_size, *_ in csv_reader:
                n_dim, min_sample, cluster_size = (
                    int(n_dim),
                    int(min_sample),
                    int(cluster_size),
                )
                try:
                    try:
                        all_clusters_to_do[n_dim][min_sample].remove(cluster_size)
                    except ValueError:  # Not in the list
                        pass
                except KeyError:
                    pass
        for n_dim in all_clusters_to_do:
            for min_sample in all_clusters_to_do[n_dim]:
                if len(all_clusters_to_do[n_dim][min_sample]) != 0:
                    try:
                        to_return[n_dim][min_sample] = all_clusters_to_do[n_dim][
                            min_sample
                        ]
                    except KeyError:
                        to_return[n_dim] = dict()
                        to_return[n_dim][min_sample] = all_clusters_to_do[n_dim][
                            min_sample
                        ]

    # In case no records found
    except FileNotFoundError:
        to_return = all_clusters_to_do

    return to_return


def get_best_model(grid_search_results_path):
    """
    Get the csv file where the results are recorded to find the best model from the grid search
    Output the metrics needed to run the hdbscan and with which dimension reduction from UMAP.

    :params:
        grid_search_results_path str(): Full pathfile from the gridsearch results csv file

    :output:
        dictionary containing:
            n_dim int()
            cluster_size int()
            min_sample int()

    """
    list_results = list()
    with open(grid_search_results_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            list_results.append(row)
    current_highest = 0
    for d in list_results:
        if float(d["validity_index_score"]) > current_highest:
            best_dict = d
            current_highest = float(d["validity_index_score"])
    for k in best_dict:
        if k == "validity_index_score":
            best_dict[k] = float(best_dict[k])
        else:
            best_dict[k] = int(best_dict[k])
    return best_dict


if __name__ == "__main__":

    # Getting the parent folder
    parent_folder = Path(__file__).resolve().parents[1]

    # Getting the variables from cluster
    folder_name = config_cluster.folder_name
    data_filename = config_cluster.data_filename
    sent_emb_filename = config_cluster.sent_emb_filename
    corpus_filename = config_cluster.corpus_filename
    preprocess_filename = config_cluster.preprocess_filename
    grid_search_results_filename = config_cluster.grid_search_results
    bert_model_name = config_cluster.bert_model_name

    # Create the full path
    data_path = os.path.join(parent_folder, folder_name, "data", data_filename)
    corpus_path = os.path.join(parent_folder, folder_name, "data", corpus_filename)
    sent_emb_path = os.path.join(parent_folder, folder_name, "data", sent_emb_filename)
    preprocess_path = os.path.join(
        parent_folder, folder_name, "data", preprocess_filename
    )
    grid_search_results_path = os.path.join(
        parent_folder, folder_name, "cluster", grid_search_results_filename
    )

    num_cpu = config_cluster.num_cpu
    list_n_dim = config_cluster.list_n_dim
    list_cluster_size = config_cluster.list_cluster_size
    list_min_size = config_cluster.list_min_size

    umap_n_neighbours = config_cluster.umap_n_neighbours
    umap_min_dist = config_cluster.umap_min_dist

    # Create the folders

    cluster_path = os.path.join(parent_folder, folder_name, "cluster")
    dim_red_path = os.path.join(parent_folder, folder_name, "dim_reduction")
    output_path = os.path.join(parent_folder, folder_name, "output")

    Path(cluster_path).mkdir(parents=True, exist_ok=True)
    Path(dim_red_path).mkdir(parents=True, exist_ok=True)

    csv_columns = [
        "n_dim",
        "min_sample",
        "min_cluster_size",
        "validity_index_score",
        "noise",
        "top_10",
        "top_1",
        "n_clusters",
        "total",
    ]

    # Create all the different models to test from the list of n_dim, min_sample, cluster size
    all_clusters_to_do = create_clusters_to_do(
        list_n_dim, list_min_size, list_cluster_size
    )

    # Removing the models already done with the grid search during previous runs
    all_clusters_to_do = get_already_done(grid_search_results_path, all_clusters_to_do)

    queue_report = multiprocessing.Queue()

    # Create the list to holds all the jobs to be run in parallel.
    logging.info("Getting the sentence to pass into cluster")
    try:
        data = load(sent_emb_path)
    except FileNotFoundError:  # need to create the embeddings
        data = create_sent_embeddings(bert_model_name, sent_emb_path, corpus_path)
    data = data.tolist()

    writer = multiprocessing.Process(
        target=write_report,
        args=(queue_report, grid_search_results_path, csv_columns),
    )
    writer.start()
    for n_dim in all_clusters_to_do:

        jobs = list()

        if n_dim != "original":
            data_reduced = create_umap_reduction(
                data,
                dim_red_path,
                n_dim,
                umap_n_neighbours,
                umap_min_dist,
                low_memory=True,
                record=True,
            )
        else:
            data_reduced = data
        for min_sample in all_clusters_to_do[n_dim]:
            list_cluster_size = all_clusters_to_do[n_dim][min_sample]
            p = multiprocessing.Process(
                target=getting_hdscan,
                args=(
                    cluster_path,
                    data_reduced,
                    n_dim,
                    list_cluster_size,
                    min_sample,
                    queue_report,
                    False,
                ),
            )
            jobs.append(p)

        logging.info("Number of jobs: {}".format(len(jobs)))

        # Start the process that wrote the dictionary into the csv file
        for i in tqdm(chunks(jobs, num_cpu), total=len(jobs)):
            for j in i:
                j.start()
            for j in i:
                j.join()
        del data_reduced
    queue_report.put("END")

    logging.info("Recording the best cluster methods based on the validation metric")
    best_models_stats = get_best_model(grid_search_results_path)
    logging.info(
        """
                 Best models:
                 \tValidity Result: {}
                 \tNumber of dimensions: {}
                 \tMin sample: {}
                 \tCluster size: {}

                 """.format(
            best_models_stats["validity_index_score"],
            best_models_stats["n_dim"],
            best_models_stats["min_sample"],
            best_models_stats["min_cluster_size"],
        )
    )
    best_data_red = create_umap_reduction(
        data,
        dim_red_path,
        n_dim=best_models_stats["n_dim"],
        umap_n_neighbours=umap_n_neighbours,
        umap_min_dist=umap_min_dist,
        low_memory=True,
        record=False,
    )

    logging.info("Fitting the best model and recording it")
    clusterer = getting_hdscan(
        cluster_path=cluster_path,
        data=best_data_red,
        n_dim=best_models_stats["n_dim"],
        list_cluster_size=best_models_stats["min_cluster_size"],
        min_sample=best_models_stats["min_sample"],
        queue_report=False,
        record="best_model.pkl",
        return_model=True,
        prediction_data=True,
    )
    logging.info("Recording information associated to the best model")
    with open(os.path.join(cluster_path, "best_model_params.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(best_models_stats)

    logging.info("Getting the cluster id for all the data point and record the lists")
    with open(os.path.join(cluster_path, "cluster_id_and_proba.csv"), "w") as f:
        fieldnames = ["cluster", "cluster_proba"]
        writer = csv.writer(f)
        writer.writerows(fieldnames)
        writer.writerows(zip(clusterer.labels_, clusterer.probabilities_))
