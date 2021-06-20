import os
import json
import logging

from pathlib import Path

import pandas as pd
import dataPreprocessing

from tqdm import tqdm

# print(help(dataPreprocessing))
# print(dataPreprocessing.__dict__)
from dataPreprocessing import preProcessingTweets

import config_cluster

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():

    # Getting the parent folder
    parent_folder = Path(__file__).resolve().parents[1]

    # Getting the variables from cluster
    folder_name = config_cluster.folder_name
    data_filename = config_cluster.data_filename
    sent_emb_filename = config_cluster.sent_emb_filename
    corpus_filename = config_cluster.corpus_filename
    preprocess_filename = config_cluster.preprocess_filename

    # Create the full path
    data_path = os.path.join(parent_folder, folder_name, "data", data_filename)
    corpus_path = os.path.join(parent_folder, folder_name, "data", corpus_filename)
    sent_emb_path = os.path.join(parent_folder, folder_name, "data", sent_emb_filename)
    preprocess_path = os.path.join(
        parent_folder, folder_name, "data", preprocess_filename
    )

    all_tweets = list()
    n = 0

    # Load dataset:
    added_features = list()
    with open(data_path, "r") as read_obj:
        for l in read_obj:
            tweet = json.loads(l)
            # tweet["created_at"] = tweet["created_at"]["$date"]

            processed_tweet = preProcessingTweets.process_tweet(tweet)
            added_features.append(processed_tweet)

    df = pd.DataFrame.from_dict(added_features)
    # df = pd.merge(df, pd.DataFrame.from_dict(added_features, orient="columns"), on="id")
    # drop previous index column
    # df = df.drop(df.columns[0], axis=1)
    # Load tweets
    df.reset_index().to_feather(preprocess_path)

    # For sentence embedding
    df[df["rt_status"] == False]["txt_wo_entities"].to_csv(
        corpus_path, index=False, header=False
    )
    # with open(corpus_path, "a") as f:
    #     for x in df["txt_wo_entities"].values:
    #
    #         f.write(x)
    # f.write("\n")
    # f.write(df.to_string(columns="txt_wo_entities", header=False, index=False))


if __name__ == "__main__":
    main()
