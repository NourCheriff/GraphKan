import pandas as pd
from enum import Enum
import torch
from torch_geometric.data import Data
from os import path

# DATASET_PATH = "./../data/BPI12W/raw/BPI12W_processed.g"
# OUTPUT_DIR = "./../data/BPI12W/processed"
DATASET_PATH = ""
OUTPUT_DIR = ""


def main():
    df = pd.read_csv(DATASET_PATH, sep=",")

    # encode activities using One Hot
    prefix = "activity"
    encoded_activities = pd.get_dummies(df["activity"], prefix=prefix)
    df = pd.concat([df, encoded_activities], axis=1)

    # drop activity column since it's categorical
    df = df.drop("activity", axis=1)

    # define feature columns
    node_features = [
        "node1",
        "norm_time",
        "trace_time",
        "prev_event_time",
    ] + list(encoded_activities.columns)
    edge_features = ["node1", "node2"]

    # extract single graphs
    graph_start_indices = df[df["type"] == Type.GRAPH.value].index.to_list()
    graphs = []

    for i, graph_start_idx in enumerate(graph_start_indices):
        start_idx = graph_start_idx + 1
        end_idx = (
            graph_start_indices[i + 1] if i + 1 < len(graph_start_indices) else len(df)
        )

        graph_df = df.iloc[start_idx:end_idx]
        graphs.append(graph_df)

    # convert graphs to PyTorch Geometric Data objects
    test_set = []
    training_set = []

    for graph in graphs:
        dataset_split = graph["set"].iloc[0]

        nodes = graph[graph["type"] == Type.NODE.value][node_features]
        edges = graph[graph["type"] == Type.EDGE.value][edge_features]

        x = torch.tensor(nodes.to_numpy(dtype=float), dtype=torch.float32)
        edge_index = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)

        if dataset_split == "train":
            training_set.append(data)
        else:
            test_set.append(data)

    torch.save(training_set, path.join(OUTPUT_DIR, "train.pt"))
    torch.save(test_set, path.join(OUTPUT_DIR, "test.pt"))


class Type(Enum):
    GRAPH = "XP"
    NODE = "v"
    EDGE = "e"


if __name__ == "__main__":
    main()
