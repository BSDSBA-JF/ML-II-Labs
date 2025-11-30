import re
import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Union, List, Tuple


def extract_phase(filename: str) -> Union[int, float]:
    """
    Extract the numeric phase from a filename.

    Parameters
    ----------
    filename : str
        Filename containing a numeric phase.

    Returns
    -------
    int or float
        Extracted phase number as int, or float('inf') if not found.
    """
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else float("inf")


def build_matrix(matrix_dict: Dict[int, Dict[str, float]],
                 name: str,
                 all_characters: list,
                 phases: list) -> pd.DataFrame:
    """
    Build a DataFrame from a centrality dictionary and save as CSV.

    Parameters
    ----------
    matrix_dict : dict
        Dictionary of centrality values per phase: {phase: {character: score}}
    name : str
        Name used for the output CSV file.
    all_characters : list
        List of all character names (rows).
    phases : list
        List of all phases (columns).

    Returns
    -------
    pd.DataFrame
        DataFrame with characters as rows and phases as columns.
    """
    df = pd.DataFrame(index=all_characters, columns=phases, dtype=float)

    for phase in phases:
        for char in all_characters:
            df.at[char, phase] = matrix_dict.get(phase, {}).get(char, np.nan)

    # Drop the traveler and rename Ambor to Amber
    df = df.drop('Traveler')
    df = df.rename(index={'Ambor':'Amber'})
    
    return df

def compute_centrality_matrices(
    base_dir: str,
    output_name_prefix: str,
    patch_labels: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute or load betweenness and eigenvector centrality matrices for a set of graphs.

    Parameters
    ----------
    base_dir : str
        Directory containing pickled graph files (graph_*.pickle).
    output_name_prefix : str
        Prefix used for output CSV filenames.
    patch_labels : List[str]
        List of patch labels to assign as column names.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Betweenness and eigenvector centrality DataFrames.
    """
    betweenness_csv = os.path.join("data", f"{output_name_prefix}_betweenness.csv")
    eigenvector_csv = os.path.join("data", f"{output_name_prefix}_eigenvector.csv")

    # If CSVs exist, load them
    if os.path.exists(betweenness_csv) and os.path.exists(eigenvector_csv):
        betweenness_df = pd.read_csv(betweenness_csv, index_col=0)
        eigenvector_df = pd.read_csv(eigenvector_csv, index_col=0)
        print(f"Loaded existing centrality matrices from {betweenness_csv} and {eigenvector_csv}")
        return betweenness_df, eigenvector_df

    # Otherwise, compute centralities
    graph_files = sorted(
        [
            f
            for f in os.listdir(base_dir)
            if f.startswith("graph_") and f.endswith(".pickle")
        ],
        key=extract_phase,
    )

    degree_matrix = {}
    betweenness_matrix = {}
    eigenvector_matrix = {}
    all_characters = set()

    for file in graph_files:
        phase = extract_phase(file)
        filepath = os.path.join(base_dir, file)

        with open(filepath, "rb") as f:
            G = pickle.load(f)

        if not isinstance(G, nx.Graph):
            G = nx.Graph(G)

        nodes = list(G.nodes())
        all_characters.update(nodes)

        degree_cent = nx.degree_centrality(G)

        # Betweenness
        for u, v, data in G.edges(data=True):
            if "weight" in data and data["weight"] != 0:
                data["distance"] = 1.0 / data["weight"]
            else:
                data["distance"] = float("inf")
        betweenness_cent = nx.betweenness_centrality(G, weight="distance", normalized=True)

        eigenvector_cent = nx.eigenvector_centrality(G, weight="weight", max_iter=500)

        degree_matrix[phase] = degree_cent
        betweenness_matrix[phase] = betweenness_cent
        eigenvector_matrix[phase] = eigenvector_cent

    all_characters = sorted(all_characters)
    phases = sorted(degree_matrix.keys())

    betweenness_df = build_matrix(
        betweenness_matrix, f"{output_name_prefix}_betweenness", all_characters, phases
    )
    betweenness_df.columns = patch_labels

    eigenvector_df = build_matrix(
        eigenvector_matrix, f"{output_name_prefix}_eigenvector", all_characters, phases
    )
    eigenvector_df.columns = patch_labels

    betweenness_df.to_csv(os.path.join("data", f"{output_name_prefix}_betweenness.csv"))
    eigenvector_df.to_csv(os.path.join("data", f"{output_name_prefix}_eigenvector.csv"))

    print(f"Computed and saved centrality matrices for {output_name_prefix}")
    return betweenness_df, eigenvector_df
