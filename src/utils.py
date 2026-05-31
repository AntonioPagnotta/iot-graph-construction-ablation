import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── COSTANTI ──────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Features per ogni configurazione di edge
EDGE_CONFIGS = {
    'A': ['comm'],
    'B': ['comm', 'context'],
    'C': ['comm', 'knowledge'],
    'D': ['comm', 'context', 'knowledge']
}

COMM_FEATURES = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
    'Flow Byts/s', 'Flow Pkts/s', 'Protocol'
]

CONTEXT_FEATURES = [
    'Timestamp', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Active Mean', 'Idle Mean'
]

KNOWLEDGE_FEATURES = [
    'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt'
]

# ── FUNZIONI DI CARICAMENTO ───────────────────────────────────────────────────

def load_dataset(path, subset_pct=None):
    """
    Carica il dataset e opzionalmente ne prende un subset stratificato.
    subset_pct: float tra 0 e 1 (es. 0.2 per il 20%)
    """
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df['Label_str'] = df['Label'].map({0: 'Benign', 1: 'Attack'})

    if subset_pct is not None:
        df, _ = train_test_split(
            df,
            test_size=1 - subset_pct,
            random_state=RANDOM_SEED,
            stratify=df['Label']
        )

    return df.reset_index(drop=True)


def train_test_split_temporal(df, test_ratio=0.3):
    """
    Split temporale: train sui primi giorni, test sugli ultimi.
    Più corretto di uno split random per dati time-series.
    """
    df_sorted = df.sort_values('Timestamp').reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    return train, test


# ── FUNZIONI DI ANALISI ───────────────────────────────────────────────────────

def print_dataset_summary(df):
    """Stampa un riassunto del dataset."""
    print(f"Righe totali:      {len(df):,}")
    print(f"Colonne:           {len(df.columns)}")
    print(f"IP sorgenti unici: {df['Src IP'].nunique():,}")
    print(f"IP dest unici:     {df['Dst IP'].nunique():,}")
    print(f"Periodo:           {df['Timestamp'].min()} → {df['Timestamp'].max()}")
    print(f"\nLabel distribution:")
    print(df['Label_str'].value_counts())






# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────

WINDOW_HOURS = 2  # dimensione finestra temporale

EDGE_FEATURES = {
    'comm': [
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
        'Flow Byts/s', 'Flow Pkts/s', 'Protocol'
    ],
    'context': [
        'Fwd IAT Mean', 'Bwd IAT Mean',
        'Active Mean', 'Idle Mean'
    ],
    'knowledge': [
        'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
        'PSH Flag Cnt', 'ACK Flag Cnt'
    ]
}



# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────

import networkx as nx

def build_graph(window_df, edge_types):
    """
    Costruisce un grafo NetworkX da un dataframe di flussi.
    edge_types: lista di tipi di arco ['comm', 'context', 'knowledge']
    """
    G = nx.MultiDiGraph()
    window_df = window_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    for _, row in window_df.iterrows():
        src = row['Src IP']
        dst = row['Dst IP']
        label = int(row['Label'])

        if src not in G:
            G.add_node(src)
        if dst not in G:
            G.add_node(dst)

        for etype in edge_types:
            features = {}
            for feat in EDGE_FEATURES[etype]:
                if feat in row.index:
                    features[feat] = row[feat]
            G.add_edge(src, dst, edge_type=etype, label=label, **features)

    return G


def build_all_graphs(df, configs, min_flows=100):
    """
    Costruisce grafi per tutte le configurazioni su tutte le finestre valide.
    Ritorna un dizionario config -> lista di grafi con metadati.
    """
    window_counts = df.groupby('window').size()
    valid_windows = window_counts[window_counts >= min_flows].index

    all_graphs = {}
    for config_name, edge_types in configs.items():
        graphs = []
        for window in valid_windows:
            window_df = df[df['window'] == window].copy()
            G = build_graph(window_df, edge_types)
            graphs.append({
                'window': window,
                'graph': G,
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'n_anomalies': sum(
                    1 for u, v, d in G.edges(data=True) if d['label'] == 1
                )
            })
        all_graphs[config_name] = graphs

    return all_graphs


def split_graphs(graphs, test_ratio=0.3):
    """Split temporale dei grafi in train e test."""
    split_idx = int(len(graphs) * (1 - test_ratio))
    return graphs[:split_idx], graphs[split_idx:]







# ── CONVERSIONE PyTorch Geometric ─────────────────────────────────────────────

import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler

def graph_to_heterodata(graph_dict, edge_types):
    """
    Converte un grafo NetworkX in formato HeteroData di PyTorch Geometric.
    """
    G = graph_dict['graph']
    nodes = list(G.nodes())
    node_idx = {ip: i for i, ip in enumerate(nodes)}
    n_nodes = len(nodes)

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    node_features = torch.tensor(
        [[in_deg[n], out_deg[n]] for n in nodes],
        dtype=torch.float
    )

    data = HeteroData()
    data['node'].x = node_features
    data['node'].num_nodes = n_nodes

    for etype in edge_types:
        src_list, dst_list, feat_list, label_list = [], [], [], []
        feat_cols = EDGE_FEATURES[etype]

        for u, v, d in G.edges(data=True):
            if d.get('edge_type') != etype:
                continue
            src_list.append(node_idx[u])
            dst_list.append(node_idx[v])
            label_list.append(d.get('label', 0))
            feat_list.append([d.get(f, 0.0) for f in feat_cols])

        if len(src_list) == 0:
            continue

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(feat_list, dtype=torch.float)
        edge_label = torch.tensor(label_list, dtype=torch.long)

        scaler = MinMaxScaler()
        edge_attr = torch.tensor(
            scaler.fit_transform(edge_attr.numpy()),
            dtype=torch.float
        )

        data['node', etype, 'node'].edge_index = edge_index
        data['node', etype, 'node'].edge_attr  = edge_attr
        data['node', etype, 'node'].edge_label = edge_label

    return data










# ── COMMUNITY DETECTION ───────────────────────────────────────────────────────

from networkx.algorithms.community import label_propagation_communities
from collections import Counter

def run_lpa(G):
    """
    Applica LPA al grafo NetworkX.
    Ritorna dizionario nodo -> community_id.
    """
    G_undirected = G.to_undirected()
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
    communities = label_propagation_communities(G_undirected)
    
    community_map = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            community_map[node] = comm_id
    
    return community_map


def add_community_labels(hetero_data, community_map, node_list):
    """
    Aggiunge community label come feature aggiuntiva ai nodi.
    """
    comm_labels = torch.tensor(
        [community_map.get(node, -1) for node in node_list],
        dtype=torch.float
    ).unsqueeze(1)
    
    hetero_data['node'].x = torch.cat(
        [hetero_data['node'].x, comm_labels], dim=1
    )
    
    return hetero_data


