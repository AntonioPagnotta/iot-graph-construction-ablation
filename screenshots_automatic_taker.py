import os
import torch
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from pyvis.network import Network
from playwright.sync_api import sync_playwright
from tqdm.auto import tqdm

# ==========================================
# CONFIGURAZIONE
# ==========================================
TARGET_CONFIG = 'CONFIG_A'
BASE_GRAPHS_DIR = os.path.join('outputs', 'labeled_graphs')
POSITIONS_CACHE_DIR = os.path.join('outputs', 'positions_cache')
CSV_LABELS_PATH = os.path.join('outputs', 'output_final.csv')
TIMESTAMPS_JSON_PATH = os.path.join('data', 'graph_timestamps.json')

# Output directory per i frames
FRAMES_DIR = os.path.join('video_frames', 'frames')
os.makedirs(FRAMES_DIR, exist_ok=True)


# ==========================================
# HELPERS
# ==========================================
def take_screenshot(html_path: str, output_path: str):
    abs_url = "file://" + os.path.abspath(html_path)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(abs_url)
        page.wait_for_selector("canvas", state="attached")
        page.wait_for_timeout(3000)
        page.screenshot(path=output_path)
        browser.close()


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    try:
        r, g, b = mcolors.to_rgb(hex_color)
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha:.2f})"
    except Exception:
        return f"rgba(150,150,150,{alpha:.2f})"


# ==========================================
# INIZIALIZZAZIONE DATI
# ==========================================
print("Caricamento configurazioni e cache...")

_raw_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
COLOR_PALETTE = [c for c in _raw_palette if sum(mcolors.to_rgb(c)) < 2.4]
inactive_alpha = 0.12

# Carica Timestamps
timestamps_dict = {}
if os.path.exists(TIMESTAMPS_JSON_PATH):
    with open(TIMESTAMPS_JSON_PATH, 'r') as f:
        timestamps_dict = json.load(f)

# Carica Labels
df_labels = pd.read_csv(CSV_LABELS_PATH) if os.path.exists(CSV_LABELS_PATH) else pd.DataFrame()
if not df_labels.empty:
    df_labels['basename'] = df_labels['file_path'].apply(lambda x: os.path.basename(x))
    config_letter = TARGET_CONFIG.split('_')[-1]
    df_labels_config = df_labels[df_labels['config'] == config_letter]

# Carica Universo Nodi
cache_path_specific = os.path.join(POSITIONS_CACHE_DIR, f"{TARGET_CONFIG}_node_positions.pkl")
cache_path_generic = os.path.join(POSITIONS_CACHE_DIR, "node_positions.pkl")

if os.path.exists(cache_path_specific):
    with open(cache_path_specific, 'rb') as f:
        pos = pickle.load(f)
elif os.path.exists(cache_path_generic):
    with open(cache_path_generic, 'rb') as f:
        pos = pickle.load(f)
else:
    print("ERRORE: Nessuna posizione cache trovata! Avvia prima la dashboard Streamlit per generarle." + cache_path_generic)
    exit(1)

sorted_universe = tuple(sorted(list(pos.keys())))
universe_size = len(sorted_universe)
node_to_idx = {n: i for i, n in enumerate(sorted_universe)}
idx_to_node = {i: n for n, i in node_to_idx.items()}

# ==========================================
# LOOP PRINCIPALE SUI GRAFI
# ==========================================
files_to_process = []
for split in ['train', 'test']:
    directory = os.path.join(BASE_GRAPHS_DIR, TARGET_CONFIG, split)
    if os.path.exists(directory):
        for fname in sorted(os.listdir(directory)):
            if fname.endswith('.pt'):
                files_to_process.append((split, fname, os.path.join(directory, fname)))

print(f"Trovati {len(files_to_process)} grafi per {TARGET_CONFIG}. Avvio generazione frame...")

for split, selected_file, file_path in tqdm(files_to_process, desc="Generazione Screenshots"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Errore caricamento {file_path}: {e}")
        continue

    # Estrai Timestamp e crea un nome file sicuro
    dict_key = f"{TARGET_CONFIG}/{split}/{selected_file}"
    graph_time = timestamps_dict.get(dict_key, 'N/D')
    safe_time_str = graph_time.replace(" ", "_").replace(":", "-")
    png_file_path = os.path.join(FRAMES_DIR, f"{safe_time_str}.png")

    # Setup Labels per questo file
    edge_meta = {}
    pt_attack_set = set()

    if not df_labels.empty:
        df_current = df_labels_config[
            (df_labels_config['split'] == split) & (df_labels_config['basename'] == selected_file)]
        for _, row in df_current.iterrows():
            u, v = int(row['source_node']), int(row['target_node'])
            k = tuple(sorted([u, v]))
            t = int(row['true_label'])

            if pd.isna(row.get('predicted_label')):
                cat = 'Attacco (Non Classificato)' if t == 1 else 'Normale (Non Classificato)'
            else:
                p = int(row['predicted_label'])
                if t == 1 and p == 1:
                    cat = 'TP'
                elif t == 0 and p == 1:
                    cat = 'FP'
                elif t == 0 and p == 0:
                    cat = 'TN'
                elif t == 1 and p == 0:
                    cat = 'FN'
                else:
                    cat = 'Normale (Non Classificato)'
            edge_meta[k] = {'cat': cat, 't': t}

    # Estrazione archi dal PT
    edge_index_filtered = np.empty((2, 0), dtype=int)
    attack_edge_index = None

    if ('node', 'comm', 'node') in data.edge_types:
        ei = data['node', 'comm', 'node'].edge_index.numpy()
        valid_mask = np.isin(ei[0], sorted_universe) & np.isin(ei[1], sorted_universe)
        if ei[:, valid_mask].shape[1] > 0:
            remap = np.vectorize(node_to_idx.get)
            edge_index_filtered = np.stack([remap(ei[0, valid_mask]), remap(ei[1, valid_mask])])

    attack_types = [et for et in data.edge_types if 'attack' in et[1].lower()]
    if attack_types:
        ei_att = data[attack_types[0]].edge_index.numpy()
        valid_mask_att = np.isin(ei_att[0], sorted_universe) & np.isin(ei_att[1], sorted_universe)
        if ei_att[:, valid_mask_att].shape[1] > 0:
            remap = np.vectorize(node_to_idx.get)
            attack_edge_index = np.stack([remap(ei_att[0, valid_mask_att]), remap(ei_att[1, valid_mask_att])])
            for i in range(attack_edge_index.shape[1]):
                pt_attack_set.add(tuple(sorted([int(attack_edge_index[0, i]), int(attack_edge_index[1, i])])))

    active_nodes = set(edge_index_filtered[0].tolist()) | set(edge_index_filtered[1].tolist())
    if attack_edge_index is not None:
        active_nodes.update(attack_edge_index[0].tolist())
        active_nodes.update(attack_edge_index[1].tolist())

    # Community calcolo veloce
    G_slot = nx.Graph()
    G_slot.add_nodes_from(range(universe_size))
    G_slot.add_edges_from(edge_index_filtered.T)

    slot_communities = np.zeros(universe_size, dtype=int)
    if hasattr(data['node'], 'community'):
        raw_comm = data['node'].community.numpy()
        for orig_id, dense_idx in node_to_idx.items():
            if orig_id < len(raw_comm): slot_communities[dense_idx] = int(raw_comm[orig_id])
    else:
        communities_set = list(label_propagation_communities(G_slot))
        for cid, members in enumerate(communities_set):
            for n in members: slot_communities[n] = cid

    # Creazione PyVis
    net = Network(height="100vh", width="100%", bgcolor="#1a1a2e", font_color="#cccccc")
    net.toggle_physics(False)

    # Nodi
    for dense_idx in range(universe_size):
        comm_id = int(slot_communities[dense_idx])
        base_hex = COLOR_PALETTE[comm_id % len(COLOR_PALETTE)]
        is_active = dense_idx in active_nodes

        if is_active:
            bg_color, border_color, node_size, font_alpha = base_hex, "#ffffff", 10, 1.0
        else:
            bg_color = hex_to_rgba(base_hex, inactive_alpha)
            border_color = hex_to_rgba(base_hex, min(inactive_alpha * 2, 0.5))
            node_size, font_alpha = 5, inactive_alpha

        orig_id = idx_to_node[dense_idx]
        x, y = pos.get(orig_id, (0, 0))
        net.add_node(
            dense_idx,
            label=str(orig_id),
            color={"background": bg_color, "border": border_color},
            font={"color": f"rgba(200,200,200,{font_alpha:.2f})", "size": 8},
            x=float(x), y=float(y), size=node_size, fixed=True,
        )

    # Archi
    all_edges_to_draw = set()
    for i in range(edge_index_filtered.shape[1]):
        all_edges_to_draw.add((int(edge_index_filtered[0, i]), int(edge_index_filtered[1, i])))
    if attack_edge_index is not None:
        for i in range(attack_edge_index.shape[1]):
            all_edges_to_draw.add((int(attack_edge_index[0, i]), int(attack_edge_index[1, i])))

    for src, dst in all_edges_to_draw:
        orig_src = idx_to_node.get(src, src)
        orig_dst = idx_to_node.get(dst, dst)
        k = tuple(sorted([orig_src, orig_dst]))

        meta = edge_meta.get(k)
        is_pt_attack = k in pt_attack_set

        if meta is not None:
            cat = meta['cat']
        else:
            cat = 'Attacco (Non Classificato)' if is_pt_attack else 'Normale'

        # Only draw TP and TN edges — skip everything else
        if 'TP' not in cat and 'TN' not in cat:
            continue

        if 'TP' in cat:
            color, width, arrows = "rgba(46, 204, 113, 0.9)", 3.5, "to"
        else:  # TN
            color, width, arrows = "rgba(52, 152, 219, 0.3)", 0.6, ""

        net.add_edge(src, dst, color=color, width=width, arrows=arrows)

    html_tmp = "tmp_headless_graph.html"
    net.save_graph(html_tmp)

    # Inietta Timestamp gigante in BASSO a sinistra
    with open(html_tmp, 'r', encoding='utf-8') as f:
        html_content = f.read()

    overlay_html = f"""
    <div style="position:absolute;bottom:30px;left:30px;z-index:9999;background:rgba(20,20,30,0.85);padding:15px 25px;border-radius:10px;border:1px solid #444;color:#eee;font-family:sans-serif;">
        <h2 style="margin:0;color:#f39c12;font-size:32px;">{graph_time}</h2>
        <div style="margin-top:10px;font-size:16px;">
            <b>Legenda Archi</b><br>
            <span style="color:#2ecc71;">■</span> TP - Attacco Rilevato (Verde)<br>
            <span style="color:#3498db;">■</span> TN - Normale Corretto (Azzurro)
        </div>
    </div>
    """
    html_content = html_content.replace('<body>', f'<body>\n{overlay_html}')
    with open(html_tmp, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Scatta Screenshot
    take_screenshot(html_tmp, png_file_path)

# Pulizia
if os.path.exists("tmp_headless_graph.html"):
    os.remove("tmp_headless_graph.html")

print("\nFatto! Tutti gli screenshot sono salvati in 'video_frames/frames'.")