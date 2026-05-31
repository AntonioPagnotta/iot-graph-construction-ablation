import streamlit as st
import streamlit.components.v1 as components
import torch
import os
import pickle
import json
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.colors as mcolors
from pyvis.network import Network
import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="IoT Graph Dashboard")
st.title("🌐 Analisi Grafi IoT & Ablation Study")

BASE_GRAPHS_DIR = os.path.join('outputs', 'labeled_graphs')
POSITIONS_CACHE_DIR = os.path.join('outputs', 'positions_cache')
CSV_LABELS_PATH = os.path.join('outputs', 'edge_labels_5ep.csv')
TIMESTAMPS_JSON_PATH = os.path.join('data', 'graph_timestamps.json')
os.makedirs(POSITIONS_CACHE_DIR, exist_ok=True)

ALL_CONFIGS = ['CONFIG_A', 'CONFIG_B', 'CONFIG_C', 'CONFIG_D']
ALL_SPLITS = ['train', 'test']


# ---------------------------------------------------------------------------
# HELPERS - GRAPH VISUALIZATION
# ---------------------------------------------------------------------------

def take_screenshot(html_path: str, output_path: str):
    abs_url = "file://" + os.path.abspath(html_path)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(abs_url)
        page.wait_for_selector("canvas", state="attached")
        page.wait_for_timeout(5000)
        page.screenshot(path=output_path)
        browser.close()


@st.cache_data
def get_available_graphs(directory: str) -> list[str]:
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.endswith('.pt')])


@st.cache_data(show_spinner="Scansione di tutti i file per trovare i nodi con almeno un arco…")
def get_ever_active_nodes(base_graphs_dir: str, configs: tuple[str, ...], splits: tuple[str, ...]) -> tuple[int, ...]:
    ever_active: set[int] = set()
    for config in configs:
        for split in splits:
            directory = os.path.join(base_graphs_dir, config, split)
            if not os.path.exists(directory):
                continue
            for fname in sorted(os.listdir(directory)):
                if not fname.endswith('.pt'):
                    continue
                try:
                    d = torch.load(os.path.join(directory, fname), weights_only=False)
                    if ('node', 'comm', 'node') in d.edge_types:
                        ei = d['node', 'comm', 'node'].edge_index.numpy()
                        ever_active.update(ei[0].tolist())
                        ever_active.update(ei[1].tolist())

                    attack_types = [et for et in d.edge_types if 'attack' in et[1].lower()]
                    for at in attack_types:
                        ei_att = d[at].edge_index.numpy()
                        ever_active.update(ei_att[0].tolist())
                        ever_active.update(ei_att[1].tolist())
                except Exception:
                    pass
    return tuple(sorted(ever_active))


@st.cache_data
def load_edge_labels(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df['basename'] = df['file_path'].apply(lambda x: os.path.basename(x))
    return df


@st.cache_data
def load_timestamps(json_path: str) -> dict:
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    try:
        r, g, b = mcolors.to_rgb(hex_color)
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha:.2f})"
    except Exception:
        return f"rgba(150,150,150,{alpha:.2f})"


def load_positions(cache_path: str) -> dict | None:
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


_raw_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
COLOR_PALETTE = [c for c in _raw_palette if sum(mcolors.to_rgb(c)) < 2.4]


# ---------------------------------------------------------------------------
# HELPERS - ABLATION STUDY (METRICS)
# ---------------------------------------------------------------------------

@st.cache_data
def load_ablation_data():
    metrics_dict = {
        "CONFIG_A": {
            "AUC": 0.9989,
            "Precision": 0.9961,
            "Recall": 0.9859,
            "F1": 0.9910,
            "Accuracy": 0.9896
        },
        "CONFIG_B": {
            "AUC": 0.9860,
            "Precision": 0.9791,
            "Recall": 0.4007,
            "F1": 0.5687,
            "Accuracy": 0.6478
        },
        "CONFIG_C": {
            "AUC": 0.9995,
            "Precision": 0.9988,
            "Recall": 0.9824,
            "F1": 0.9905,
            "Accuracy": 0.9891
        },
        "CONFIG_D": {
            "AUC": 0.9645,
            "Precision": 0.9855,
            "Recall": 0.4092,
            "F1": 0.5783,
            "Accuracy": 0.6542
        }
    }

    df_metrics = pd.DataFrame(metrics_dict).T.reset_index()
    df_metrics.rename(columns={'index': 'Configuration'}, inplace=True)

    if 'F1' in df_metrics.columns:
        df_metrics.rename(columns={'F1': 'F1-Score'}, inplace=True)

    return df_metrics


# ---------------------------------------------------------------------------
# STEP 0 — Build the ever-active node universe & load timestamps
# ---------------------------------------------------------------------------
sorted_universe: tuple[int, ...] = get_ever_active_nodes(
    BASE_GRAPHS_DIR, tuple(ALL_CONFIGS), tuple(ALL_SPLITS)
)
universe_size = len(sorted_universe)
node_to_idx: dict[int, int] = {n: i for i, n in enumerate(sorted_universe)}
idx_to_node: dict[int, int] = {i: n for n, i in node_to_idx.items()}

timestamps_dict = load_timestamps(TIMESTAMPS_JSON_PATH)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 1. Esploratore Grafi")
    st.info(f"**Universo nodi:** {universe_size:,} nodi")
    selected_config = st.selectbox("Seleziona Configurazione (Grafo):", ALL_CONFIGS)
    selected_split = st.selectbox("Seleziona Split (Grafo):", ALL_SPLITS)

    current_dir = os.path.join(BASE_GRAPHS_DIR, selected_config, selected_split)
    files = get_available_graphs(current_dir)

    if files:
        selected_file = st.selectbox("Seleziona lo Slot Temporale:", files)
    else:
        selected_file = None
        st.error(f"Nessun file `.pt` trovato in:\n`{current_dir}`")

    st.markdown("---")
    st.header("🎨 Filtri Archi & Predizioni (Grafo)")

    ALL_EDGE_CATS = [
        "Attacchi Generali (Tutti i True Label = 1)",
        "TP (Attacco Rilevato)",
        "FP (Falso Allarme)",
        "FN (Attacco Mancato)",
        "TN (Normale Corretto)",
        "Attacco (Non Classificato)",
        "Normale (Non Classificato)"
    ]
    show_categories = st.multiselect(
        "Seleziona quali archi mostrare nel grafo:",
        options=ALL_EDGE_CATS,
        default=["Attacchi Generali (Tutti i True Label = 1)", "FP (Falso Allarme)"]
    )

    inactive_alpha = st.slider("Trasparenza nodi inattivi", min_value=0.02, max_value=0.5, value=0.12, step=0.02)
    st.info("ℹ️ Il calcolo automatico della posizione è disabilitato (letta dalla cache statica).")

    st.markdown("---")

    st.header("⚙️ 2. Ablation Study & Metriche")
    df_metrics = load_ablation_data()

    selected_configs = st.multiselect(
        "Seleziona Configurazioni da confrontare:",
        options=df_metrics['Configuration'].tolist(),
        default=df_metrics['Configuration'].tolist()
    )

# ---------------------------------------------------------------------------
# TABS SETUP
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🌐 Visualizzatore Grafi (Community Detection)", "📊 Valutazione Modelli (Ablation Study)"])

# ===========================================================================
# TAB 1: GRAPH VISUALIZER
# ===========================================================================
with tab1:
    st.markdown(
        "Posizioni fisse per tutti i nodi **mai attivi**. "
        "Questa vista combina la classificazione del modello con la realtà (True Labels) per visualizzare i risultati come TP, FP, TN e FN."
    )

    data_loaded_successfully = False

    if universe_size == 0:
        st.error("Nessun nodo con archi trovato in nessun file di nessuna configurazione.")
    elif not selected_file:
        st.warning(f"Seleziona o carica un file PT per analizzare il grafo.")
    else:
        file_path = os.path.join(current_dir, selected_file)
        try:
            data = torch.load(file_path, weights_only=False)
            data_loaded_successfully = True
        except Exception as e:
            st.error(f"Errore nel caricamento del file PT: {e}")

    if data_loaded_successfully:
        # Load CSV and filter for current graph
        df_labels = load_edge_labels(CSV_LABELS_PATH)
        config_letter = selected_config.split('_')[-1] if '_' in selected_config else selected_config

        if not df_labels.empty:
            df_current = df_labels[
                (df_labels['config'] == config_letter) &
                (df_labels['split'] == selected_split) &
                (df_labels['basename'] == selected_file)
                ]
        else:
            df_current = pd.DataFrame()

        # Parse classifications into a dictionary
        edge_meta = {}
        attack_details = []

        for _, row in df_current.iterrows():
            try:
                u, v = int(row['source_node']), int(row['target_node'])
                k = tuple(sorted([u, v]))
                t = int(row['true_label'])

                if pd.isna(row.get('predicted_label')):
                    cat = 'Attacco (Non Classificato)' if t == 1 else 'Normale (Non Classificato)'
                    p = None
                else:
                    p = int(row['predicted_label'])
                    if t == 1 and p == 1:
                        cat = 'TP (Attacco Rilevato)'
                    elif t == 0 and p == 1:
                        cat = 'FP (Falso Allarme)'
                    elif t == 0 and p == 0:
                        cat = 'TN (Normale Corretto)'
                    elif t == 1 and p == 0:
                        cat = 'FN (Attacco Mancato)'
                    else:
                        cat = 'Normale (Non Classificato)'

                edge_meta[k] = {'cat': cat, 't': t, 'p': p}

                if t == 1 or p == 1:
                    attack_details.append({
                        "Origine": u,
                        "Destinazione": v,
                        "Tipo Arco": row.get('edge_type', 'N/D'),
                        "Classificazione": cat.split(' ')[0],
                        "Probabilità Anomalia": f"{row.get('probability', 0):.4f}" if not pd.isna(
                            row.get('probability')) else "N/D"
                    })
            except Exception:
                pass

        # ------------------- Metrics display -------------------
        # Fetch timestamp mapped from JSON
        dict_key = f"{selected_config}/{selected_split}/{selected_file}"
        graph_time = timestamps_dict.get(dict_key, 'N/D')

        st.markdown(f"### 🕒 Time Slot: `{graph_time}`")

        col1_1, col1_2, col1_3, col1_4 = st.columns(4)
        col1_1.metric("File", selected_file)
        col1_2.metric("Nodi Universo", f"{universe_size:,}")

        # Extract edges
        edge_index_filtered = np.empty((2, 0), dtype=int)
        attack_edge_index = None

        if ('node', 'comm', 'node') in data.edge_types:
            ei = data['node', 'comm', 'node'].edge_index.numpy()
            valid_mask = np.isin(ei[0], sorted_universe) & np.isin(ei[1], sorted_universe)
            if ei[:, valid_mask].shape[1] > 0:
                remap = np.vectorize(node_to_idx.get)
                edge_index_filtered = np.stack([remap(ei[0, valid_mask]), remap(ei[1, valid_mask])])

        attack_types = [et for et in data.edge_types if 'attack' in et[1].lower()]
        pt_attack_set = set()
        if attack_types:
            ei_att = data[attack_types[0]].edge_index.numpy()
            valid_mask_att = np.isin(ei_att[0], sorted_universe) & np.isin(ei_att[1], sorted_universe)
            if ei_att[:, valid_mask_att].shape[1] > 0:
                remap = np.vectorize(node_to_idx.get)
                attack_edge_index = np.stack([remap(ei_att[0, valid_mask_att]), remap(ei_att[1, valid_mask_att])])
                for i in range(attack_edge_index.shape[1]):
                    pt_attack_set.add(tuple(sorted([int(attack_edge_index[0, i]), int(attack_edge_index[1, i])])))

        total_edges = edge_index_filtered.shape[1] + (
            attack_edge_index.shape[1] if attack_edge_index is not None else 0)
        col1_3.metric("Archi Totali nel PT", f"{total_edges:,}")

        active_nodes: set[int] = set(edge_index_filtered[0].tolist()) | set(edge_index_filtered[1].tolist())
        if attack_edge_index is not None:
            active_nodes.update(attack_edge_index[0].tolist())
            active_nodes.update(attack_edge_index[1].tolist())

        col1_4.metric("Nodi Attivi in questo Slot", f"{len(active_nodes):,}")

        # Sub-Metrics - Calculate all required counts
        tp_c = len([1 for m in edge_meta.values() if 'TP' in m['cat']])
        fp_c = len([1 for m in edge_meta.values() if 'FP' in m['cat']])
        tn_c = len([1 for m in edge_meta.values() if 'TN' in m['cat']])
        fn_c = len([1 for m in edge_meta.values() if 'FN' in m['cat']])
        uncl_a = len([1 for m in edge_meta.values() if m['cat'] == 'Attacco (Non Classificato)'])

        # Ground Truth totals
        att_gen_c = len([1 for m in edge_meta.values() if m['t'] == 1])
        norm_gen_c = len([1 for m in edge_meta.values() if m['t'] == 0])

        if not df_current.empty:
            st.info(
                f"📊 **Statistiche Dettagliate Archi:** \n\n"
                f"🔴 **Attacco Generale (Tutti i True Label = 1):** `{att_gen_c}` | "
                f"⚪ **Normale (Totale):** `{norm_gen_c}`\n\n"
                f"🟢 **TP (Rilevati):** `{tp_c}` | "
                f"🟠 **FP (Falsi Allarmi):** `{fp_c}` | "
                f"🟣 **FN (Mancati):** `{fn_c}` | "
                f"🔵 **TN (Corretti):** `{tn_c}` | "
                f"🌺 **Attacco Senza Predizione:** `{uncl_a}`"
            )

            if attack_details:
                with st.expander(
                        f"🔍 Dettaglio degli {len(attack_details)} archi (Attacchi Reali o Falsi Allarmi) in questo grafo"):
                    st.dataframe(pd.DataFrame(attack_details), use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Nessun dato sulle label trovato nel file CSV per questo time-slot.")

        st.markdown("---")

        # ------------------- Static Positions & Communities -------------------
        if 'global_communities' not in st.session_state:
            st.session_state['global_communities'] = np.zeros(universe_size, dtype=int)

        G_slot = nx.Graph()
        G_slot.add_nodes_from(range(universe_size))
        G_slot.add_edges_from(edge_index_filtered.T)

        if hasattr(data['node'], 'community'):
            raw_comm = data['node'].community.numpy()
            slot_communities = np.zeros(universe_size, dtype=int)
            for orig_id, dense_idx in node_to_idx.items():
                if orig_id < len(raw_comm):
                    slot_communities[dense_idx] = int(raw_comm[orig_id])
        else:
            communities_set = list(label_propagation_communities(G_slot))
            slot_communities = np.zeros(universe_size, dtype=int)
            for cid, members in enumerate(communities_set):
                for n in members:
                    slot_communities[n] = cid

        st.session_state['global_communities'] = slot_communities.copy()
        global_communities = st.session_state['global_communities']

        cache_path_specific = os.path.join(POSITIONS_CACHE_DIR, f"{selected_config}_node_positions.pkl")
        cache_path_generic = os.path.join(POSITIONS_CACHE_DIR, "node_positions.pkl")

        pos = None
        if os.path.exists(cache_path_specific):
            pos = load_positions(cache_path_specific)
        elif os.path.exists(cache_path_generic):
            pos = load_positions(cache_path_generic)

        if pos is None:
            st.error(
                f"❌ Impossibile trovare il file delle posizioni!\nAssicurati che `{cache_path_specific}` o `{cache_path_generic}` esista.")
        else:
            missing = [n for n in range(universe_size) if n not in pos]
            if missing:
                rng = np.random.default_rng(seed=2)
                for n in missing:
                    angle = 2 * np.pi * rng.random()
                    r = 3500.0 + rng.random() * 500.0
                    pos[n] = np.array([r * np.cos(angle), r * np.sin(angle)])

            # ------------------- Render pyvis -------------------
            html_file_path = "tmp_graph.html"
            png_file_path = "graph_screenshot.png"
            attack_pairs_list = []

            with st.spinner("Generazione visualizzazione interattiva…"):
                net = Network(height="100vh", width="100%", bgcolor="#1a1a2e", font_color="#cccccc")
                net.toggle_physics(False)

                for dense_idx in range(universe_size):
                    comm_id = int(global_communities[dense_idx])
                    base_hex = COLOR_PALETTE[comm_id % len(COLOR_PALETTE)]
                    is_active = dense_idx in active_nodes

                    if is_active:
                        bg_color, border_color, node_size, font_alpha = base_hex, "#ffffff", 10, 1.0
                    else:
                        bg_color = hex_to_rgba(base_hex, inactive_alpha)
                        border_color = hex_to_rgba(base_hex, min(inactive_alpha * 2, 0.5))
                        node_size, font_alpha = 5, inactive_alpha

                    orig_id = idx_to_node[dense_idx]
                    x, y = pos[dense_idx]
                    net.add_node(
                        dense_idx,
                        label=str(orig_id),
                        title=f"Node {orig_id} | Comm {comm_id} | {'ATTIVO' if is_active else 'inattivo'}",
                        color={"background": bg_color, "border": border_color},
                        font={"color": f"rgba(200,200,200,{font_alpha:.2f})", "size": 8},
                        x=float(x), y=float(y), size=node_size, fixed=True,
                    )

                all_edges_to_draw = set()
                for i in range(edge_index_filtered.shape[1]):
                    all_edges_to_draw.add((int(edge_index_filtered[0, i]), int(edge_index_filtered[1, i])))
                if attack_edge_index is not None:
                    for i in range(attack_edge_index.shape[1]):
                        all_edges_to_draw.add((int(attack_edge_index[0, i]), int(attack_edge_index[1, i])))

                show_general_attacks = "Attacchi Generali (Tutti i True Label = 1)" in show_categories

                for src, dst in all_edges_to_draw:
                    orig_src = idx_to_node.get(src, src)
                    orig_dst = idx_to_node.get(dst, dst)
                    k = tuple(sorted([orig_src, orig_dst]))

                    meta = edge_meta.get(k)
                    is_pt_attack = tuple(sorted([src, dst])) in pt_attack_set

                    if meta is not None:
                        cat = meta['cat']
                        t_label = meta['t']
                    else:
                        cat = 'Attacco (Non Classificato)' if is_pt_attack else 'Normale (Non Classificato)'
                        t_label = 1 if is_pt_attack else 0

                    # Valutazione override per "Attacchi Generali"
                    if t_label == 1 and show_general_attacks:
                        color, width, label, arrows = "rgba(231, 76, 60, 0.9)", 3.5, "Attacco Reale (Generale)", "to"  # Rosso
                    else:
                        # Logica matrice standard
                        if cat not in show_categories:
                            continue

                        if 'TP' in cat:
                            color, width, label, arrows = "rgba(46, 204, 113, 0.9)", 3.5, "TP (Attacco Rilevato)", "to"  # Verde
                        elif 'FP' in cat:
                            color, width, label, arrows = "rgba(230, 126, 34, 0.9)", 2.5, "FP (Falso Allarme)", "to"  # Arancio
                        elif 'FN' in cat:
                            color, width, label, arrows = "rgba(155, 89, 182, 0.9)", 3.0, "FN (Attacco Mancato)", "to"  # Viola
                        elif 'TN' in cat:
                            color, width, label, arrows = "rgba(52, 152, 219, 0.3)", 0.6, "TN (Normale Corretto)", ""  # Azzurro
                        elif cat == 'Attacco (Non Classificato)':
                            color, width, label, arrows = "rgba(232, 67, 147, 0.9)", 3.0, "Attacco (Nessuna Predizione)", "to"  # Rosa scuro / Magenta
                        else:
                            color, width, label, arrows = "rgba(180, 180, 180, 0.3)", 0.6, cat, ""  # Grigio

                    net.add_edge(src, dst, color=color, width=width, title=label, arrows=arrows)

                    if t_label == 1 or 'FP' in cat:
                        attack_pairs_list.append([src, dst])

                attack_pairs_js = json.dumps(attack_pairs_list)
                net.save_graph(html_file_path)

                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                overlay_html = f"""
                <div style="position:absolute;top:12px;left:12px;z-index:9999;display:flex;flex-direction:column;gap:10px;">
                  <div style="display:flex;gap:8px;">
                    <button onclick="toggleFS()" style="padding:8px 14px;background:#34495e;color:#fff;border:none;border-radius:5px;cursor:pointer;">⛶ Full Screen</button>
                    <button onclick="resetView()" style="padding:8px 14px;background:#2c7a4b;color:#fff;border:none;border-radius:5px;cursor:pointer;">⌖ Reset View</button>
                    <button onclick="zoomRandomAttack()" style="padding:8px 14px;background:#c0392b;color:#fff;border:none;border-radius:5px;cursor:pointer;">🎯 Zoom Anomalia</button>
                  </div>
                  <div style="background:rgba(20,20,30,0.85);padding:10px;border-radius:6px;border:1px solid #444;color:#eee;font-size:13px;width:fit-content;">
                    <b>Legenda Archi</b><br>
                    <span style="color:#e74c3c;">■</span> Attacco Generale (Rosso)<br>
                    <span style="color:#2ecc71;">■</span> TP - Attacco Rilevato (Verde)<br>
                    <span style="color:#e67e22;">■</span> FP - Falso Allarme (Arancio)<br>
                    <span style="color:#9b59b6;">■</span> FN - Attacco Mancato (Viola)<br>
                    <span style="color:#3498db;">■</span> TN - Normale Corretto (Azzurro)<br>
                    <span style="color:#e84393;">■</span> Attacco Senza Predizione (Rosa)<br>
                    <span style="color:#999999;">■</span> Normale
                  </div>
                </div>
                <script>
                var attackPairs = {attack_pairs_js};
                function toggleFS(){{
                    if(!document.fullscreenElement) document.documentElement.requestFullscreen();
                    else if(document.exitFullscreen) document.exitFullscreen();
                }}
                function resetView(){{
                    if(typeof network !== 'undefined'){{ network.fit({{ animation: {{ duration: 600, easingFunction: 'easeInOutQuad' }} }}); }}
                }}
                function zoomRandomAttack(){{
                    if(typeof network !== 'undefined'){{
                        if(attackPairs.length > 0) {{
                            var idx = Math.floor(Math.random() * attackPairs.length);
                            network.fit({{ nodes: attackPairs[idx], animation: {{ duration: 800, easingFunction: 'easeInOutQuad' }} }});
                        }} else {{
                            alert('Nessuna anomalia (TP, FP, FN o Attacchi) visibile in questo time-slot!');
                        }}
                    }}
                }}
                </script>
                """
                html_content = html_content.replace('<body>', f'<body>\n{overlay_html}')
                components.html(html_content, height=780)

            st.markdown("---")
            if st.button("📸 Take Screenshot"):
                with st.spinner("Avvio del browser in background e acquisizione in corso..."):
                    try:
                        take_screenshot(html_file_path, png_file_path)
                        st.success("Screenshot salvato con successo!")
                        st.image(png_file_path, caption="Graph Screenshot", use_container_width=True)
                        if os.path.exists(png_file_path):
                            os.remove(png_file_path)
                    except Exception as e:
                        st.error(f"Si è verificato un errore durante lo screenshot: {e}")

            if os.path.exists(html_file_path):
                os.remove(html_file_path)

# ===========================================================================
# TAB 2: ABLATION STUDY / METRICS PANEL
# ===========================================================================
with tab2:
    st.markdown(
        "Analisi delle performance complessive delle varie configurazioni di grafo.")

    if not selected_configs:
        st.warning("Seleziona almeno una configurazione dalla barra laterale sinistra (Sezione 2).")
    else:
        filtered_df = df_metrics[df_metrics['Configuration'].isin(selected_configs)]

        # --- Row 1: Grouped Bar Chart ---
        st.subheader("Performance Metrics Comparison")
        df_melted = filtered_df.melt(id_vars='Configuration', var_name='Metric', value_name='Score')

        fig_bar = px.bar(
            df_melted,
            x='Metric',
            y='Score',
            color='Configuration',
            barmode='group',
            range_y=[0.0, 1.0],
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Raw Metrics Data")
        st.dataframe(
            filtered_df.style.format(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'], formatter="{:.4f}"),
            use_container_width=True)