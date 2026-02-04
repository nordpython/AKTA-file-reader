import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
import contextlib
import io
import proteovis as pv

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓ DE LA PÀGINA
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Akta Viewer Pro", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    .stSidebar {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Visualitzador de Cromatografia Akta")

# ───────────────────────────────────────────────────────────────────────────────
# FUNCIONS DE CÀRREGA DE DADES
# ───────────────────────────────────────────────────────────────────────────────
def _xy_from_series_value(val):
    def to_float_list(a):
        out = []
        for v in a:
            try:
                out.append(float(v))
            except Exception:
                return None
        return out
    if isinstance(val, dict) and 'x' in val and 'y' in val:
        x = to_float_list(val['x']); y = to_float_list(val['y'])
        if x is None or y is None or len(x) < 2 or len(x) != len(y): return None, None
        return np.asarray(x, float), np.asarray(y, float)
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            xs, ys = [], []
            for p in val:
                try:
                    xs.append(float(p[0])); ys.append(float(p[1]))
                except Exception: return None, None
            if len(xs) < 2: return None, None
            return np.asarray(xs, float), np.asarray(ys, float)
        if isinstance(first, dict) and 'x' in first and 'y' in first:
            xs = to_float_list([d['x'] for d in val]); ys = to_float_list([d['y'] for d in val])
            if xs is None or ys is None or len(xs) < 2 or len(xs) != len(ys): return None, None
            return np.asarray(xs, float), np.asarray(ys, float)
    return None, None

def carregar_fitxer(path):
    ext = os.path.splitext(path)[1].lower()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        if ext in (".zip", ".result"):
            data = pv.pycorn.load_uni_zip(path)
        elif ext == ".res":
            parsers = [getattr(pv.pycorn, n, None) for n in ("pc_res3","pc_res","pc_res2")]
            parsers = [p for p in parsers if p is not None]
            data = None
            for Parser in parsers:
                try:
                    obj = Parser(path); obj.load(); data = obj; break
                except Exception: pass
            if data is None: raise RuntimeError("No s'ha pogut llegir el .res")
        else:
            raise ValueError(f"Extensió no suportada: {ext}")

    curve_keys = []
    for k in data.keys():
        if k in ("Fractions","Method","Meta","System","Instrument"): continue
        series = data.get(k, {})
        if not isinstance(series, dict) or "data" not in series: continue
        x, y = _xy_from_series_value(series["data"])
        if x is None or y is None: continue
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 2: curve_keys.append(k)

    if not curve_keys: raise RuntimeError("No s'han trobat corbes vàlides.")

    x0, y0 = _xy_from_series_value(data[curve_keys[0]]["data"])
    df = pd.DataFrame({"mL": x0})
    for k in curve_keys:
        x, y = _xy_from_series_value(data[k]["data"])
        if x is None or y is None: continue
        if len(x) == len(df["mL"]) and np.allclose(x, df["mL"].values, equal_nan=True):
            df[k] = y
        else:
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 2: df[k] = np.interp(df["mL"].values, x[valid], y[valid])

    if "Fractions" in data and isinstance(data["Fractions"], dict) and "data" in data["Fractions"]:
        if "Fractions" not in df.columns: df["Fractions"] = pd.Series(dtype=object)
        for frac_ml, label in data["Fractions"]["data"]:
            try:
                idx = (df["mL"] - float(frac_ml)).abs().idxmin()
                df.at[idx, "Fractions"] = label
            except: pass
            
    if "UV 2_260" in df.columns and "UV 1_280" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["260/280"] = df["UV 2_260"] / df["UV 1_280"]

    return df, data, os.path.basename(path)

# ───────────────────────────────────────────────────────────────────────────────
# UI PRINCIPAL
# ───────────────────────────────────────────────────────────────────────────────

# 🟢 CANVI 1: 'accept_multiple_files=True' per permetre pujar-ne molts
uploaded_files = st.file_uploader("📂 Arrossega fitxers (.zip, .res, .result)", 
                                  type=['zip', 'res', 'result'], 
                                  accept_multiple_files=True)

if uploaded_files:
    # Llista per guardar tots els datasets carregats
    all_datasets = []
    
    # Bucle de càrrega
    for up_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up_file.name)[1]) as tmp:
            tmp.write(up_file.getvalue())
            tmp_path = tmp.name
        
        try:
            df, data, fname = carregar_fitxer(tmp_path)
            all_datasets.append({'name': fname, 'df': df})
        except Exception as e:
            st.error(f"Error carregant {up_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_datasets:
        st.stop()

    # ───────────────────────────────────────────────────────────────────────────
    # MODE 1: UN SOL FITXER (La teva visualització detallada original)
    # ───────────────────────────────────────────────────────────────────────────
    if len(all_datasets) == 1:
        dataset = all_datasets[0]
        df = dataset['df']
        file_name = dataset['name']
        
        # --- (Aquí va tota la lògica "Pro" que ja tenies) ---
        
        # 1. Recuperem memòria
        current_uv1_off = st.session_state.get('uv1_off', 0.0)
        current_uv2_off = st.session_state.get('uv2_off', 0.0)
        
        cols = list(df.columns)
        possibles_uv = [k for k in cols if "UV" in k.upper()]
        possibles_y2 = [k for k in cols if k not in possibles_uv and k not in ["mL", "Fractions", "260/280"]]
        
        # Auto-calc inicial
        calc_min_y, calc_max_y = 0.0, 100.0
        default_y1 = possibles_uv[0] if possibles_uv else None
        default_y2 = possibles_uv[2] if len(possibles_uv)>2 else (possibles_uv[1] if len(possibles_uv)>1 else None)
        
        temp_data = []
        if default_y1 in df.columns: temp_data.append(df[default_y1] + current_uv1_off)
        if default_y2 in df.columns: temp_data.append(df[default_y2] + current_uv2_off)
        if temp_data:
            combined = pd.concat(temp_data)
            calc_min_y = float(combined.min()) - 4.0
            calc_max_y = float(combined.max()) + 4.0
        
        ml_min_val, ml_max_val = float(df["mL"].min()), float(df["mL"].max())

        if 'ymin_input' not in st.session_state: st.session_state.ymin_input = calc_min_y
        if 'ymax_input' not in st.session_state: st.session_state.ymax_input = calc_max_y
        if 'xmin_input' not in st.session_state: st.session_state.xmin_input = ml_min_val
        if 'xmax_input' not in st.session_state: st.session_state.xmax_input = ml_max_val

        st.sidebar.header("⚙️ Configuració (Mode Detall)")
        
        with st.sidebar.expander("📊 Senyals i Colors", expanded=True):
            c1, c2 = st.columns(2)
            y1a_label = c1.selectbox("UV 1 (Principal)", options=possibles_uv, index=0 if possibles_uv else 0, key='sel_uv1')
            y1a_color = c2.color_picker("Color UV1", "#1f77b4", key='col_uv1')
            
            c3, c4 = st.columns(2)
            y1b_label = c3.selectbox("UV 2", options=[""] + possibles_uv, index=2 if len(possibles_uv)>1 else 0, key='sel_uv2')
            y1b_color = c4.color_picker("Color UV2", "#ff0000", key='col_uv2')
            
            c5, c6 = st.columns(2)
            y2_label = c5.selectbox("Eix Y Secundari", options=[""] + possibles_y2, key='sel_y2')
            y2_color = c6.color_picker("Color Y2", "#2ca02c", key='col_y2')

        with st.sidebar.expander("📏 Mides i Rangs (Zoom)", expanded=True):
            st.markdown("**Dimensions**")
            cd1, cd2 = st.columns(2)
            figwidth = cd1.number_input("Amplada", value=14, step=1, key='fig_w')
            figheight = cd2.number_input("Altura", value=6, step=1, key='fig_h')
            st.markdown("---")
            st.markdown("**Eix X (mL)**")
            col_x1, col_x2 = st.columns(2)
            xmin = col_x1.number_input("Mínim X", step=1.0, key='xmin_input')
            xmax = col_x2.number_input("Màxim X", step=1.0, key='xmax_input')
            x_tick_step = st.number_input("Pas dels Ticks X (mL)", value=5.0, min_value=0.1, step=0.5, key='x_step')
            st.markdown("---")
            st.markdown("**Eix Y (Absorbància)**")
            col_y1, col_y2 = st.columns(2)
            ymin = col_y1.number_input("Mínim Y", step=5.0, format="%.1f", key='ymin_input')
            ymax = col_y2.number_input("Màxim Y", step=5.0, format="%.1f", key='ymax_input')
            
            y2_ymin, y2_ymax = 0.0, 100.0
            if y2_label:
                st.markdown("**Eix Y Secundari**")
                c_y2_1, c_y2_2 = st.columns(2)
                if 'y2_max_input' not in st.session_state:
                     y2_curr_max = float(df[y2_label].max())
                     st.session_state.y2_max_input = y2_curr_max + 10.0
                y2_ymin = c_y2_1.number_input("Mínim Y2", value=0.0, key='y2_min_input')
                y2_ymax = c_y2_2.number_input("Màxim Y2", key='y2_max_input')

        with st.sidebar.expander("🧪 Fraccions", expanded=False):
            show_fractions = st.checkbox("Mostrar Fraccions", value=True, key='show_fracs')
            frac_step = st.number_input("Etiquetar cada N fraccions", value=1, min_value=1, step=1, key='frac_step')
            default_tick_h = (ymax - ymin) * 0.1
            tick_h = st.slider("Alçada marca vermella", 1.0, 300.0, float(default_tick_h) if default_tick_h > 0 else 10.0, key='frac_h')
            frac_lw = st.slider("Gruix línia", 0.2, 5.0, 1.0, key='frac_lw')
            label_offset = st.number_input("Posició Text (Vertical)", min_value=0.0, value=2.0, step=0.5, key='frac_offset')
            font_frac = st.slider("Mida Text Fracció", 6, 20, 9, key='frac_font')

        with st.sidebar.expander("🎨 Estils de Text", expanded=False):
            font_title = st.slider("Mida Títol", 10, 40, 16, key='f_title')
            font_labels = st.slider("Mida Etiquetes Eixos", 8, 30, 12, key='f_labels')
            font_ticks = st.slider("Mida Números Eixos", 8, 20, 10, key='f_ticks')
            font_legend = st.slider("Mida Llegenda", 8, 20, 10, key='f_legend')

        with st.sidebar.expander("🛠️ Extres (Offsets)", expanded=False):
            uv1_offset = st.number_input("Offset UV1 (mAU)", step=0.5, key='uv1_off')
            uv2_offset = st.number_input("Offset UV2 (mAU)", step=0.5, key='uv2_off')

        # PLOT SINGLE
        fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
        if y1a_label and y1a_label in df.columns:
            ax1.plot(df["mL"], df[y1a_label] + uv1_offset, label=y1a_label, color=y1a_color)
        if y1b_label and y1b_label in df.columns and y1b_label != y1a_label:
            ax1.plot(df["mL"], df[y1b_label] + uv2_offset, label=y1b_label, color=y1b_color)
        
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Elution volume (mL)", fontsize=font_labels)
        ax1.set_ylabel("Absorbance (mAU)", fontsize=font_labels)
        ax1.tick_params(axis='both', labelsize=font_ticks)
        ax1.set_title(f"Chromatogram – {file_name}", fontsize=font_title)
        if x_tick_step > 0: ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))

        if show_fractions and "Fractions" in df.columns:
            fractions = df[(df['Fractions'].notna()) & (df['mL'].between(xmin, xmax))].reset_index()
            for i in range(len(fractions)):
                x = fractions.loc[i, 'mL']
                label = fractions.loc[i, 'Fractions']
                ax1.vlines(x, ymin, ymin + tick_h, color='red', linewidth=frac_lw, zorder=5)
                if i % frac_step == 0:
                    txt = 'W' if str(label).lower() == 'waste' else str(label)
                    ax1.text(x, ymin + tick_h + label_offset, txt, ha='center', va='bottom', fontsize=font_frac, color='black', clip_on=False, zorder=6)

        ax2 = None
        if y2_label and y2_label in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df["mL"], df[y2_label], label=y2_label, color=y2_color, linestyle="--")
            ax2.set_ylabel(y2_label, fontsize=font_labels)
            ax2.tick_params(axis='y', labelsize=font_ticks)
            ax2.set_ylim(y2_ymin, y2_ymax)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=font_legend)
        st.pyplot(fig)
        with st.expander("📋 Veure Dades en Taula"): st.dataframe(df)

    # ───────────────────────────────────────────────────────────────────────────
    # MODE 2: MÚLTIPLES FITXERS (COMPARATIVA)
    # ───────────────────────────────────────────────────────────────────────────
    else:
        st.success(f"Mode Comparació activat: {len(all_datasets)} fitxers carregats.")
        
        # Agafem les columnes del primer fitxer com a referència
        ref_df = all_datasets[0]['df']
        ref_cols = list(ref_df.columns)
        possibles_uv_comp = [k for k in ref_cols if "UV" in k.upper()]
        
        st.sidebar.header("⚙️ Configuració (Comparació)")
        
        # Selecció del senyal a comparar (ex: UV 280)
        signal_to_compare = st.sidebar.selectbox("Quin senyal vols comparar?", options=possibles_uv_comp, index=0)
        
        with st.sidebar.expander("📏 Mides i Rangs", expanded=True):
            st.markdown("**Dimensions**")
            cd1, cd2 = st.columns(2)
            figwidth_c = cd1.number_input("Amplada", value=14, step=1, key='fig_w_comp')
            figheight_c = cd2.number_input("Altura", value=6, step=1, key='fig_h_comp')
            
            # Càlcul automàtic de límits globals
            all_max_x = max([d['df']['mL'].max() for d in all_datasets])
            all_min_x = min([d['df']['mL'].min() for d in all_datasets])
            
            # Busquem el Y max de tots els datasets pel senyal triat
            max_y_vals = []
            for d in all_datasets:
                if signal_to_compare in d['df'].columns:
                    max_y_vals.append(d['df'][signal_to_compare].max())
            all_max_y = max(max_y_vals) if max_y_vals else 100
            
            col_xc1, col_xc2 = st.columns(2)
            xmin_c = col_xc1.number_input("Mínim X", value=float(all_min_x), step=1.0, key='xmin_c')
            xmax_c = col_xc2.number_input("Màxim X", value=float(all_max_x), step=1.0, key='xmax_c')
            
            col_yc1, col_yc2 = st.columns(2)
            ymin_c = col_yc1.number_input("Mínim Y", value=-10.0, step=5.0, key='ymin_c')
            ymax_c = col_yc2.number_input("Màxim Y", value=float(all_max_y)+10, step=5.0, key='ymax_c')

        # PLOT COMPARATIU
        fig, ax = plt.subplots(figsize=(figwidth_c, figheight_c))
        
        # Bucle per pintar cada fitxer
        for dataset in all_datasets:
            dname = dataset['name']
            ddf = dataset['df']
            
            if signal_to_compare in ddf.columns:
                ax.plot(ddf["mL"], ddf[signal_to_compare], label=dname, alpha=0.8)
            else:
                st.warning(f"El fitxer {dname} no té el senyal {signal_to_compare}")

        ax.set_xlim(xmin_c, xmax_c)
        ax.set_ylim(ymin_c, ymax_c)
        ax.set_xlabel("Elution volume (mL)", fontsize=12)
        ax.set_ylabel(f"{signal_to_compare} (mAU)", fontsize=12)
        ax.set_title(f"Comparativa – {signal_to_compare}", fontsize=16)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig)
