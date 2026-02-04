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

# ───────────────────────────────────────────────────────────────────────────────
# 🚑 NUMPY 2.0 PATCH
# ───────────────────────────────────────────────────────────────────────────────
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

import proteovis as pv

# ───────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Akta Viewer Pro", layout="wide", page_icon="🧬")

st.markdown("""
<style>
    .stSidebar { background-color: #f0f2f6; }
    .main .block-container { padding-top: 2rem; }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center;
    }
    /* Custom style for legend inputs */
    .stTextInput > label { font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

# TITOL NET (Sense "Full Version")
st.title("🧬 Akta Chromatogram Viewer")

# ───────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────
def _xy_from_series_value(val):
    def to_float_list(a):
        out = []
        for v in a:
            try: out.append(float(v))
            except: return None
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
                try: xs.append(float(p[0])); ys.append(float(p[1]))
                except: return None, None
            if len(xs) < 2: return None, None
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
                try: obj = Parser(path); obj.load(); data = obj; break
                except: pass
            if data is None: raise RuntimeError("Could not read .res file")
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    curve_keys = []
    for k in data.keys():
        if k in ("Fractions","Method","Meta","System","Instrument"): continue
        series = data.get(k, {})
        if not isinstance(series, dict) or "data" not in series: continue
        x, y = _xy_from_series_value(series["data"])
        if x is None or y is None: continue
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 2: curve_keys.append(k)

    if not curve_keys: raise RuntimeError("No valid curves found.")

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
# 1. FILE UPLOADER (MAIN AREA)
# ───────────────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader("📂 Drag and drop files here (.zip, .res, .result)", type=['zip', 'res', 'result'], accept_multiple_files=True)

# ───────────────────────────────────────────────────────────────────────────────
# DATA LOADING & MODE SELECTION
# ───────────────────────────────────────────────────────────────────────────────
if uploaded_files:
    
    # Mode Selector
    mode = st.sidebar.radio("Analysis Mode", ["📊 Detailed Analysis (Single)", "📈 Multi-File Comparison (Overlay)"])
    
    # ───────────────────────────────────────────────────────────────────────────
    # MODE 1: SINGLE FILE
    # ───────────────────────────────────────────────────────────────────────────
    if mode == "📊 Detailed Analysis (Single)":
        
        # File Selection
        file_names = [f.name for f in uploaded_files]
        selected_name = st.sidebar.selectbox("Select File to Analyze", file_names)
        target_file = next(f for f in uploaded_files if f.name == selected_name)

        # Load
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(target_file.name)[1]) as tmp:
            tmp.write(target_file.getvalue())
            tmp_path = tmp.name

        try:
            df, data, real_filename = carregar_fitxer(tmp_path)
            
            # --- Memory & Defaults ---
            cols = list(df.columns)
            possibles_uv = [k for k in cols if "UV" in k.upper()]
            possibles_y2 = [k for k in cols if k not in possibles_uv and k not in ["mL", "Fractions", "260/280"]]
            
            # Init Memory
            if 'uv1_off' not in st.session_state: st.session_state.uv1_off = 0.0
            if 'uv2_off' not in st.session_state: st.session_state.uv2_off = 0.0
            
            # Auto-Calc for Memory
            if 'ymin_input' not in st.session_state:
                # Simple logic for default
                val_max = df[possibles_uv[0]].max() if possibles_uv else 100
                st.session_state.ymin_input = 0.0
                st.session_state.ymax_input = float(val_max) + 50.0
                st.session_state.xmin_input = float(df["mL"].min())
                st.session_state.xmax_input = float(df["mL"].max())

            # --- SIDEBAR (SINGLE) ---
            st.sidebar.markdown("---")
            with st.sidebar.expander("📊 Signals & Colors", expanded=True):
                c1, c2 = st.columns(2)
                y1a_label = c1.selectbox("UV 1 (Main)", options=possibles_uv, index=0, key='s_uv1')
                y1a_color = c2.color_picker("Color", "#1f77b4", key='c_uv1')
                c3, c4 = st.columns(2)
                y1b_label = c3.selectbox("UV 2", options=[""] + possibles_uv, index=2 if len(possibles_uv)>1 else 0, key='s_uv2')
                y1b_color = c4.color_picker("Color", "#ff0000", key='c_uv2')
                c5, c6 = st.columns(2)
                y2_label = c5.selectbox("Secondary Y", options=[""] + possibles_y2, key='s_y2')
                y2_color = c6.color_picker("Color", "#2ca02c", key='c_y2')

            with st.sidebar.expander("📏 Dimensions & Ranges", expanded=True):
                cd1, cd2 = st.columns(2)
                figwidth = cd1.number_input("Width", value=14, step=1, key='s_fw')
                figheight = cd2.number_input("Height", value=6, step=1, key='s_fh')
                st.markdown("---")
                col_x1, col_x2 = st.columns(2)
                xmin = col_x1.number_input("Min X", step=1.0, key='xmin_input')
                xmax = col_x2.number_input("Max X", step=1.0, key='xmax_input')
                x_tick_step = st.number_input("X Step", value=5.0, min_value=0.1, key='s_xstep')
                col_y1, col_y2 = st.columns(2)
                ymin = col_y1.number_input("Min Y", step=5.0, format="%.1f", key='ymin_input')
                ymax = col_y2.number_input("Max Y", step=5.0, format="%.1f", key='ymax_input')
                
                y2_ymin, y2_ymax = 0.0, 100.0
                if y2_label:
                    if 'y2_max' not in st.session_state: st.session_state.y2_max = float(df[y2_label].max()) + 10.0
                    c_y2_1, c_y2_2 = st.columns(2)
                    y2_ymin = c_y2_1.number_input("Min Y2", value=0.0, key='y2_min')
                    y2_ymax = c_y2_2.number_input("Max Y2", key='y2_max')

            with st.sidebar.expander("🧪 Fractions", expanded=False):
                show_fractions = st.checkbox("Show Fractions", value=True, key='s_frac')
                frac_step = st.number_input("Label every N", value=1, min_value=1, key='s_fstep')
                tick_h = st.slider("Line Height", 1.0, 300.0, 1.0, key='s_fh_val') # DEFAULT 1.0
                label_offset = st.number_input("Text Pos", value=0.0, step=0.5, key='s_foff') # DEFAULT 0.0
                font_frac = st.slider("Font Size", 6, 20, 9, key='s_ffont')

            with st.sidebar.expander("🎨 Styles & Offsets", expanded=False):
                plot_title = st.text_input("Title", value=f"Chromatogram – {real_filename}")
                font_title = st.slider("Title Size", 10, 40, 16, key='s_ftitle')
                font_legend = st.slider("Legend Size", 8, 20, 10, key='s_fleg')
                st.markdown("---")
                uv1_offset = st.number_input("Offset UV1", step=0.5, key='uv1_off')
                uv2_offset = st.number_input("Offset UV2", step=0.5, key='uv2_off')

            # --- PLOT SINGLE ---
            fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
            if y1a_label in df.columns: ax1.plot(df["mL"], df[y1a_label] + uv1_offset, label=y1a_label, color=y1a_color)
            if y1b_label in df.columns and y1b_label != y1a_label: ax1.plot(df["mL"], df[y1b_label] + uv2_offset, label=y1b_label, color=y1b_color)

            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_xlabel("Elution volume (mL)")
            ax1.set_ylabel("Absorbance (mAU)")
            ax1.set_title(plot_title, fontsize=font_title)
            if x_tick_step > 0: ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))

            if show_fractions and "Fractions" in df.columns:
                fractions = df[(df['Fractions'].notna()) & (df['mL'].between(xmin, xmax))].reset_index()
                for i in range(len(fractions)):
                    x = fractions.loc[i, 'mL']
                    label = fractions.loc[i, 'Fractions']
                    ax1.vlines(x, ymin, ymin + tick_h, color='red', linewidth=1, zorder=5)
                    if i % frac_step == 0:
                        txt = 'W' if str(label).lower() == 'waste' else str(label)
                        ax1.text(x, ymin + tick_h + label_offset, txt, ha='center', va='bottom', fontsize=font_frac, color='black', clip_on=False, zorder=6)

            ax2 = None
            if y2_label and y2_label in df.columns:
                ax2 = ax1.twinx()
                ax2.plot(df["mL"], df[y2_label], label=y2_label, color=y2_color, linestyle="--")
                ax2.set_ylabel(y2_label)
                ax2.set_ylim(y2_ymin, y2_ymax)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=font_legend)
            st.pyplot(fig)

            # --- CALCULATIONS ---
            with st.expander("🧮 Calculations", expanded=True):
                c_c1, c_c2 = st.columns([1, 2])
                with c_c1:
                    st.markdown("#### Parameters")
                    int_start = st.number_input("Start", value=xmin, step=0.5)
                    int_end = st.number_input("End", value=xmax, step=0.5)
                    target_sig = st.selectbox("Signal", [y1a_label, y1b_label])
                    base_mode = st.selectbox("Baseline", ["None", "Linear (Start-End)"])
                    st.markdown("#### Protein")
                    path_l = st.number_input("Path (cm)", value=0.2, format="%.2f")
                    c_type = st.radio("Type", ["Abs 0.1%", "Molar"], help="Sum Molar for complexes.")
                    if c_type == "Abs 0.1%":
                        e_mass = st.number_input("Abs 0.1%", value=1.0, format="%.3f")
                        e_molar = None
                    else:
                        e_molar = st.number_input("Molar", value=50000.0, format="%.1f")
                        e_mass = None
                    mw = st.number_input("MW (Da)", value=10000.0, format="%.1f")
                    decs = st.number_input("Decimals", value=4, min_value=1, max_value=8)

                with c_c2:
                    if target_sig in df.columns:
                        sub = df[(df["mL"] >= int_start) & (df["mL"] <= int_end)].copy()
                        if not sub.empty:
                            y_vals = sub[target_sig].values + (uv1_offset if target_sig==y1a_label else uv2_offset)
                            if base_mode == "Linear (Start-End)":
                                slope = (y_vals[-1] - y_vals[0]) / (sub["mL"].values[-1] - sub["mL"].values[0])
                                base = y_vals[0] + slope * (sub["mL"].values - sub["mL"].values[0])
                                y_proc = y_vals - base
                            else:
                                y_proc = y_vals
                            
                            # Calcs
                            avg_au = np.mean(y_proc) / 1000.0
                            area_au = np.trapz(y_proc, sub["mL"].values) / 1000.0
                            
                            c_mg = 0.0
                            if path_l > 0:
                                if c_type == "Abs 0.1%": c_mg = avg_au / (e_mass * path_l)
                                else: c_mg = (avg_au / (e_molar * path_l)) * mw
                            
                            c_um = (c_mg / mw) * 1e6 if mw > 0 else 0.0
                            
                            m_mg = 0.0
                            if path_l > 0:
                                if c_type == "Abs 0.1%": m_mg = area_au / (e_mass * path_l)
                                else: m_mg = (area_au * mw) / (e_molar * path_l)

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Mass", f"{m_mg:.{decs}f} mg")
                            c2.metric("Vol", f"{int_end-int_start:.2f} mL")
                            c3.metric("Conc.", f"{c_mg:.{decs}f} mg/mL")
                            c4.metric("Conc.", f"{c_um:.{decs}f} µM")

                            st.markdown("#### Fraction Details")
                            if "Fractions" in df.columns:
                                f_idxs = df[df['Fractions'].notna()].index
                                f_list = []
                                for i in range(len(f_idxs)):
                                    idx_s = f_idxs[i]
                                    idx_e = f_idxs[i+1] if i < len(f_idxs)-1 else df.index[-1]
                                    ov_s = max(df.loc[idx_s, "mL"], int_start)
                                    ov_e = min(df.loc[idx_e, "mL"], int_end)
                                    if ov_s < ov_e:
                                        f_sub = df[(df["mL"] >= ov_s) & (df["mL"] <= ov_e)]
                                        if not f_sub.empty:
                                            fy = f_sub[target_sig].values + (uv1_offset if target_sig==y1a_label else uv2_offset)
                                            if base_mode == "Linear (Start-End)":
                                                fb = y_vals[0] + slope * (f_sub["mL"].values - sub["mL"].values[0])
                                                fy = fy - fb
                                            
                                            f_au = np.mean(fy) / 1000.0
                                            f_cmg = 0.0
                                            if path_l > 0:
                                                if c_type == "Abs 0.1%": f_cmg = f_au / (e_mass * path_l)
                                                else: f_cmg = (f_au / (e_molar * path_l)) * mw
                                            f_cum = (f_cmg / mw) * 1e6 if mw > 0 else 0.0
                                            
                                            f_list.append({
                                                "Fraction": df.loc[idx_s, "Fractions"],
                                                "Vol (mL)": f"{ov_e-ov_s:.2f}",
                                                "mAU": f"{np.mean(fy):.1f}",
                                                "mg/mL": f"{f_cmg:.{decs}f}",
                                                "µM": f"{f_cum:.{decs}f}"
                                            })
                                if f_list: st.dataframe(pd.DataFrame(f_list), use_container_width=True)
                                else: st.info("No fractions.")
        except Exception as e: st.error(f"Error: {e}")
        finally: os.remove(tmp_path)

    # ───────────────────────────────────────────────────────────────────────────
    # MODE 2: MULTI-FILE OVERLAY (MILLORAT)
    # ───────────────────────────────────────────────────────────────────────────
    elif mode == "📈 Multi-File Comparison (Overlay)":
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Overlay Settings")
        
        # 1. Carreguem tots els fitxers
        loaded_dfs = []
        possible_signals = set()
        
        with st.spinner("Loading files..."):
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                try:
                    df, _, _ = carregar_fitxer(tmp_path)
                    loaded_dfs.append({"name": f.name, "df": df})
                    for col in df.columns:
                        if "UV" in col.upper() or "Cond" in col or "Conc" in col:
                            possible_signals.add(col)
                except: pass
                finally: os.remove(tmp_path)
        
        if not loaded_dfs:
            st.warning("No valid files loaded.")
        else:
            sorted_sig = sorted(list(possible_signals))
            
            # --- OVERLAY CONTROLS ---
            with st.sidebar.expander("📊 Signals Selection", expanded=True):
                # Eix Y1 (Esquerra)
                y1_sig = st.selectbox("Left Axis (Y1)", sorted_sig, index=0 if sorted_sig else 0)
                # Eix Y2 (Dreta) - Opcional
                y2_sig = st.selectbox("Right Axis (Y2 - Optional)", ["None"] + sorted_sig, index=0)
            
            with st.sidebar.expander("📏 Ranges & Options", expanded=True):
                # Rangs X
                all_min_x = min([d["df"]["mL"].min() for d in loaded_dfs])
                all_max_x = max([d["df"]["mL"].max() for d in loaded_dfs])
                
                c_rx1, c_rx2 = st.columns(2)
                mx_min = c_rx1.number_input("Min X", value=float(all_min_x), step=1.0)
                mx_max = c_rx2.number_input("Max X", value=float(all_max_x), step=1.0)
                
                # Rangs Y Manuals
                auto_y = st.checkbox("Auto Y-Scale", value=True)
                if not auto_y:
                    c_ry1, c_ry2 = st.columns(2)
                    my_min = c_ry1.number_input("Min Y", value=0.0)
                    my_max = c_ry2.number_input("Max Y", value=100.0)
                
                normalize = st.checkbox("Normalize Baseline (Start at 0)", value=True)
                line_width = st.slider("Line Width", 0.5, 3.0, 1.5)
                alpha = st.slider("Transparency", 0.1, 1.0, 0.8)

            with st.sidebar.expander("📝 Edit Legend Names", expanded=False):
                # Permet canviar el nom de cada fitxer per a la llegenda
                custom_names = {}
                for item in loaded_dfs:
                    orig = item["name"]
                    custom_names[orig] = st.text_input(f"Name for {orig}", value=orig)

            # --- PLOT OVERLAY ---
            st.markdown("### Comparison Chart")
            
            # Dimensions
            f_w = st.sidebar.number_input("Width", 14, key="mw")
            f_h = st.sidebar.number_input("Height", 6, key="mh")
            
            fig, ax1 = plt.subplots(figsize=(f_w, f_h))
            
            # Paleta de colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(loaded_dfs)))
            
            # Loop per fitxers (Eix Esquerre)
            for i, item in enumerate(loaded_dfs):
                df = item["df"]
                label = custom_names[item["name"]]
                
                if y1_sig in df.columns:
                    y_data = df[y1_sig].values
                    if normalize: y_data = y_data - y_data[0]
                    
                    ax1.plot(df["mL"], y_data, label=label, color=colors[i], linewidth=line_width, alpha=alpha)
            
            ax1.set_xlim(mx_min, mx_max)
            if not auto_y: ax1.set_ylim(my_min, my_max)
            
            ax1.set_xlabel("Elution Volume (mL)")
            ax1.set_ylabel(f"{y1_sig} (mAU)")
            ax1.tick_params(axis='y')
            
            # Eix Dret (Si està seleccionat)
            ax2 = None
            if y2_sig != "None":
                ax2 = ax1.twinx()
                # Loop per fitxers (Eix Dret)
                for i, item in enumerate(loaded_dfs):
                    df = item["df"]
                    # No posem label al segon eix per no duplicar llegenda, o usem linia discontinua
                    if y2_sig in df.columns:
                        y2_data = df[y2_sig].values
                        if normalize: y2_data = y2_data - y2_data[0]
                        
                        ax2.plot(df["mL"], y2_data, color=colors[i], linestyle="--", linewidth=1, alpha=0.6)
                
                ax2.set_ylabel(f"{y2_sig} (Dashed)")
            
            # Llegenda (només del primer eix per no fer un embolic, ja que els colors coincideixen)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Overlay: {y1_sig} " + (f"vs {y2_sig}" if y2_sig != "None" else ""))
            
            st.pyplot(fig)
            st.info(f"Displaying {len(loaded_dfs)} files. Solid lines: {y1_sig}. Dashed lines: {y2_sig}.")

else:
    st.info("👆 Please upload files to start.")
