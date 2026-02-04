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
</style>
""", unsafe_allow_html=True)

st.title("🧬 Akta Chromatogram Viewer (Full Version)")

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

    if not curve_keys: raise RuntimeError("No valid curves found in data.")

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
# MAIN UI
# ───────────────────────────────────────────────────────────────────────────────

# 1. FILE UPLOADER (Supports Multiple)
uploaded_files = st.sidebar.file_uploader("📂 Upload Files (.zip, .res)", type=['zip', 'res', 'result'], accept_multiple_files=True)

# 2. MODE SELECTOR
mode = st.sidebar.radio("Analysis Mode", ["📊 Detailed Analysis (Single)", "📈 Multi-File Comparison (Overlay)"])

if uploaded_files:
    
    # ───────────────────────────────────────────────────────────────────────────
    # MODE 1: SINGLE FILE (DETAILED)
    # ───────────────────────────────────────────────────────────────────────────
    if mode == "📊 Detailed Analysis (Single)":
        
        # Selector de fitxer si n'hi ha més d'un
        names = [f.name for f in uploaded_files]
        selected_name = st.sidebar.selectbox("Select File to Analyze", names)
        
        # Busquem l'objecte file correcte
        target_file = next(f for f in uploaded_files if f.name == selected_name)

        # Càrrega (Codi anterior millorat)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(target_file.name)[1]) as tmp:
            tmp.write(target_file.getvalue())
            tmp_path = tmp.name

        try:
            df, data, real_filename = carregar_fitxer(tmp_path)
            
            # --- Memory & Defaults ---
            current_uv1_off = st.session_state.get('uv1_off', 0.0)
            current_uv2_off = st.session_state.get('uv2_off', 0.0)
            
            cols = list(df.columns)
            possibles_uv = [k for k in cols if "UV" in k.upper()]
            possibles_y2 = [k for k in cols if k not in possibles_uv and k not in ["mL", "Fractions", "260/280"]]
            
            # Auto-Calc Ranges
            calc_min_y, calc_max_y = 0.0, 100.0
            default_y1 = possibles_uv[0] if possibles_uv else None
            temp_data = []
            if default_y1 in df.columns: temp_data.append(df[default_y1] + current_uv1_off)
            if temp_data:
                combined = pd.concat(temp_data)
                calc_min_y = float(combined.min()) - 4.0
                calc_max_y = float(combined.max()) + 4.0
            
            ml_min_val, ml_max_val = float(df["mL"].min()), float(df["mL"].max())

            if 'ymin_input' not in st.session_state: st.session_state.ymin_input = calc_min_y
            if 'ymax_input' not in st.session_state: st.session_state.ymax_input = calc_max_y
            if 'xmin_input' not in st.session_state: st.session_state.xmin_input = ml_min_val
            if 'xmax_input' not in st.session_state: st.session_state.xmax_input = ml_max_val

            # Sidebar Controls (Single)
            st.sidebar.markdown("---")
            with st.sidebar.expander("📊 Signals & Colors", expanded=True):
                c1, c2 = st.columns(2)
                y1a_label = c1.selectbox("UV 1 (Main)", options=possibles_uv, index=0, key='sel_uv1')
                y1a_color = c2.color_picker("Color", "#1f77b4", key='col_uv1')
                c3, c4 = st.columns(2)
                y1b_label = c3.selectbox("UV 2", options=[""] + possibles_uv, index=2 if len(possibles_uv)>1 else 0, key='sel_uv2')
                y1b_color = c4.color_picker("Color", "#ff0000", key='col_uv2')
                c5, c6 = st.columns(2)
                y2_label = c5.selectbox("Secondary Y", options=[""] + possibles_y2, key='sel_y2')
                y2_color = c6.color_picker("Color", "#2ca02c", key='col_y2')

            with st.sidebar.expander("📏 Dimensions & Ranges", expanded=True):
                cd1, cd2 = st.columns(2)
                figwidth = cd1.number_input("Width", value=14, step=1, key='fig_w')
                figheight = cd2.number_input("Height", value=6, step=1, key='fig_h')
                st.markdown("---")
                col_x1, col_x2 = st.columns(2)
                xmin = col_x1.number_input("Min X", step=1.0, key='xmin_input')
                xmax = col_x2.number_input("Max X", step=1.0, key='xmax_input')
                x_tick_step = st.number_input("X Step", value=5.0, min_value=0.1, key='x_step')
                col_y1, col_y2 = st.columns(2)
                ymin = col_y1.number_input("Min Y", step=5.0, format="%.1f", key='ymin_input')
                ymax = col_y2.number_input("Max Y", step=5.0, format="%.1f", key='ymax_input')
                y2_ymin, y2_ymax = 0.0, 100.0
                if y2_label:
                    if 'y2_max_input' not in st.session_state: st.session_state.y2_max_input = float(df[y2_label].max()) + 10.0
                    c_y2_1, c_y2_2 = st.columns(2)
                    y2_ymin = c_y2_1.number_input("Min Y2", value=0.0, key='y2_min_input')
                    y2_ymax = c_y2_2.number_input("Max Y2", key='y2_max_input')

            with st.sidebar.expander("🧪 Fractions", expanded=False):
                show_fractions = st.checkbox("Show Fractions", value=True, key='show_fracs')
                frac_step = st.number_input("Label every N", value=1, min_value=1, key='frac_step')
                tick_h = st.slider("Line Height", 1.0, 300.0, 1.0, key='frac_h')
                label_offset = st.number_input("Text Pos", min_value=0.0, value=0.0, step=0.5, key='frac_offset')
                font_frac = st.slider("Font Size", 6, 20, 9, key='frac_font')

            with st.sidebar.expander("🎨 Styles", expanded=False):
                plot_title = st.text_input("Title", value=f"Chromatogram – {real_filename}")
                font_title = st.slider("Title Size", 10, 40, 16, key='f_title')
                font_labels = st.slider("Labels Size", 8, 30, 12, key='f_labels')
                font_ticks = st.slider("Ticks Size", 8, 20, 10, key='f_ticks')
                font_legend = st.slider("Legend Size", 8, 20, 10, key='f_legend')

            with st.sidebar.expander("🛠️ Extras", expanded=False):
                uv1_offset = st.number_input("Offset UV1", step=0.5, key='uv1_off')
                uv2_offset = st.number_input("Offset UV2", step=0.5, key='uv2_off')

            # --- PLOTTING (Single) ---
            fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
            if y1a_label in df.columns: ax1.plot(df["mL"], df[y1a_label] + uv1_offset, label=y1a_label, color=y1a_color)
            if y1b_label in df.columns and y1b_label != y1a_label: ax1.plot(df["mL"], df[y1b_label] + uv2_offset, label=y1b_label, color=y1b_color)

            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_xlabel("Elution volume (mL)", fontsize=font_labels)
            ax1.set_ylabel("Absorbance (mAU)", fontsize=font_labels)
            ax1.tick_params(axis='both', labelsize=font_ticks)
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
                ax2.set_ylabel(y2_label, fontsize=font_labels)
                ax2.tick_params(axis='y', labelsize=font_ticks)
                ax2.set_ylim(y2_ymin, y2_ymax)

            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=font_legend)
            st.pyplot(fig)

            # --- CALCULATIONS (Single) ---
            with st.expander("🧮 Calculations & Peak Integration", expanded=True):
                col_calc1, col_calc2 = st.columns([1, 2])
                with col_calc1:
                    st.markdown("#### 1. Integration")
                    int_start = st.number_input("Start (mL)", value=xmin, step=0.5)
                    int_end = st.number_input("End (mL)", value=xmax, step=0.5)
                    target_signal = st.selectbox("Signal", [y1a_label, y1b_label])
                    baseline_mode = st.selectbox("Baseline", ["None", "Linear (Start-End)"])
                    st.markdown("#### 2. Protein Data")
                    path_length = st.number_input("Path Length (cm)", value=0.2, format="%.2f")
                    coeff_type = st.radio("Coeff Type", ["Abs 0.1% (1 g/L)", "Molar (M⁻¹ cm⁻¹)"], help="For complexes: Sum Molar values.")
                    if coeff_type == "Abs 0.1% (1 g/L)":
                        ext_coeff_mass = st.number_input("Abs 0.1% Value", value=1.0, format="%.3f")
                        ext_coeff_molar = None
                    else:
                        ext_coeff_molar = st.number_input("Molar Value (ε)", value=50000.0, format="%.1f")
                        ext_coeff_mass = None
                    mol_weight = st.number_input("MW (Da)", value=10000.0, format="%.1f")
                    decimals = st.number_input("Decimals", value=4, min_value=1, max_value=8)

                with col_calc2:
                    if target_signal and target_signal in df.columns:
                        mask = (df["mL"] >= int_start) & (df["mL"] <= int_end)
                        sub_df = df[mask].copy()
                        if not sub_df.empty:
                            x_vals = sub_df["mL"].values
                            offset_val = uv1_offset if target_signal == y1a_label else uv2_offset
                            y_vals = sub_df[target_signal].values + offset_val
                            if baseline_mode == "Linear (Start-End)":
                                slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                                baseline = y_vals[0] + slope * (x_vals - x_vals[0])
                                y_processed = y_vals - baseline
                            else:
                                y_processed = y_vals
                                baseline = np.zeros_like(y_vals)
                            
                            avg_AU = np.mean(y_processed) / 1000.0
                            conc_mg_ml, conc_uM, mass_mg = 0.0, 0.0, 0.0
                            
                            if path_length > 0:
                                if coeff_type == "Abs 0.1% (1 g/L)" and ext_coeff_mass > 0:
                                    conc_mg_ml = avg_AU / (ext_coeff_mass * path_length)
                                elif coeff_type == "Molar (M⁻¹ cm⁻¹)" and ext_coeff_molar > 0 and mol_weight > 0:
                                    conc_mg_ml = (avg_AU / (ext_coeff_molar * path_length)) * mol_weight
                                if mol_weight > 0: conc_uM = (conc_mg_ml / mol_weight) * 1e6
                            
                            area_AU_mL = np.trapz(y_processed, x_vals) / 1000.0
                            if path_length > 0:
                                if coeff_type == "Abs 0.1% (1 g/L)" and ext_coeff_mass > 0:
                                    mass_mg = area_AU_mL / (ext_coeff_mass * path_length)
                                elif coeff_type == "Molar (M⁻¹ cm⁻¹)" and ext_coeff_molar > 0:
                                    mass_mg = (area_AU_mL * mol_weight) / (ext_coeff_molar * path_length)
                            
                            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                            c_res1.metric("Total Mass", f"{mass_mg:.{decimals}f} mg")
                            c_res2.metric("Peak Vol", f"{int_end - int_start:.2f} mL")
                            c_res3.metric("Avg Conc", f"{conc_mg_ml:.{decimals}f} mg/mL")
                            c_res4.metric("Avg Conc", f"{conc_uM:.{decimals}f} µM")

                            st.markdown("#### 🧪 Fractions in Peak")
                            if "Fractions" in df.columns:
                                frac_indices = df[df['Fractions'].notna()].index
                                frac_data = []
                                for i in range(len(frac_indices)):
                                    idx_s = frac_indices[i]
                                    idx_e = frac_indices[i+1] if i < len(frac_indices)-1 else df.index[-1]
                                    overlap_s = max(df.loc[idx_s, "mL"], int_start)
                                    overlap_e = min(df.loc[idx_e, "mL"], int_end)
                                    if overlap_s < overlap_e:
                                        f_sub = df[(df["mL"] >= overlap_s) & (df["mL"] <= overlap_e)]
                                        if not f_sub.empty:
                                            f_y = f_sub[target_signal].values + offset_val
                                            if baseline_mode == "Linear (Start-End)":
                                                f_base = y_vals[0] + slope * (f_sub["mL"].values - x_vals[0])
                                                f_y = f_y - f_base
                                            f_conc_mg = 0.0
                                            f_conc_uM = 0.0
                                            f_avg_AU = np.mean(f_y) / 1000.0
                                            if path_length > 0:
                                                if coeff_type == "Abs 0.1% (1 g/L)" and ext_coeff_mass > 0: f_conc_mg = f_avg_AU/(ext_coeff_mass*path_length)
                                                elif coeff_type == "Molar (M⁻¹ cm⁻¹)" and ext_coeff_molar > 0: f_conc_mg = (f_avg_AU/(ext_coeff_molar*path_length))*mol_weight
                                                if mol_weight > 0: f_conc_uM = (f_conc_mg/mol_weight)*1e6
                                            
                                            frac_data.append({
                                                "Frac": df.loc[idx_s, "Fractions"],
                                                "Vol": f"{overlap_e - overlap_s:.2f}",
                                                "mAU": f"{np.mean(f_y):.1f}",
                                                "mg/mL": f"{f_conc_mg:.{decimals}f}",
                                                "µM": f"{f_conc_uM:.{decimals}f}"
                                            })
                                if frac_data: st.dataframe(pd.DataFrame(frac_data), use_container_width=True)
                                else: st.info("No fractions inside integration range.")
        except Exception as e: st.error(f"Error: {e}")
        finally: os.remove(tmp_path)

    # ───────────────────────────────────────────────────────────────────────────
    # MODE 2: MULTI-FILE OVERLAY (NEW!)
    # ───────────────────────────────────────────────────────────────────────────
    elif mode == "📈 Multi-File Comparison (Overlay)":
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Comparison Settings")
        
        # Carreguem tots els fitxers a la memòria
        loaded_dfs = []
        possible_signals = set()
        
        with st.spinner("Loading all files..."):
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                try:
                    df, _, _ = carregar_fitxer(tmp_path)
                    loaded_dfs.append({"name": f.name, "df": df})
                    # Busquem columnes UV
                    for col in df.columns:
                        if "UV" in col.upper(): possible_signals.add(col)
                except: pass
                finally: os.remove(tmp_path)
        
        if not loaded_dfs:
            st.warning("No valid files loaded.")
        else:
            # Controls per Overlay
            sorted_signals = sorted(list(possible_signals))
            target_sig = st.sidebar.selectbox("Signal to Compare", sorted_signals, index=0 if sorted_signals else 0)
            
            # Dimensions
            c_d1, c_d2 = st.sidebar.columns(2)
            fig_w = c_d1.number_input("Width", 14, key="mw")
            fig_h = c_d2.number_input("Height", 6, key="mh")
            
            # Rangs
            # Calculem rang global
            all_min_x = min([d["df"]["mL"].min() for d in loaded_dfs])
            all_max_x = max([d["df"]["mL"].max() for d in loaded_dfs])
            
            c_r1, c_r2 = st.sidebar.columns(2)
            mx_min = c_r1.number_input("Min X", value=float(all_min_x), step=1.0)
            mx_max = c_r2.number_input("Max X", value=float(all_max_x), step=1.0)
            
            # Normalize?
            normalize = st.sidebar.checkbox("Normalize to Baseline (Start at 0)", value=True)
            
            # Plot
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            
            # Colors automàtics
            colors = plt.cm.tab10(np.linspace(0, 1, len(loaded_dfs)))
            
            for i, item in enumerate(loaded_dfs):
                df = item["df"]
                name = item["name"]
                
                if target_sig in df.columns:
                    y_data = df[target_sig].values
                    if normalize:
                        y_data = y_data - y_data[0] # Simple offset correction
                    
                    ax.plot(df["mL"], y_data, label=name, color=colors[i], alpha=0.8)
            
            ax.set_xlim(mx_min, mx_max)
            ax.set_xlabel("Elution Volume (mL)")
            ax.set_ylabel(f"{target_sig} (mAU)")
            ax.set_title(f"Comparison: {target_sig}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.info(f"Overlaying {len(loaded_dfs)} files. Note: Only plotting selected signal.")

else:
    st.info("👆 Please upload .zip, .res or .result files in the sidebar.")
