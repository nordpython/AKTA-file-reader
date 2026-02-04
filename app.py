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
# 🚑 PARCHE NUMPY 2.0
# ───────────────────────────────────────────────────────────────────────────────
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

import proteovis as pv

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓ DE LA PÀGINA
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
    .fraction-table {
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Visualitzador de Cromatografia Akta (Versió Completa)")

# ───────────────────────────────────────────────────────────────────────────────
# FUNCIONS AUXILIARS
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

uploaded_file = st.file_uploader("📂 Arrossega el fitxer (.zip, .res, .result)", type=['zip', 'res', 'result'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        df, data, file_name = carregar_fitxer(tmp_path)
        
        # --- Inicialització Memòria ---
        current_uv1_off = st.session_state.get('uv1_off', 0.0)
        current_uv2_off = st.session_state.get('uv2_off', 0.0)
        
        cols = list(df.columns)
        possibles_uv = [k for k in cols if "UV" in k.upper()]
        possibles_y2 = [k for k in cols if k not in possibles_uv and k not in ["mL", "Fractions", "260/280"]]
        
        # Auto-Càlcul inicial Rangs
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

        # ───────────────────────────────────────────────────────────────────────────────
        # SIDEBAR
        # ───────────────────────────────────────────────────────────────────────────────
        st.sidebar.header("⚙️ Configuració")
        
        with st.sidebar.expander("📊 Senyals", expanded=True):
            c1, c2 = st.columns(2)
            y1a_label = c1.selectbox("UV 1", options=possibles_uv, index=0, key='sel_uv1')
            y1a_color = c2.color_picker("Color", "#1f77b4", key='col_uv1')
            
            c3, c4 = st.columns(2)
            y1b_label = c3.selectbox("UV 2", options=[""] + possibles_uv, index=2 if len(possibles_uv)>1 else 0, key='sel_uv2')
            y1b_color = c4.color_picker("Color", "#ff0000", key='col_uv2')
            
            c5, c6 = st.columns(2)
            y2_label = c5.selectbox("Y Secundari", options=[""] + possibles_y2, key='sel_y2')
            y2_color = c6.color_picker("Color", "#2ca02c", key='col_y2')

        with st.sidebar.expander("📏 Zoom i Rangs", expanded=True):
            cd1, cd2 = st.columns(2)
            figwidth = cd1.number_input("Amplada", value=14, step=1, key='fig_w')
            figheight = cd2.number_input("Altura", value=6, step=1, key='fig_h')
            
            st.markdown("---")
            col_x1, col_x2 = st.columns(2)
            xmin = col_x1.number_input("Min X (mL)", step=1.0, key='xmin_input')
            xmax = col_x2.number_input("Max X (mL)", step=1.0, key='xmax_input')
            x_tick_step = st.number_input("Pas Ticks X", value=5.0, min_value=0.1, step=0.5, key='x_step')

            col_y1, col_y2 = st.columns(2)
            ymin = col_y1.number_input("Min Y (mAU)", step=5.0, format="%.1f", key='ymin_input')
            ymax = col_y2.number_input("Max Y (mAU)", step=5.0, format="%.1f", key='ymax_input')
            
            y2_ymin, y2_ymax = 0.0, 100.0
            if y2_label:
                st.markdown("**Eix Secundari**")
                c_y2_1, c_y2_2 = st.columns(2)
                if 'y2_max_input' not in st.session_state:
                     st.session_state.y2_max_input = float(df[y2_label].max()) + 10.0
                y2_ymin = c_y2_1.number_input("Min Y2", value=0.0, key='y2_min_input')
                y2_ymax = c_y2_2.number_input("Max Y2", key='y2_max_input')

        with st.sidebar.expander("🧪 Fraccions", expanded=False):
            show_fractions = st.checkbox("Mostrar Fraccions", value=True, key='show_fracs')
            frac_step = st.number_input("Etiqueta cada N", value=1, min_value=1, key='frac_step')
            tick_h = st.slider("Alçada", 1.0, 300.0, float((ymax-ymin)*0.1) if (ymax-ymin) > 0 else 10.0, key='frac_h')
            label_offset = st.number_input("Posició Text", min_value=0.0, value=2.0, step=0.5, key='frac_offset')
            font_frac = st.slider("Mida Text", 6, 20, 9, key='frac_font')

        with st.sidebar.expander("🎨 Estils", expanded=False):
            font_title = st.slider("Títol", 10, 40, 16, key='f_title')
            font_labels = st.slider("Etiquetes", 8, 30, 12, key='f_labels')
            font_ticks = st.slider("Ticks", 8, 20, 10, key='f_ticks')
            font_legend = st.slider("Llegenda", 8, 20, 10, key='f_legend')

        with st.sidebar.expander("🛠️ Extres", expanded=False):
            uv1_offset = st.number_input("Offset UV1", step=0.5, key='uv1_off')
            uv2_offset = st.number_input("Offset UV2", step=0.5, key='uv2_off')

        # ───────────────────────────────────────────────────────────────────────────────
        # GRÀFIC
        # ───────────────────────────────────────────────────────────────────────────────
        fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

        if y1a_label in df.columns:
            ax1.plot(df["mL"], df[y1a_label] + uv1_offset, label=y1a_label, color=y1a_color)
        if y1b_label in df.columns and y1b_label != y1a_label:
            ax1.plot(df["mL"], df[y1b_label] + uv2_offset, label=y1b_label, color=y1b_color)

        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Elution volume (mL)", fontsize=font_labels)
        ax1.set_ylabel("Absorbance (mAU)", fontsize=font_labels)
        ax1.tick_params(axis='both', labelsize=font_ticks)
        ax1.set_title(f"Chromatogram – {file_name}", fontsize=font_title)

        if x_tick_step > 0:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))

        if show_fractions and "Fractions" in df.columns:
            fractions = df[(df['Fractions'].notna()) & (df['mL'].between(xmin, xmax))].reset_index()
            for i in range(len(fractions)):
                x = fractions.loc[i, 'mL']
                label = fractions.loc[i, 'Fractions']
                ax1.vlines(x, ymin, ymin + tick_h, color='red', linewidth=1, zorder=5)
                if i % frac_step == 0:
                    txt = 'W' if str(label).lower() == 'waste' else str(label)
                    ax1.text(x, ymin + tick_h + label_offset, txt, ha='center', va='bottom', 
                             fontsize=font_frac, color='black', clip_on=False, zorder=6)

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

        # ───────────────────────────────────────────────────────────────────────────────
        # MÒDUL D'INTEGRACIÓ DE PICS
        # ───────────────────────────────────────────────────────────────────────────────
        with st.expander("🧮 Càlculs i Integració de Pics", expanded=True):
            col_calc1, col_calc2 = st.columns([1, 2])
            
            with col_calc1:
                st.markdown("#### 1. Paràmetres")
                int_start = st.number_input("Inici (mL)", value=xmin, step=0.5)
                int_end = st.number_input("Final (mL)", value=xmax, step=0.5)
                
                target_signal = st.selectbox("Senyal a Integrar", [y1a_label, y1b_label])
                baseline_mode = st.selectbox("Correcció Base", ["Cap", "Lineal (Inici-Fi)"])
                
                st.markdown("#### 2. Dades Proteïna")
                path_length = st.number_input("Camí Òptic (cm)", value=0.2, format="%.2f", help="Normalment 0.2 o 0.5 a l'Akta")
                
                coeff_type = st.radio("Tipus Coeficient", ["Abs 0.1% (1 g/L)", "Molar (M⁻¹ cm⁻¹)"])
                
                if coeff_type == "Abs 0.1% (1 g/L)":
                    ext_coeff_mass = st.number_input("Valor Abs 0.1%", value=1.0, format="%.3f")
                    ext_coeff_molar = None
                else:
                    ext_coeff_molar = st.number_input("Valor Molar (ε)", value=50000.0, format="%.1f")
                    ext_coeff_mass = None
                
                mol_weight = st.number_input("Pes Molecular (Da)", value=10000.0, format="%.1f", help="Necessari per calcular µM")
                
                # 🟢 NOU: Control de Decimals
                st.markdown("#### 3. Format Taula")
                decimals = st.number_input("Decimals (Taula)", value=4, min_value=1, max_value=8)

            with col_calc2:
                st.markdown("#### Resultats del Pic")
                if target_signal and target_signal in df.columns:
                    mask = (df["mL"] >= int_start) & (df["mL"] <= int_end)
                    sub_df = df[mask].copy()
                    
                    if not sub_df.empty:
                        x_vals = sub_df["mL"].values
                        offset_val = uv1_offset if target_signal == y1a_label else uv2_offset
                        y_vals = sub_df[target_signal].values + offset_val
                        
                        if baseline_mode == "Lineal (Inici-Fi)":
                            slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                            baseline = y_vals[0] + slope * (x_vals - x_vals[0])
                            y_processed = y_vals - baseline
                        else:
                            y_processed = y_vals
                            baseline = np.zeros_like(y_vals)
                        
                        avg_mAU = np.mean(y_processed)
                        avg_AU = avg_mAU / 1000.0

                        # --- CÀLCUL DE CONCENTRACIONS ---
                        conc_mg_ml = 0.0
                        conc_uM = 0.0
                        
                        if path_length > 0:
                            if coeff_type == "Abs 0.1% (1 g/L)" and ext_coeff_mass > 0:
                                conc_mg_ml = avg_AU / (ext_coeff_mass * path_length)
                            elif coeff_type == "Molar (M⁻¹ cm⁻¹)" and ext_coeff_molar > 0 and mol_weight > 0:
                                molarity = avg_AU / (ext_coeff_molar * path_length)
                                conc_mg_ml = molarity * mol_weight 
                            
                            if mol_weight > 0:
                                molarity_calc = (conc_mg_ml) / mol_weight 
                                conc_uM = molarity_calc * 1e6

                        area_mAU_mL = np.trapz(y_processed, x_vals)
                        area_AU_mL = area_mAU_mL / 1000.0
                        
                        mass_mg = 0.0
                        if path_length > 0:
                            if coeff_type == "Abs 0.1% (1 g/L)" and ext_coeff_mass > 0:
                                mass_mg = area_AU_mL / (ext_coeff_mass * path_length)
                            elif coeff_type == "Molar (M⁻¹ cm⁻¹)" and ext_coeff_molar > 0 and mol_weight > 0:
                                mass_mg = (area_AU_mL * mol_weight) / (ext_coeff_molar * path_length)

                        peak_vol = int_end - int_start
                        
                        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                        c_res1.metric("Massa Total", f"{mass_mg:.{decimals}f} mg")
                        c_res2.metric("Volum Pic", f"{peak_vol:.2f} mL")
                        c_res3.metric("Conc. Mitjana", f"{conc_mg_ml:.{decimals}f} mg/mL")
                        c_res4.metric("Conc. Mitjana", f"{conc_uM:.{decimals}f} µM")

                        # ───────────────────────────────────────────────────────────
                        # ANÀLISI PER FRACCIÓ (FORMAT CONFIGURABLE)
                        # ───────────────────────────────────────────────────────────
                        st.markdown("#### 🧪 Detall per Fracció (Dins del pic)")
                        
                        if "Fractions" in df.columns:
                            frac_indices = df[df['Fractions'].notna()].index
                            frac_data_list = []
                            
                            for i in range(len(frac_indices)):
                                idx_start = frac_indices[i]
                                idx_end = frac_indices[i+1] if i < len(frac_indices)-1 else df.index[-1]
                                
                                f_ml_start = df.loc[idx_start, "mL"]
                                f_ml_end = df.loc[idx_end, "mL"]
                                f_name = df.loc[idx_start, "Fractions"]
                                
                                overlap_start = max(f_ml_start, int_start)
                                overlap_end = min(f_ml_end, int_end)
                                
                                if overlap_start < overlap_end:
                                    f_mask = (df["mL"] >= overlap_start) & (df["mL"] <= overlap_end)
                                    f_sub = df[f_mask]
                                    
                                    if not f_sub.empty:
                                        f_y_vals = f_sub[target_signal].values + offset_val
                                        if baseline_mode == "Lineal (Inici-Fi)":
                                            f_x_vals = f_sub["mL"].values
                                            f_base = y_vals[0] + slope * (f_x_vals - x_vals[0])
                                            f_y_processed = f_y_vals - f_base
                                        else:
                                            f_y_processed = f_y_vals
                                        
                                        f_avg_mAU = np.mean(f_y_processed)
                                        f_avg_AU = f_avg_mAU / 1000.0
                                        
                                        f_mg_ml = 0.0
                                        f_uM = 0.0
                                        if path_length > 0:
                                            if coeff_type == "Abs 0.1% (1 g/L)" and ext_coeff_mass > 0:
                                                f_mg_ml = f_avg_AU / (ext_coeff_mass * path_length)
                                            elif coeff_type == "Molar (M⁻¹ cm⁻¹)" and ext_coeff_molar > 0:
                                                molar = f_avg_AU / (ext_coeff_molar * path_length)
                                                f_mg_ml = molar * mol_weight
                                            
                                            if mol_weight > 0:
                                                f_uM = (f_mg_ml / mol_weight) * 1e6

                                        frac_data_list.append({
                                            "Fracció": f_name,
                                            "Volum (mL)": f"{overlap_end - overlap_start:.2f}",
                                            # ÚS DE F-STRINGS DINÀMICS PER DECIMALS
                                            "Abs Mitjana (mAU)": f"{f_avg_mAU:.1f}",
                                            "Conc (mg/mL)": f"{f_mg_ml:.{decimals}f}",
                                            "Conc (µM)": f"{f_uM:.{decimals}f}"
                                        })
                            
                            if frac_data_list:
                                st.dataframe(pd.DataFrame(frac_data_list), use_container_width=True)
                            else:
                                st.info("No s'han trobat fraccions dins d'aquest rang.")

                        with st.expander("Veure àrea integrada"):
                            fig_area, ax_area = plt.subplots(figsize=(6, 2))
                            ax_area.plot(x_vals, y_vals, 'b-', label="Senyal")
                            if baseline_mode == "Lineal (Inici-Fi)":
                                ax_area.plot(x_vals, baseline, 'k--', label="Base", alpha=0.5)
                            ax_area.fill_between(x_vals, y_vals, baseline if baseline_mode == "Lineal (Inici-Fi)" else 0, alpha=0.3, color='green')
                            st.pyplot(fig_area)
                    else:
                        st.warning("No hi ha dades en aquest rang.")

        with st.expander("📋 Dades Brutes"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        os.remove(tmp_path)
