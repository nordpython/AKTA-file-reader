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

st.title("🧬 Visualitzador de Cromatografia Akta (Versió Completa)")

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
            
    # Ratio Calculation if available
    if "UV 2_260" in df.columns and "UV 1_280" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["260/280"] = df["UV 2_260"] / df["UV 1_280"]

    return df, data, os.path.basename(path)

# ───────────────────────────────────────────────────────────────────────────────
# UI PRINCIPAL
# ───────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("📂 Arrossega el fitxer (.zip, .res, .result)", type=['zip', 'res', 'result'])

if uploaded_file is not None:
    # 1. Càrrega del fitxer
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        df, data, file_name = carregar_fitxer(tmp_path)
        
        # 2. Sidebar de Controls
        st.sidebar.header("⚙️ Configuració del Gràfic")
        
        # --- Selecció de Dades ---
        cols = list(df.columns)
        possibles_uv = [k for k in cols if "UV" in k.upper()]
        possibles_y2 = [k for k in cols if k not in possibles_uv and k not in ["mL", "Fractions", "260/280"]]
        
        with st.sidebar.expander("📊 Senyals i Colors", expanded=True):
            c1, c2 = st.columns(2)
            y1a_label = c1.selectbox("UV 1 (Principal)", options=possibles_uv, index=0 if possibles_uv else 0)
            y1a_color = c2.color_picker("Color UV1", "#1f77b4")
            
            c3, c4 = st.columns(2)
            y1b_label = c3.selectbox("UV 2", options=[""] + possibles_uv, index=2 if len(possibles_uv)>1 else 0)
            y1b_color = c4.color_picker("Color UV2", "#ff0000")
            
            c5, c6 = st.columns(2)
            y2_label = c5.selectbox("Eix Y Secundari", options=[""] + possibles_y2)
            y2_color = c6.color_picker("Color Y2", "#2ca02c")
            
            st.markdown("---")
            uv1_offset = st.number_input("Offset UV1 (mAU)", value=0.0, step=0.5)
            uv2_offset = st.number_input("Offset UV2 (mAU)", value=0.0, step=0.5)

        # --- Límits ---
        with st.sidebar.expander("📏 Rangs i Eixos (Zoom)", expanded=True):
            ml_min, ml_max = float(df["mL"].min()), float(df["mL"].max())
            xmin, xmax = st.slider("Rang Eix X (mL)", ml_min, ml_max, (ml_min, ml_max))
            
            st.markdown("##### Eix Y (Absorbància)")
            # 🟢 NOVETAT: Checkbox per Auto-escala
            auto_scale_y = st.checkbox("Auto-ajustar Y (+/- 4 unitats)", value=True)
            
            # Càlcul dinàmic dels màxims i mínims de les dades seleccionades
            current_uv_data = []
            if y1a_label in df.columns:
                current_uv_data.append(df[y1a_label] + uv1_offset)
            if y1b_label in df.columns and y1b_label != y1a_label:
                current_uv_data.append(df[y1b_label] + uv2_offset)
            
            # Valors per defecte si no hi ha dades
            calc_min, calc_max = 0.0, 100.0
            if current_uv_data:
                # Concatenem per trobar el min/max global de les corbes actives
                combined = pd.concat(current_uv_data)
                calc_min = combined.min()
                calc_max = combined.max()

            if auto_scale_y:
                # Apliquem el marge de 4 unitats
                ymin = calc_min - 4.0
                ymax = calc_max + 4.0
                st.info(f"Escala automàtica: {ymin:.1f} a {ymax:.1f} mAU")
            else:
                # Si no és auto, mostrem el slider manual
                default_ymax = float(calc_max) + 50
                ymin, ymax = st.slider("Rang Manual Y", -20.0, default_ymax + 100, (0.0, default_ymax))
            
            # Configuració Eix secundari
            y2_ymin, y2_ymax = 0.0, 100.0
            if y2_label:
                y2_curr_max = float(df[y2_label].max())
                y2_ymin, y2_ymax = st.slider("Rang Eix Y Secundari", -20.0, y2_curr_max+100, (0.0, y2_curr_max+10))
                
            x_tick_step = st.number_input("Pas dels Ticks Eix X (cada quants mL)", value=5.0, min_value=0.1, step=0.5)

        # --- Estètica ---
        with st.sidebar.expander("🎨 Fonts i Estils", expanded=False):
            figwidth = st.slider("Amplada Gràfic", 8, 30, 14)
            font_title = st.slider("Mida Títol", 10, 40, 16)
            font_labels = st.slider("Mida Etiquetes Eixos", 8, 30, 12)
            font_ticks = st.slider("Mida Números Eixos", 8, 20, 10)
            font_legend = st.slider("Mida Llegenda", 8, 20, 10)

        # --- Fraccions ---
        with st.sidebar.expander("🧪 Fraccions", expanded=False):
            show_fractions = st.checkbox("Mostrar Fraccions", value=True)
            # Ajustem l'alçada de les fraccions dinàmicament si estem en mode auto
            default_tick_h = (ymax - ymin) * 0.1 # 10% de l'alçada del gràfic
            tick_h = st.slider("Alçada marca vermella", 1.0, 300.0, float(default_tick_h))
            
            frac_lw = st.slider("Gruix línia", 0.2, 5.0, 1.0)
            label_offset = st.slider("Posició Text (Vertical)", -20.0, 50.0, 2.0)
            font_frac = st.slider("Mida Text Fracció", 6, 20, 9)

        # 3. Generació del Gràfic
        fig, ax1 = plt.subplots(figsize=(figwidth, 6))

        # Plot UV1
        if y1a_label and y1a_label in df.columns:
            ax1.plot(df["mL"], df[y1a_label] + uv1_offset, label=y1a_label, color=y1a_color)
        
        # Plot UV2
        if y1b_label and y1b_label in df.columns and y1b_label != y1a_label:
            ax1.plot(df["mL"], df[y1b_label] + uv2_offset, label=y1b_label, color=y1b_color)

        # Configuració Eix 1
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Elution volume (mL)", fontsize=font_labels)
        ax1.set_ylabel("Absorbance (mAU)", fontsize=font_labels)
        ax1.tick_params(axis='both', labelsize=font_ticks)
        ax1.set_title(f"Chromatogram – {file_name}", fontsize=font_title)

        if x_tick_step > 0:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))

        # Plot Fraccions
        if show_fractions and "Fractions" in df.columns:
            last_label_x = None
            fractions = df[(df['Fractions'].notna()) & (df['mL'].between(xmin, xmax))].reset_index()
            for i in range(len(fractions)):
                x = fractions.loc[i, 'mL']
                label = fractions.loc[i, 'Fractions']
                
                # Línia vermella
                ax1.vlines(x, ymin, ymin + tick_h, color='red', linewidth=frac_lw, zorder=5)
                
                # Text
                if i % 2 == 0 and (last_label_x is None or abs(x - (last_label_x or 0)) > 1.0):
                    txt = 'W' if str(label).lower() == 'waste' else str(label)
                    ax1.text(x, ymin + tick_h + label_offset, txt, 
                             ha='center', va='bottom', fontsize=font_frac, color='black', clip_on=True, zorder=6)
                    last_label_x = x

        # Segon Eix Y
        ax2 = None
        if y2_label and y2_label in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df["mL"], df[y2_label], label=y2_label, color=y2_color, linestyle="--")
            ax2.set_ylabel(y2_label, fontsize=font_labels)
            ax2.tick_params(axis='y', labelsize=font_ticks)
            ax2.set_ylim(y2_ymin, y2_ymax)

        # Llegenda unificada
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels() if ax2 else ([], [])
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=font_legend)

        st.pyplot(fig)
        
        with st.expander("📋 Veure Dades en Taula"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error processant el fitxer: {e}")
    finally:
        os.remove(tmp_path)
