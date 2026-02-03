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

# Importació normal (ara ja funcionarà perquè ho hem instal·lat via requirements)
import proteovis as pv

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓ DE LA PÀGINA
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Akta Viewer Web", layout="wide", page_icon="🧬")

st.title("🧬 Visualitzador de Cromatografia Akta")
st.markdown("Puja els teus fitxers `.zip`, `.res` o `.result` per visualitzar-los.")

# ───────────────────────────────────────────────────────────────────────────────
# LÒGICA DE CÀRREGA DE DADES
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
            data, last_err = None, None
            for Parser in parsers:
                try:
                    obj = Parser(path); obj.load(); data = obj; break
                except Exception as e: last_err = e
            if data is None: raise RuntimeError(f"No s'ha pogut llegir el .res ({last_err})")
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

    return df, data

# ───────────────────────────────────────────────────────────────────────────────
# INTERFÍCIE D'USUARI (UI)
# ───────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Arrossega el fitxer aquí", type=['zip', 'res', 'result'])

if uploaded_file is not None:
    # Guardem el fitxer temporalment
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner('Llegint fitxer...'):
            df, data = carregar_fitxer(tmp_path)
        
        st.success(f"✅ Fitxer carregat: {len(df)} punts de dades.")

        # --- Sidebar Controls ---
        st.sidebar.header("Paràmetres del Gràfic")

        cols = list(df.columns)
        possibles_uv = [k for k in cols if "UV" in k.upper()]
        possibles_y2 = [k for k in cols if k not in possibles_uv and k not in ["mL", "Fractions"]]

        col1, col2 = st.sidebar.columns(2)
        y1a_label = col1.selectbox("UV 1", options=possibles_uv, index=0 if possibles_uv else 0)
        y1a_color = col2.color_picker("Color UV1", "#1f77b4")
        
        col3, col4 = st.sidebar.columns(2)
        y1b_label = col3.selectbox("UV 2", options=possibles_uv, index=1 if len(possibles_uv)>1 else 0)
        y1b_color = col4.color_picker("Color UV2", "#ff0000")

        col5, col6 = st.sidebar.columns(2)
        y2_label = col5.selectbox("Eix Y Secundari", options=[""] + possibles_y2)
        y2_color = col6.color_picker("Color Y2", "#2ca02c")

        # Offsets
        uv1_offset = st.sidebar.number_input("Offset UV1", value=0.0, step=0.5)
        uv2_offset = st.sidebar.number_input("Offset UV2", value=0.0, step=0.5)

        # Limits
        st.sidebar.subheader("Límits dels Eixos")
        ml_min, ml_max = float(df["mL"].min()), float(df["mL"].max())
        xmin, xmax = st.sidebar.slider("Rang mL (X)", ml_min, ml_max, (ml_min, ml_max))
        
        uv_max = df[[c for c in [y1a_label, y1b_label] if c]].max().max() if possibles_uv else 100
        ymin, ymax = st.sidebar.slider("Rang Absorbància (Y1)", -20.0, uv_max+100, (0.0, uv_max+50))

        y2_ymin, y2_ymax = 0, 100
        if y2_label:
             y2_max_val = df[y2_label].max()
             y2_ymin, y2_ymax = st.sidebar.slider("Rang Y2", -20.0, float(y2_max_val)+100, (0.0, float(y2_max_val)+10))

        # Estils
        with st.sidebar.expander("Estils i Fraccions"):
            show_fractions = st.checkbox("Mostrar Fraccions", value=True)
            tick_h = st.slider("Alçada marca fracció", 1, 200, 10)
            frac_lw = st.slider("Gruix línia fracció", 0.2, 5.0, 1.0)
            font_labels = st.slider("Mida text etiquetes", 8, 30, 14)
            x_tick_step = st.slider("Pas del Tick Eix X", 0.5, 50.0, 5.0)

        # --- Plotting ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        if y1a_label in df.columns:
            ax1.plot(df["mL"], df[y1a_label] + uv1_offset, label=y1a_label, color=y1a_color)
        if y1b_label in df.columns and y1b_label != y1a_label:
            ax1.plot(df["mL"], df[y1b_label] + uv2_offset, label=y1b_label, color=y1b_color)

        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Elution volume (mL)", fontsize=font_labels)
        ax1.set_ylabel("Absorbance (mAU)", fontsize=font_labels)
        
        if x_tick_step > 0:
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))

        # Fraccions logic
        if show_fractions and "Fractions" in df.columns:
            last_label_x = None
            fractions = df[(df['Fractions'].notna()) & (df['mL'].between(xmin, xmax))].reset_index()
            for i in range(len(fractions)):
                x = fractions.loc[i, 'mL']
                ax1.vlines(x, ymin, ymin + tick_h, color='red', linewidth=frac_lw)
                if i % 2 == 0 and (last_label_x is None or abs(x - (last_label_x or 0)) > 1.0):
                    label = fractions.loc[i, 'Fractions']
                    ax1.text(x, ymin + tick_h + 1, str(label), ha='center', fontsize=10, clip_on=True)
                    last_label_x = x

        # Segon eix
        if y2_label and y2_label in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(df["mL"], df[y2_label], label=y2_label, color=y2_color)
            ax2.set_ylabel(y2_label, fontsize=font_labels)
            ax2.set_ylim(y2_ymin, y2_ymax)
        
        ax1.legend(loc='upper right')
        st.pyplot(fig)

        st.markdown("### Dades processades")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error processant el fitxer: {e}")
    finally:
        os.remove(tmp_path)
