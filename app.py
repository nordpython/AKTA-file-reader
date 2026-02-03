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
        
        # --- Variables per a offsets (les necessitem abans de pintar) ---
        # S'inicialitzen aquí però es modifiquen al menú d'Extres al final del codi
        # Per ordre d'execució d'Streamlit, primer definim valors per defecte, 
        # però farem servir session_state o simplement les llegirem del menú després.
        # Truc: posem el menú d'extres al final de la sidebar, però llegim els valors ara.
        
        # Per tenir els valors abans del plot, hem de pintar els controls.
        
        # 2. Sidebar de Controls
        st.sidebar.header("⚙️ Configuració del Gràfic")
        
        # --- Senyals i Colors ---
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

        # --- Límits (Caixes numèriques) ---
        with st.sidebar.expander("📏 Rangs i Eixos (Zoom)", expanded=True):
            # EIX X
            st.markdown("**Eix X (mL)**")
            col_x1, col_x2 = st.columns(2)
            ml_min_val, ml_max_val = float(df["mL"].min()), float(df["mL"].max())
            xmin = col_x1.number_input("Mínim X", value=ml_min_val, step=1.0)
            xmax = col_x2.number_input("Màxim X", value=ml_max_val, step=1.0)
            
            x_tick_step = st.number_input("Pas dels Ticks Eix X (cada quants mL)", value=5.0, min_value=0.1, step=0.5)

            st.markdown("---")
            # EIX Y
            st.markdown("**Eix Y (Absorbància)**")
            auto_scale_y = st.checkbox("Auto-ajustar Y (+/- 4 unitats)", value=True)
            
            # Offsets (necessaris per calcular el max/min) - Els definim al final, però els necessitem aquí.
            # Per no trencar l'ordre visual, posarem els inputs aquí o farem un placeholder.
            # El millor en Streamlit és definir els inputs on toquen visualment.
            # Com que l'usuari vol els offsets a "Extres", haurem de moure aquell codi abans d'aquest bloc o acceptar que es llegeixin després.
            # Farem el menú "Extres" ara mateix per tenir les variables disponibles, però usant un truc visual 
            # o simplement acceptant l'ordre. Per complir la petició 1, el posarem al final.
            # Per tant, inicialitzarem variables a 0 i després les llegirem.
            # NO, Streamlit executa de dalt a baix. Si volem offsets al final, els hem de pintar al final.
            # Solució: Pintar el menú "Extres" ARA però que surti abaix? No es pot fàcilment.
            # Solució pràctica: Pintem el menú "Extres" aquí al codi, però li diem a l'usuari que està al final.
            # Més fàcil: Creem els placeholders.
            
        # --- MENU EXTRES (Offsets) ---
        # Ho poso aquí al codi perquè necessito els valors 'uv1_offset' per calcular l'Auto-Scale.
        # Però visualment, vull que surti al final de la sidebar.
        # Utilitzaré st.sidebar.empty() després.
        
        # Com que no puc moure el widget visualment avall si el defineixo amunt,
        # definiré els offsets amb un valor per defecte 0.0 temporalment per al càlcul,
        # i després redibuixaré el gràfic? No.
        # Simplement acceptaré que els offsets es defineixin en un expander AQUI, 
        # o bé, faré el càlcul de l'auto-scale sense els offsets (només dades crues) 
        # i aplicaré els offsets visualment. Això és més segur.
        
        # Càlcul Auto-Scale (amb dades crues)
        current_uv_data = []
        if y1a_label in df.columns:
            current_uv_data.append(df[y1a_label]) # Sense offset encara
        if y1b_label in df.columns and y1b_label != y1a_label:
            current_uv_data.append(df[y1b_label]) # Sense offset encara
        
        calc_min, calc_max = 0.0, 100.0
        if current_uv_data:
            combined = pd.concat(current_uv_data)
            calc_min = combined.min()
            calc_max = combined.max()

        # Ara recuperem els Offsets del final (fem servir session_state o valors per defecte)
        # Per simplificar i que funcioni bé: Crearem l'Expander "Extres" aquí, però el marcarem com a tancat.
        # Si l'usuari vol que estigui visualment AL FINAL DE TOT, haurem de moure els altres expanders abans.
        
        # ORDRE VISUAL: 
        # 1. Senyals (Fet)
        # 2. Rangs (Estem a dins)
        # 3. Estils
        # 4. Fraccions
        # 5. Extres (Offsets)
        
        # Problema: Necessito els offsets PER als rangs si vull que l'auto-scale sigui perfecte.
        # Solució: Llegeixo els offsets PRIMER de tot (invisible o al principi), o els poso dins de Senyals.
        # Si l'usuari vol "Extres" al final, definirem els inputs al final.
        # Llavors l'Auto-scale NO tindrà en compte l'offset fins al següent refresc.
        # Farem un compromís: Poso "Extres" just després de "Rangs" o abans.
        
        # D'acord, per fer-ho bé, poso l'expander "Extres" al final del codi Python,
        # i per al càlcul automàtic assumeixo offset 0 en la primera passada o faig servir st.session_state.
        uv1_offset_val = st.session_state.get('uv1_off', 0.0)
        uv2_offset_val = st.session_state.get('uv2_off', 0.0)

        if auto_scale_y:
            # Apliquem el marge de 4 unitats tenint en compte l'offset (encara que sigui 0 al principi)
            ymin = (calc_min + uv1_offset_val) - 4.0
            ymax = (calc_max + uv1_offset_val) + 4.0 # Assumint que el offset principal mana
            st.info(f"Escala automàtica: {ymin:.1f} a {ymax:.1f} mAU (Inclou offsets)")
        else:
            col_y1, col_y2 = st.columns(2)
            ymin = col_y1.number_input("Mínim Y", value=0.0, step=10.0)
            ymax = col_y2.number_input("Màxim Y", value=float(calc_max)+50, step=10.0)
            
        # Segon Eix
        y2_ymin, y2_ymax = 0.0, 100.0
        if y2_label:
            st.markdown("**Eix Y Secundari**")
            y2_curr_max = float(df[y2_label].max())
            c_y2_1, c_y2_2 = st.columns(2)
            y2_ymin = c_y2_1.number_input("Mínim Y2", value=0.0)
            y2_ymax = c_y2_2.number_input("Màxim Y2", value=y2_curr_max+10)

        # --- Fraccions (Modificat) ---
        with st.sidebar.expander("🧪 Fraccions", expanded=False):
            show_fractions = st.checkbox("Mostrar Fraccions", value=True)
            
            # 🟢 NOVETAT: Frac Step en lloc de Min Spacing
            frac_step = st.number_input("Etiquetar cada N fraccions (Step)", value=1, min_value=1, step=1)
            
            default_tick_h = (ymax - ymin) * 0.1
            tick_h = st.slider("Alçada marca vermella", 1.0, 300.0, float(default_tick_h) if default_tick_h > 0 else 10.0)
            frac_lw = st.slider("Gruix línia", 0.2, 5.0, 1.0)
            label_offset = st.slider("Posició Text (Vertical)", -50.0, 100.0, 2.0)
            font_frac = st.slider("Mida Text Fracció", 6, 20, 9)

        # --- Estètica ---
        with st.sidebar.expander("🎨 Fonts i Estils", expanded=False):
            figwidth = st.slider("Amplada Gràfic", 8, 30, 14)
            font_title = st.slider("Mida Títol", 10, 40, 16)
            font_labels = st.slider("Mida Etiquetes Eixos", 8, 30, 12)
            font_ticks = st.slider("Mida Números Eixos", 8, 20, 10)
            font_legend = st.slider("Mida Llegenda", 8, 20, 10)

        # --- Extres (Offsets) --- 
        # 🟢 NOVETAT: Menú al final
        with st.sidebar.expander("🛠️ Extres (Offsets)", expanded=False):
            uv1_offset = st.number_input("Offset UV1 (mAU)", value=0.0, step=0.5, key='uv1_off')
            uv2_offset = st.number_input("Offset UV2 (mAU)", value=0.0, step=0.5, key='uv2_off')

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

        # Plot Fraccions (Lògica Step)
        if show_fractions and "Fractions" in df.columns:
            # Filtrem primer per rang X per no processar coses fora de visió
            fractions = df[(df['Fractions'].notna()) & (df['mL'].between(xmin, xmax))].reset_index()
            
            for i in range(len(fractions)):
                # Dibuixem la línia SEMPRE (totes les fraccions tenen línia)
                x = fractions.loc[i, 'mL']
                label = fractions.loc[i, 'Fractions']
                ax1.vlines(x, ymin, ymin + tick_h, color='red', linewidth=frac_lw, zorder=5)
                
                # Etiquetem NOMÉS si compleix el STEP
                # i+1 perquè l'humà compta des de 1, no 0 (opcional, però més intuïtiu si step=5 volem la 1, 6, 11...)
                if i % frac_step == 0:
                    txt = 'W' if str(label).lower() == 'waste' else str(label)
                    ax1.text(x, ymin + tick_h + label_offset, txt, 
                             ha='center', va='bottom', fontsize=font_frac, color='black', clip_on=False, zorder=6)

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
