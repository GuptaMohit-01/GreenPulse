# app.py
import streamlit as st
import joblib
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from io import BytesIO
import requests
import os
import datetime
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ---- Page Config ----
st.set_page_config(
    page_title="GreenPulse - Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# ---- Theme CSS (light + dark toggle support) ----
LIGHT_CSS = """
:root {
  --accent: #16a34a;
  --accent-2: #22c55e;
  --bg-top: #f7fff7;
  --bg-bottom: #ffffff;
  --surface: #ffffff;
  --text: #0f172a;
  --muted: #475569;
  --card-shadow: rgba(2,6,23,0.06);
}
"""
DARK_CSS = """
:root {
  --accent: #16a34a;
  --accent-2: #22c55e;
  --bg-top: #071013;
  --bg-bottom: #0b1220;
  --surface: #071620;
  --text: #e6eef6;
  --muted: #9fb0c1;
  --card-shadow: rgba(0,0,0,0.6);
}
"""

BASE_CSS = """
.stApp {
  background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 85%);
  color: var(--text);
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
}
.card {
  background: var(--surface);
  border-radius: 12px;
  padding: 18px;
  margin-top: 12px;
  box-shadow: 0 6px 18px var(--card-shadow);
  color: var(--text);
}
.stButton>button {
  background: linear-gradient(90deg, var(--accent-2), var(--accent));
  color: #ffffff;
  border: none;
  padding: 10px 16px;
  border-radius: 10px;
  font-weight: 600;
}
section[data-testid="stSidebar"] {
  background-color: rgba(255,255,255,0.02) !important;
  color: var(--text) !important;
  padding: 12px 12px 24px 12px;
}
section[data-testid="stSidebar"] pre,
section[data-testid="stSidebar"] code,
section[data-testid="stSidebar"] .stJson {
  color: var(--text) !important;
  background: var(--surface) !important;
  border: 1px solid rgba(0,0,0,0.06) !important;
  padding: 8px 10px !important;
  border-radius: 8px !important;
  font-family: Menlo, Monaco, "Courier New", monospace !important;
  font-size: 13px !important;
  overflow-x: auto !important;
  white-space: pre-wrap;
}
"""

# theme selector in sidebar
with st.sidebar:
    st.markdown("### 🎨 Theme")
    # default to Dark theme
    theme = st.radio("Choose theme", ("Light", "Dark"), index=1)
css = (LIGHT_CSS if theme == "Light" else DARK_CSS) + BASE_CSS
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# ---- Load Artifacts ----
@st.cache_resource
def load_artifacts():
    model_path = Path("artifacts/model.pkl")
    labels_path = Path("artifacts/label_classes.json")
    meta_path = Path("artifacts/meta.json")

    if not model_path.exists():
        return None, None, {}

    model = joblib.load(model_path)

    labels = None
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = None
        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            # try numeric-key mapping first
            try:
                numeric_items = {int(k): v for k, v in data.items() if str(k).isdigit()}
                labels = [numeric_items[i] for i in sorted(numeric_items)]
            except Exception:
                try:
                    inv = {int(v): k for k, v in data.items()}
                    labels = [inv[i] for i in range(len(inv))]
                except Exception:
                    labels = list(map(str, model.classes_))
        else:
            labels = list(map(str, model.classes_))
    else:
        labels = list(map(str, model.classes_))

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                meta = json.load(f)
            except Exception:
                meta = {}

    return model, labels, meta

model, label_classes, meta = load_artifacts()

# ---- Helpers: normalize and build mapping editor ----
def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())

UI_FEATURES = ["N", "P", "K", "Temp", "Humidity", "ph", "Rain"]  # UI order

# expected feature names from meta or model
expected_from_meta = meta.get("feature_order") if isinstance(meta.get("feature_order"), list) else None
if not expected_from_meta:
    if model is not None and hasattr(model, "feature_names_in_"):
        expected_from_meta = list(model.feature_names_in_)
    else:
        expected_from_meta = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# build default mapping (best normalized match)
ui_norm_map = {norm(u): u for u in UI_FEATURES}
exp_norm_map = {norm(e): e for e in expected_from_meta}
default_map = {}
for en_norm, en_name in exp_norm_map.items():
    if en_norm in ui_norm_map:
        default_map[ui_norm_map[en_norm]] = en_name
# heuristics
heur = {"temp": "temperature", "rain": "rainfall", "humid": "humidity", "n": "N", "p": "P", "k": "K", "ph": "ph"}
for k, v in heur.items():
    if k in ui_norm_map and (v in expected_from_meta or norm(v) in exp_norm_map):
        default_map.setdefault(ui_norm_map[k], v)

# session mapping store
if "feature_map" not in st.session_state:
    st.session_state["feature_map"] = default_map.copy()

# ---- Sidebar: grouped inputs + mapping editor + artifacts expander + actions
with st.sidebar:
    st.header("Input Soil & Weather Data")

    # --- Map pin / coordinates for weather ---
    st.markdown("#### 📍 Location (optional)")
    lat = st.text_input("Latitude", value=str(st.session_state.get("lat", "")), key="lat_input")
    lon = st.text_input("Longitude", value=str(st.session_state.get("lon", "")), key="lon_input")
    if st.button("Set location & show map"):
        try:
            st.session_state["lat"] = float(lat)
            st.session_state["lon"] = float(lon)
        except Exception:
            st.error("Invalid lat/lon")
    if "lat" in st.session_state and "lon" in st.session_state:
        st.map(pd.DataFrame([{"lat": st.session_state["lat"], "lon": st.session_state["lon"]}]))

    # --- Weather integration controls (updated to support coords) ---
    st.markdown("#### 🌤️ Live weather (OpenWeatherMap)")
    api_key = st.text_input("OpenWeather API Key (optional)", type="password", help="Get a key from https://openweathermap.org/")
    city = st.text_input("City (e.g. New Delhi)", value="", help="City name to fetch current weather")
    auto_fill = st.checkbox("Auto-fill sliders from weather when fetched", value=True)

    @st.cache_data(ttl=600)
    def fetch_weather(city_name, key):
        if not city_name or not key:
            return {"error": "city or api key missing"}
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {"q": city_name, "appid": key, "units": "metric"}
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            main = data.get("main", {})
            rain_obj = data.get("rain", {})
            temp = main.get("temp")
            humidity = main.get("humidity")
            rain = float(rain_obj.get("1h", rain_obj.get("3h", 0.0)) or 0.0)
            return {"temp": temp, "humidity": humidity, "rain": rain, "raw": data}
        except Exception as e:
            return {"error": str(e)}

    @st.cache_data(ttl=600)
    def fetch_weather_by_coords(lat_f, lon_f, key):
        if lat_f is None or lon_f is None or not key:
            return {"error": "coords or api key missing"}
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {"lat": lat_f, "lon": lon_f, "appid": key, "units": "metric"}
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            main = data.get("main", {})
            rain_obj = data.get("rain", {})
            temp = main.get("temp")
            humidity = main.get("humidity")
            rain = float(rain_obj.get("1h", rain_obj.get("3h", 0.0)) or 0.0)
            return {"temp": temp, "humidity": humidity, "rain": rain, "raw": data}
        except Exception as e:
            return {"error": str(e)}

    colw1, colw2 = st.columns([3,1])
    with colw1:
        if st.button("Fetch Weather by City"):
            w = fetch_weather(city.strip(), api_key.strip())
            st.session_state["weather_last"] = w
            if "error" in w:
                st.error("Weather fetch failed: " + str(w.get("error")))
            else:
                st.success(f"Weather for {city.strip()} fetched")
                if auto_fill:
                    if w.get("temp") is not None:
                        st.session_state["Temp"] = float(round(w["temp"], 1))
                    if w.get("humidity") is not None:
                        st.session_state["Humidity"] = float(round(w["humidity"], 1))
                    if w.get("rain") is not None:
                        st.session_state["Rain"] = float(round(w["rain"], 1))
        if st.button("Fetch Weather by Coords"):
            lat_val = st.session_state.get("lat")
            lon_val = st.session_state.get("lon")
            w = fetch_weather_by_coords(lat_val, lon_val, api_key.strip())
            st.session_state["weather_last"] = w
            if "error" in w:
                st.error("Weather fetch failed: " + str(w.get("error")))
            else:
                st.success(f"Weather for coords fetched")
                if auto_fill:
                    if w.get("temp") is not None:
                        st.session_state["Temp"] = float(round(w["temp"], 1))
                    if w.get("humidity") is not None:
                        st.session_state["Humidity"] = float(round(w["humidity"], 1))
                    if w.get("rain") is not None:
                        st.session_state["Rain"] = float(round(w["rain"], 1))
    with colw2:
        if st.session_state.get("weather_last"):
            last = st.session_state["weather_last"]
            if "error" in last:
                st.write("No data")
            else:
                st.write(f"{last.get('temp', '?')}°C")
# -- end weather controls --

    # grouped inputs (use keys so session_state updates work)
    with st.expander("Inputs (soil & weather)", expanded=True):
        N = st.slider("Nitrogen (N)", 0, 140, 50, step=1, key="N")
        P = st.slider("Phosphorus (P)", 0, 140, 50, step=1, key="P")
        K = st.slider("Potassium (K)", 0, 200, 50, step=1, key="K")
        Temp = st.slider("Temperature (°C)", 0.0, 50.0, st.session_state.get("Temp", 25.0), step=0.5, key="Temp")
        Hum = st.slider("Humidity (%)", 0.0, 100.0, st.session_state.get("Humidity", 50.0), step=0.5, key="Humidity")
        ph = st.slider("pH Level", 0.0, 14.0, 6.5, step=0.1, key="ph")
        Rain = st.slider("Rainfall (mm)", 0.0, 300.0, st.session_state.get("Rain", 100.0), step=1.0, key="Rain")

    # mapping editor to align UI names -> model expected names
    with st.expander("Feature Mapping (UI → model) — edit if needed", expanded=False):
        st.write("Adjust how UI inputs map to model feature names. Useful if model expects e.g. 'temperature' instead of 'Temp'.")
        fm = st.session_state["feature_map"]
        for ui in UI_FEATURES:
            curr = fm.get(ui, "")
            new = st.text_input(f"{ui} →", value=curr, key=f"map_{ui}")
            fm[ui] = new.strip() if isinstance(new, str) and new.strip() else curr

        # -------- Mapping presets: save/load/delete ----------
        st.markdown("**Mapping presets**")
        presets_dir = Path("mappings")
        presets_dir.mkdir(exist_ok=True)
        # Save current mapping
        preset_name = st.text_input("Preset name (save current)", value="", key="preset_name")
        if st.button("Save mapping preset"):
            name = preset_name.strip() or datetime.datetime.now().strftime("preset_%Y%m%d_%H%M%S")
            pfile = presets_dir / f"{name}.json"
            with open(pfile, "w", encoding="utf-8") as pf:
                json.dump(st.session_state["feature_map"], pf, indent=2)
            st.success(f"Saved preset: {pfile.name}")

        # List presets
        preset_files = [p.name for p in presets_dir.glob("*.json")]
        sel = st.selectbox("Load preset", options=["-- none --"] + preset_files, index=0, key="preset_select")
        if sel and sel != "-- none --":
            if st.button("Load selected preset"):
                with open(presets_dir / sel, "r", encoding="utf-8") as pf:
                    st.session_state["feature_map"] = json.load(pf)
                st.success(f"Loaded {sel}")
                st.experimental_rerun()
        if preset_files:
            if st.button("Delete all presets"):
                for p in presets_dir.glob("*.json"):
                    p.unlink()
                st.success("Deleted presets")
                st.experimental_rerun()

        if st.button("Reset mapping to defaults"):
            st.session_state["feature_map"] = default_map.copy()
            st.experimental_rerun()
        st.session_state["feature_map"] = fm

    # artifacts / debug
    st.markdown("---")
    st.markdown("### 🔧 Artifacts / Debug")
    with st.expander("View artifacts (meta & labels)", expanded=False):
        st.write("meta:", meta)
        st.write("labels (first 10):", label_classes[:10] if label_classes else None)

    st.markdown("---")
    st.caption("Use 'Recommend' in main area to get predictions.")

# ---- Prediction helper (uses mapping from session_state) ----
def build_input_df(values_list):
    # UI order -> create DataFrame with UI column names then rename to model expected
    X_arr = np.asarray(values_list, dtype=float).reshape(1, -1)
    try:
        X_df = pd.DataFrame(X_arr, columns=UI_FEATURES)
    except Exception:
        X_df = pd.DataFrame(X_arr)
    # apply mapping overrides
    fmap = st.session_state.get("feature_map", {})
    if fmap:
        # only keep non-empty mappings
        rename_map = {k: v for k, v in fmap.items() if isinstance(v, str) and v}
        if rename_map:
            X_df = X_df.rename(columns=rename_map)
    # try to reorder columns to expected where possible
    expected = expected_from_meta
    cols_to_use = [c for c in expected if c in X_df.columns]
    if cols_to_use:
        X_df = X_df[cols_to_use + [c for c in X_df.columns if c not in cols_to_use]]
    return X_df

def get_top_crops_safe(model, X_df, top_n=5):
    if model is None:
        return []
    try:
        if not hasattr(model, "predict_proba"):
            pred = model.predict(X_df)[0]
            return [{"crop": str(pred), "confidence": None}]
        probs = model.predict_proba(X_df)[0]
    except Exception as e:
        st.error("Prediction failed — model expects different columns or mapping.")
        st.write("Model expected (example):", expected_from_meta)
        st.write("Dataframe columns sent:", list(X_df.columns))
        st.write("Feature map used:", st.session_state.get("feature_map"))
        st.write("Error:", e)
        return []
    idxs = np.argsort(probs)[::-1][:top_n]
    results = []
    for i in idxs:
        label = None
        try:
            if label_classes and i < len(label_classes):
                label = label_classes[i]
        except Exception:
            label = None
        if not label:
            label = str(model.classes_[i])
        results.append({"crop": label, "confidence": float(probs[i])})
    return results

# ---- Main UI ----
st.markdown("<div class='card'><h1>🌱 GreenPulse</h1><p class='muted'>Empowering Farmers with Smart AI Crop Recommendations</p></div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Controls")
    st.write("Use the sidebar to set input values and mapping. Click Recommend to see predictions.")
    if model is None:
        st.error("Model artifact not found in artifacts/model.pkl — place the trained pipeline there.")
    recommend = st.button("🔍 Recommend")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Batch CSV upload / bulk predictions ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Batch predictions (CSV)")
    st.write("Upload CSV with columns matching UI names or model expected names. Mapping will be applied.")
    uploaded = st.file_uploader("Upload CSV for batch predict", type=["csv"], key="batch_csv")
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.write("Preview (first 5 rows):")
            st.dataframe(df_upload.head())
            if st.button("Run batch predictions"):
                # Build X_df from upload using mapping: if upload already has expected columns, use them; else use mapping
                fmap = st.session_state.get("feature_map", {})
                # Prepare columns in UI order or expected order
                rows_out = []
                for _, row in df_upload.iterrows():
                    # extract input values using mapping fallbacks
                    vals = []
                    for ui in UI_FEATURES:
                        mapped = fmap.get(ui)
                        # if input csv has mapped column, use that; else if has ui name, use; else try normalized match
                        if mapped and mapped in row:
                            vals.append(row[mapped])
                        elif ui in row:
                            vals.append(row[ui])
                        else:
                            # try normalized search
                            found = None
                            for c in df_upload.columns:
                                if norm(c) == norm(mapped or "") or norm(c) == norm(ui):
                                    found = row[c]; break
                            if found is not None:
                                vals.append(found)
                            else:
                                vals.append(np.nan)
                    rows_out.append(vals)
                X_batch = pd.DataFrame(rows_out, columns=UI_FEATURES)
                # apply mapping rename to X_batch for model
                rename_map = {k: v for k, v in fmap.items() if isinstance(v, str) and v}
                if rename_map:
                    X_model = X_batch.rename(columns=rename_map)
                else:
                    X_model = X_batch
                # reorder to expected
                expected = expected_from_meta
                cols_to_use = [c for c in expected if c in X_model.columns]
                if cols_to_use:
                    X_model = X_model[cols_to_use]
                # predict
                try:
                    if not hasattr(model, "predict_proba"):
                        preds = model.predict(X_model)
                        probs = [None]*len(preds)
                    else:
                        probs_arr = model.predict_proba(X_model)
                        preds = [str(model.classes_[np.argmax(p)]) for p in probs_arr]
                        probs = [float(max(p)) for p in probs_arr]
                except Exception as e:
                    st.error("Batch predict failed: " + str(e))
                    preds = []
                    probs = []
                # attach results to original df
                out_df = df_upload.copy()
                out_df["predicted_crop"] = preds
                out_df["pred_confidence"] = probs
                st.success("Batch prediction finished - preview:")
                st.dataframe(out_df.head())
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download results CSV", data=csv_bytes, file_name="greenpulse_batch_results.csv", mime="text/csv")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    # Quick stats / last inputs
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Last Inputs")
    last_vals = {"N": N, "P": P, "K": K, "Temp": Temp, "Humidity": Hum, "ph": ph, "Rain": Rain}
    st.json(last_vals)
    st.markdown("</div>", unsafe_allow_html=True)

    # small mapping export/import in main UI
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Mapping Presets")
    if st.button("Export current mapping JSON"):
        try:
            data = json.dumps(st.session_state.get("feature_map", {}), indent=2)
            st.download_button("📥 Download mapping JSON", data=data, file_name="feature_map.json", mime="application/json")
        except Exception:
            st.error("Failed to export mapping")
    uploaded_map = st.file_uploader("Upload mapping JSON", type=["json"], key="upload_map")
    if uploaded_map:
        try:
            mp = json.load(uploaded_map)
            if isinstance(mp, dict):
                st.session_state["feature_map"] = mp
                st.success("Mapping uploaded and applied")
            else:
                st.error("Uploaded file not a mapping dict")
        except Exception as e:
            st.error("Invalid JSON: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

# analytics log in session
if "analytics" not in st.session_state:
    st.session_state["analytics"] = []

if recommend:
    X_df = build_input_df([N, P, K, Temp, Hum, ph, Rain])
    results = get_top_crops_safe(model, X_df, top_n=5)

    # store analytics
    log_entry = {"N": N, "P": P, "K": K, "Temp": Temp, "Humidity": Hum, "ph": ph, "Rain": Rain, "results": results}
    st.session_state["analytics"].append(log_entry)

    # Results card: top crop highlight + confidence
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recommended Crops")

    if not results:
        st.info("No results — check artifacts, feature mapping, or model.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        top = results[0]
        conf = top.get("confidence")
        st.markdown(f"<h2 style='margin:6px 0 4px 0'>🌾 {top['crop']}</h2>", unsafe_allow_html=True)
        if conf is None:
            st.markdown("<div class='muted'>Confidence unavailable</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='muted'>Confidence: {conf*100:.2f}%</div>", unsafe_allow_html=True)

        # show ranked list and interactive plot (robust handling of None)
        try:
            df_res = pd.DataFrame(results)
            if "confidence" not in df_res.columns:
                df_res["confidence"] = None
            # replace None with 0.0 for numeric operations, keep original for display if needed
            df_res["confidence_filled"] = df_res["confidence"].fillna(0.0).astype(float)
            df_res["confidence_pct"] = (df_res["confidence_filled"] * 100).round(2)
            st.markdown("**Top predictions**")
            st.dataframe(df_res[["crop", "confidence_pct"]].rename(columns={"crop": "Crop", "confidence_pct": "Confidence (%)"}))

            # interactive bar chart (only if there is at least one non-zero confidence or we still want to show)
            if not df_res.empty:
                fig = px.bar(df_res.sort_values("confidence_filled", ascending=True),
                             x="confidence_filled", y="crop", orientation="h",
                             labels={"confidence_filled": "Confidence"}, color="confidence_filled",
                             color_continuous_scale="greens")
                fig.update_layout(margin=dict(l=8, r=8, t=24, b=8), height=320,
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Unable to render predictions table/plot.")
            st.write("Error:", e)

        st.markdown("</div>", unsafe_allow_html=True)

        # Downloads and analytics card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Logs & Export")
        # build log DataFrame safely
        try:
            df_log = pd.DataFrame([{
                "N": e.get("N"), "P": e.get("P"), "K": e.get("K"), "Temp": e.get("Temp"),
                "Humidity": e.get("Humidity"), "ph": e.get("ph"), "Rain": e.get("Rain"),
                "top_crop": (e.get("results")[0].get("crop") if e.get("results") else ""),
                "top_confidence": (e.get("results")[0].get("confidence") if e.get("results") else None)
            } for e in st.session_state.get("analytics", [])])
        except Exception:
            df_log = pd.DataFrame(columns=["N","P","K","Temp","Humidity","ph","Rain","top_crop","top_confidence"])

        st.dataframe(df_log.tail(10))

        # CSV download (always safe)
        try:
            csv_buf = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV (logs)", data=csv_buf, file_name="greenpulse_logs.csv", mime="text/csv")
        except Exception:
            st.warning("CSV export currently unavailable.")

        # PDF fallback: only attempt if there is a top result
        if results and results[0]:
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet

                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = [Paragraph("GreenPulse Recommendation Report", styles["Title"]), Spacer(1, 12)]
                if top.get("confidence") is not None:
                    story.append(Paragraph(f"Top recommendation: {top['crop']} ({(top['confidence']*100):.2f}%)", styles["Normal"]))
                else:
                    story.append(Paragraph(f"Top recommendation: {top['crop']}", styles["Normal"]))
                story.append(Spacer(1, 12))
                story.append(Paragraph("Inputs:", styles["Heading3"]))
                for k, v in last_vals.items():
                    story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
                doc.build(story)
                buffer.seek(0)
                st.download_button("📄 Download PDF (report)", data=buffer, file_name="GreenPulse_Report.pdf", mime="application/pdf")
            except Exception:
                st.info("PDF export unavailable (reportlab not installed or failed). You can still download CSV logs.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Disease diagnosis & protection recommendations ---
    with st.expander("🩺 Disease Diagnosis & Protection", expanded=False):
        st.markdown("Use visible symptoms to get likely disease(s) and practical protection steps.")
        # default crop selection: prefer top predicted crop otherwise allow manual
        default_crop = top['crop'] if (results and len(results) > 0) else (label_classes[0] if label_classes else "")
        crop_choice = st.selectbox("Crop (select target crop)", options=[default_crop] + ([c for c in (label_classes or []) if c != default_crop][:20]), index=0)
        st.markdown("Select observed symptoms (or type custom ones).")
        selected_symptoms = st.multiselect("Symptoms (choose)", options=_symptom_options, default=None, key="disease_symptoms")
        custom_sym = st.text_input("Add custom symptom (comma-separated)", value="", placeholder="e.g. yellowing of veins, sticky sap")
        if custom_sym:
            # split and append
            more = [s.strip() for s in custom_sym.split(",") if s.strip()]
            for m in more:
                if m and m not in selected_symptoms:
                    selected_symptoms.append(m)

        # optional image upload for future image-model prediction (not required)
        st.markdown("Optional: upload leaf image (if you have an image model in artifacts).")
        img_file = st.file_uploader("Leaf image (optional)", type=["png","jpg","jpeg"], key="disease_image")

        if st.button("Diagnose Disease"):
            if not selected_symptoms and not img_file:
                st.warning("Provide at least one symptom or an image.")
            else:
                # 1) if image model present, attempt image-based prediction (best-effort)
                image_based = None
                if disease_model and img_file:
                    try:
                        # best-effort: if model expects preprocessed features this will fail; keep non-fatal
                        # here we do NOT implement heavy image preprocessing; leave as placeholder
                        image_based = None
                    except Exception:
                        image_based = None

                # 2) symptom-based rule matching
                matches = diagnose_by_symptoms(selected_symptoms, crop_choice, disease_info, top_n=3)
                if not matches and not image_based:
                    st.info("No likely diseases found from symptoms. Consider expanding symptom list or consult an expert.")
                else:
                    if image_based and not matches:
                        st.markdown("**Image-based prediction** (model output)")
                        st.write(image_based)
                    else:
                        st.markdown("**Likely diseases (ranked)**")
                        for m in matches:
                            e = m["entry"]
                            sc = m["score"]
                            st.markdown(f"- **{e['disease']}** (score: {sc:.2f}) — severity: {e.get('severity', 'unknown')}")
                            st.markdown("  Recommendations:")
                            for rec in e.get("recommendations", []):
                                st.markdown(f"  - {rec}")
                            st.markdown("")  # spacing

                    # allow download of short recommendations
                    try:
                        # prepare plain text recommendations
                        txt_lines = []
                        txt_lines.append(f"GreenPulse - Disease Diagnosis Report\nCrop: {crop_choice}\nDate: {datetime.datetime.utcnow().isoformat()}Z\n")
                        txt_lines.append("Observed symptoms: " + ", ".join(selected_symptoms))
                        txt_lines.append("\nTop matches:\n")
                        for m in matches:
                            e = m["entry"]
                            txt_lines.append(f"- {e['disease']} (score: {m['score']:.2f}, severity: {e.get('severity', 'unknown')})")
                            txt_lines.append("  Recommendations:")
                            for rec in e.get("recommendations", []):
                                txt_lines.append(f"    * {rec}")
                            txt_lines.append("")
                        txt_bytes = ("\n".join(txt_lines)).encode("utf-8")
                        st.download_button("📥 Download diagnosis (txt)", data=txt_bytes, file_name="disease_diagnosis.txt", mime="text/plain")
                    except Exception:
                        st.warning("Download unavailable.")
    # ...existing code: continue logs, PDF, footer ...
# ...existing code...
