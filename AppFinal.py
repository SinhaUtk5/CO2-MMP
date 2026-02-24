import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json

st.set_page_config(page_title="UH-MMP Calculator", layout="wide")

# --------------------------
# RF JSON model loader + inference
# --------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_JSON_PATH = APP_DIR / "mmp_model_full.json"


@st.cache_resource
def load_rf_json_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing RF JSON model file: {path.name}")

    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    return model


def _predict_tree(nodes: list, x_row: np.ndarray) -> float:
    idx = 0
    while True:
        node = nodes[idx]
        if "value" in node:
            return float(node["value"])
        f = int(node["feature"])
        thr = float(node["threshold"])
        idx = int(node["left"]) if x_row[f] <= thr else int(node["right"])


def rf_predict(model: dict, X: np.ndarray) -> np.ndarray:
    trees = model["trees"]
    preds = np.zeros(X.shape[0], dtype=float)

    for t in trees:
        nodes = t["nodes"]
        for i in range(X.shape[0]):
            preds[i] += _predict_tree(nodes, X[i])

    preds /= float(len(trees))
    return preds


rf_model = load_rf_json_model(MODEL_JSON_PATH)
st.success("Random Forest JSON model loaded.")

# --------------------------
# UI: Input & Prediction
# --------------------------
st.header("Input & Prediction")

st.markdown(
    "Enter the composition up to C7+, MWC7+ and temperature (symb(Deg.) C)"
)

uploaded = st.file_uploader("Upload the input CSV file here", type=["csv"])

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded, header=None)
        st.session_state["input_df"] = df_in
        st.success("CSV loaded.")
        st.dataframe(df_in.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to enable prediction.")
