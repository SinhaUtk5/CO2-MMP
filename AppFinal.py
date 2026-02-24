
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json

st.set_page_config(page_title="UH-MMP Calculator", layout="wide")

# --------------------------
# Header / Title / Links
# --------------------------
st.title("UH-MMP Calculator")
st.markdown("A product of Interaction of Phase-Behavior and Flow (IPB&F) Consortium")

st.markdown("**Developed by** - Utkarsh Sinha, Dr. Birol Dindoruk and Dr. M.Y. Soliman")

st.markdown(
    "[Ref. - Sinha, U., Dindoruk, B., & Soliman, M. (2021). Prediction of CO2 Minimum Miscibility Pressure Using an Augmented Machine-Learning-Based Model. SPE Journal, 1-13.](https://doi.org/10.2118/200326-PA)"
)
st.markdown(
    "**Product Description:** Calculates the Minimum Miscibility Pressure (psia) for pure-CO2 injection."
)
st.markdown(
    "1) **Download the example input CSV template file**: "
    "[Click here](https://drive.google.com/file/d/1S60FPJcZbqxhCGJ3xTUMhniTqZK0jHC2/view?usp=sharing)"
)
st.markdown(
    "2) **Watch short help video (no sound)**: "
    "[Click here](https://drive.google.com/file/d/1yst9Xzy0wvapuigszIOOZhuhD4NLbH1H/view?usp=sharing)"
)
st.divider()

# --------------------------
# Contributor images / blocks
# --------------------------
APP_DIR = Path(__file__).resolve().parent


def show_resized_image(img_name: str, target_height: int = 200):
    img_path = APP_DIR / img_name
    if not img_path.exists():
        st.warning(f"Missing image file: {img_name} (place it next to this app if you want it shown).")
        return
    img = Image.open(img_path)
    w, h = img.size
    new_h = target_height
    new_w = int(w * (new_h / h))
    st.image(img.resize((new_w, new_h)))


c1, c2 = st.columns([1, 1])
with c1:
    show_resized_image("dindoruk_birol_2023_ns.png", 180)
    st.markdown(
        "**Dr. Birol Dindoruk**  \nProfessor  \nHarold Vance Department of Petroleum Engineering,  \nTexas A&M University"
    )
with c2:
    show_resized_image("Utk.jpeg", 180)
    st.markdown(
        "**Utkarsh Sinha**  \nVolunteer Research Associate  \nInteraction of Phase-Behavior and Flow (IPB&F) Consortium"
    )

st.divider()

# --------------------------
# RF JSON model loader + inference
# --------------------------
MODEL_JSON_PATH = APP_DIR / "mmp_model_full.json"


@st.cache_resource
def load_rf_json_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing RF JSON model file: {path.name} (place it next to the app).")

    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    # Minimal validation
    if model.get("model_type") not in {"random_forest_regressor", "rf_regressor", "random_forest"}:
        raise ValueError(
            "JSON model does not look like a Random Forest JSON.\n"
            "Expected model_type in {'random_forest_regressor','rf_regressor','random_forest'}."
        )
    if "trees" not in model or not isinstance(model["trees"], list) or len(model["trees"]) == 0:
        raise ValueError("JSON model must contain a non-empty list: model['trees'].")

    return model


def _predict_tree(nodes: list, x_row: np.ndarray) -> float:
    """
    One tree prediction for one row.
    Expected node format:
      - internal node: {"feature": int, "threshold": float, "left": int, "right": int}
      - leaf node: {"value": float}
    """
    idx = 0
    while True:
        node = nodes[idx]
        if "value" in node:
            return float(node["value"])
        f = int(node["feature"])
        thr = float(node["threshold"])
        idx = int(node["left"]) if x_row[f] <= thr else int(node["right"])


def rf_predict(model: dict, X: np.ndarray) -> np.ndarray:
    """
    Predict y for all rows in X using RF JSON model.
    Aggregation = mean across trees (standard RF regressor).
    """
    trees = model["trees"]
    n_rows = X.shape[0]
    preds = np.zeros(n_rows, dtype=float)

    for t in trees:
        nodes = t["nodes"]
        # accumulate tree preds
        for i in range(n_rows):
            preds[i] += _predict_tree(nodes, X[i])

    preds /= float(len(trees))
    return preds


# Load model
try:
    rf_model = load_rf_json_model(MODEL_JSON_PATH)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# --------------------------
# Feature-engineering function (match R code exactly)
# --------------------------
def build_cf_from_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must be positional columns exactly like Shiny app (>=12 cols), headerless.
    Returns cf with the same engineered columns used in R.
    """
    if df.shape[1] < 12:
        raise ValueError("Input CSV must contain at least 12 columns (positional). Use the template.")

    arr = df.astype(float).to_numpy()
    col = lambda i: arr[:, i - 1]  # R 1-based -> Python 0-based

    H2S = np.power(col(1), 0.8)
    N2 = col(3)
    Co2 = np.power(col(2), 1.38)
    C1 = np.power(col(4), 0.7)
    C2 = np.power(col(5), 0.8)
    C3 = np.power(col(6), 0.8)
    C4 = np.power(col(7), 0.8)
    C5 = col(8)
    C6 = col(9)
    C7p = np.power(col(10), 0.38)
    MWC7p = np.power(col(11), 0.9)
    Tres = col(12)

    MW_oil1 = (
        col(1) * 34.1
        + col(2) * 44.01
        + col(4) * 16.04
        + col(5) * 30.07
        + col(3) * 28.0134
        + col(6) * 44.1
        + col(7) * 58.16
        + col(8) * 72.15
        + col(9) * 86.18
        + col(10) * col(11)
    ) / 100.0
    MW_oil = np.power(MW_oil1, -2)

    MW_ap_C7p1 = col(10) * col(11) / 100.0
    MW_ap_C7p = np.power(MW_ap_C7p1, -1.9)

    frn1 = (1 + col(3) + col(4)) / (1 + col(1) + col(2) + col(5) + col(6) + col(7) + col(8) + col(9))
    frn = np.power(frn1, 0.63)

    EVP1 = np.exp(8.243 / (1 + 0.002177 * col(12)) - 10.91)
    EVP = EVP1 / 14.7

    Tsq = np.power(col(12), 1.9)

    # Compute Pred exactly as in R (even if RF may/may not use it; keep for fidelity)
    A0 = -9092.137474
    A1 = 31.28911371
    A2 = 5.32589172
    A3 = -3.744553796
    A4 = -3.744840184
    A5 = 34.5695143
    A6 = 24.45732707
    A7 = 14.2445869
    A8 = 0.538538391
    A9 = -3.915630434
    A10 = 600.9850745
    A11 = 446.2606576
    A12 = 17.75623621
    A13 = -1328.354546
    A14 = 6407937.97
    A15 = 39.14271794
    A16 = -0.002677839
    A17 = 78781.43976

    Part1 = A0 + A1 * H2S + A2 * Co2 + A3 * N2 + A4 * C1 + A5 * C2 + A6 * C3
    Part2 = A7 * C4 + C5 * A8 + C6 * A9 + C7p * A10 + A11 * frn
    Part3 = A12 * (MWC7p) * (1 + A13 * np.power(col(11), -2.8) * np.power((col(10) / 100.0), -1.9))
    Part4 = (
        A14 * MW_oil
        + A15 * Tres
        + A15 * A16 * Tsq
        + A17 * np.exp(8.243 / (1 + 0.002177 * col(12)) - 10.91)
    )
    Pred = Part1 + Part2 + Part3 + Part4

    cf = pd.DataFrame(
        {
            "H2S": H2S,
            "N2": N2,
            "Co2": Co2,
            "C1": C1,
            "C2": C2,
            "C3": C3,
            "C4": C4,
            "C5": C5,
            "C6": C6,
            "C7p": C7p,
            "MWC7p": MWC7p,
            "Tres": Tres,
            "MW_oil": MW_oil,
            "MW_ap_C7p": MW_ap_C7p,
            "frn": frn,
            "Tsq": Tsq,
            "EVP": EVP,
            "Pred": Pred,
        }
    )
    return cf


def predict_mmp(df_in: pd.DataFrame) -> pd.DataFrame:
    cf = build_cf_from_input(df_in)

    # Ensure model feature count alignment
    X = cf.to_numpy(dtype=float)
    expected = rf_model.get("n_features", X.shape[1])
    if X.shape[1] != expected:
        raise ValueError(
            f"Feature count mismatch. Model expects n_features={expected}, but computed features={X.shape[1]}.\n"
            f"Computed columns: {list(cf.columns)}"
        )

    # RF prediction
    y = rf_predict(rf_model, X).reshape(-1)

    # B coefficients from Shiny app
    B0 = 0.0000003179012
    B1 = 0.00033465866193139
    B2 = 4.24937726064401
    B3 = -1926.85329595309
    B4 = 0.002159795

    mmp = (B0 * y**3 + B1 * y**2 + B2 * y + B3) / (1.0 + B4 * y)

    out = df_in.copy().reset_index(drop=True)
    out["MMP_predicted_psia"] = mmp
    return out


# --------------------------
# UI: file uploader, run button, show results & download
# --------------------------
st.header("Inputs")

st.header("Input")

st.markdown("**Enter the composition up to C7+, MWC7+ and Temperature (°C).**")


uploaded = st.file_uploader("Upload the input CSV file here", type=["csv"])
if uploaded is not None:
    try:
        # IMPORTANT: match your “headerless positional columns” assumption
        input_df = pd.read_csv(uploaded, header=None)

        # Drop totally empty columns (common when CSV has a trailing comma)
        input_df = input_df.dropna(axis=1, how="all")

        st.session_state["input_df"] = input_df
        st.success(f"Loaded CSV with shape: {input_df.shape} (rows, cols)")
        st.dataframe(input_df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")



run_clicked = st.button("Run prediction", type="primary", disabled=("input_df" not in st.session_state))

if run_clicked:
    try:
        result_df = predict_mmp(st.session_state["input_df"])
        st.session_state["result_df"] = result_df
        st.success("Prediction complete.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

if "result_df" in st.session_state:
    st.subheader("Results (first 20 rows)")
    st.dataframe(st.session_state["result_df"].head(20), use_container_width=True)

    csv_bytes = st.session_state["result_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download the MMP results (CSV)",
        data=csv_bytes,
        file_name=f"MMP_Results-{pd.Timestamp.today().date()}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    "**Reference:** Sinha, U., Dindoruk, B., & Soliman, M. (2021). Prediction of CO2 Minimum Miscibility Pressure Using an Augmented Machine-Learning-Based Model. SPE Journal, 1-13."
)





