# streamlit_co2_mmp.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import joblib
import xgboost as xgb
import json
import io

st.set_page_config(page_title="UH-MMP Calculator", layout="wide")

# --------------------------
# Header / Title / Links
# --------------------------
st.title("UH-MMP Calculator")
st.markdown("A product of Interaction of Phase-Behavior and Flow (IPB&F) Consortium")
st.markdown(
    '**Department of Petroleum Engineering**: "Interaction of Phase Behavior and Flow in Porous Media (IPBFPM) Consortium"'
)
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
        st.warning(
            f"Missing image file: {img_name} (place it next to this app if you want it shown)."
        )
        return
    img = Image.open(img_path)
    w, h = img.size
    new_h = target_height
    new_w = int(w * (new_h / h))
    img_resized = img.resize((new_w, new_h))
    st.image(img_resized)


col1, col2 = st.columns([1, 1])
with col1:
    show_resized_image("dindoruk_birol_2023_ns.png", 180)
    st.markdown(
        "**Dr. Birol Dindoruk**  \nProfessor  \nHarold Vance Department of Petroleum Engineering,  \nTexas A&M University"
    )
with col2:
    show_resized_image("Utk.jpeg", 180)
    st.markdown(
        "**Utkarsh Sinha**  \nVolunteer Research Associate  \nInteraction of Phase-Behavior and Flow (IPB&F) Consortium"
    )

st.divider()

# --------------------------
# Model loading utilities
# --------------------------
MODEL_JSON_PATH = "mmp_model_full.json"  # preferred per your note
MODEL_JOBLIB_PATH = "mmp-rf-SL-final.joblib"  # fallback if you used joblib


@st.cache_resource
def try_load_model(
    json_path: str = MODEL_JSON_PATH, joblib_path: str = MODEL_JOBLIB_PATH
):
    """
    Tries to load model in this order:
      1) xgboost.Booster() from JSON (xgboost.save_model(..., format='json'))
      2) sklearn-like model via joblib (RandomForestRegressor saved with joblib.dump)
    Returns a tuple: (model_object, model_type) where model_type is one of {'xgb_booster', 'sklearn'}.
    """
    # 1) Try to load as XGBoost Booster JSON
    json_file = Path(json_path)
    if json_file.exists():
        try:
            booster = xgb.Booster()
            booster.load_model(str(json_file))
            return booster, "xgb_booster"
        except Exception as e:
            # not xgboost JSON or failed load — continue to try joblib
            st.warning(
                f"Found {json_file}, but failed to load as xgboost.Booster(): {e}"
            )

    # 2) Try joblib (sklearn)
    joblib_file = Path(joblib_path)
    if joblib_file.exists():
        try:
            obj = joblib.load(str(joblib_file))
            return obj, "sklearn"
        except Exception as e:
            st.warning(f"Found {joblib_file}, but failed to load via joblib: {e}")

    # 3) If nothing found, raise
    raise FileNotFoundError(
        f"No model found. Place either:\n - an XGBoost JSON model at: {json_path} (preferred),\n"
        f" - or a scikit-learn model saved with joblib at: {joblib_path}."
    )


# Attempt load and provide useful error messaging
try:
    model_obj, model_type = try_load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error while loading model: {e}")
    st.stop()

st.success(f"Model loaded ({model_type}).")


# --------------------------
# Feature-engineering function (match R code exactly)
# --------------------------
def build_cf_from_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the uploaded input dataframe (positional columns as in Shiny app),
    returns the 'cf' DataFrame with columns:
    H2S,N2,Co2,C1,C2,C3,C4,C5,C6,C7p,MWC7p,Tres,MW_oil,MW_ap_C7p,frn,Tsq,EVP,Pred_placeholder
    (we will not compute Pred here — that's part of the R code but final prediction uses the model's y).
    """
    # ensure at least 12 columns (R code referenced up to column 12)
    if df.shape[1] < 12:
        raise ValueError(
            "Input CSV must contain at least 12 columns (positional). See Shiny template."
        )

    # Convert to numeric (coerce)
    arr = df.copy().astype(float).to_numpy()
    num = arr.shape[0]

    # Indexing: R code uses df[1:num,1] etc where R columns are 1-based.
    col = lambda i: arr[:, i - 1]  # i is 1-based column index

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

    # MW_oil1 = ((df[1:num,1 ])*34.1+(df[1:num,2 ])*44.01+df[1:num,4 ]*16.04+df[1:num,5 ]*30.07+df[1:num,3 ]*28.0134+df[1:num,6 ]*44.1+df[1:num,7 ]*58.16+df[1:num,8 ]*72.15+df[1:num,9 ]*86.18+df[1:num,10 ]*df[1:num,11 ])/100
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

    frn1 = (1 + col(3) + col(4)) / (
        1 + col(1) + col(2) + col(5) + col(6) + col(7) + col(8) + col(9)
    )
    frn = np.power(frn1, 0.63)

    EVP1 = np.exp(8.243 / (1 + 0.002177 * col(12)) - 10.91)
    EVP = EVP1 / 14.7

    Tsq = np.power(col(12), 1.9)

    # The R code also defines A0..A17 and computes Part1..Part4 and Pred = Part1+Part2+Part3+Part4.
    # We'll keep Pred placeholder (not used by model) but compute it to remain faithful.
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
    Part3 = (
        A12
        * (MWC7p)
        * (1 + A13 * (np.power(col(11), -2.8)) * (np.power((col(10) / 100.0), -1.9)))
    )
    Part4 = (
        A14 * (MW_oil)
        + A15 * Tres
        + A15 * A16 * Tsq
        + A17 * np.exp((8.243 / (1 + 0.002177 * col(12)) - 10.91))
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


# --------------------------
# Prediction function
# --------------------------
def predict_mmp(df: pd.DataFrame, model_obj, model_type: str) -> pd.DataFrame:
    """
    Takes original input dataframe (positional), computes cf, predicts y from model,
    then computes MMP_predicted_psia using the B0..B4 rational polynomial.
    Returns original df with appended 'MMP_predicted_psia' column.
    """
    cf = build_cf_from_input(df)

    # Model expects columns in the same order as cf (H2S,N2, Co2, C1...Pred)
    X = cf.copy()

    # Predict y using appropriate API
    if model_type == "xgb_booster":
        # xgboost Booster expects DMatrix; ensure numeric numpy array
        dtest = xgb.DMatrix(X.values)
        y = model_obj.predict(dtest)
    else:
        # sklearn-like model (RandomForestRegressor etc.)
        y = model_obj.predict(X.values)

    # coerce y to 1D numpy
    y = np.asarray(y).reshape(-1)

    # B coefficients from the Shiny app
    B0 = 0.0000003179012
    B1 = 0.00033465866193139
    B2 = 4.24937726064401
    B3 = -1926.85329595309
    B4 = 0.002159795

    # In R: df$MMP_predicted_psia<-(B0*y*y*y+B1*y*y+B2*y+B3)/(1+B4*y)
    numer = B0 * (y**3) + B1 * (y**2) + B2 * y + B3
    denom = 1.0 + B4 * y
    mmp = numer / denom

    out = df.copy().reset_index(drop=True)
    out["MMP_predicted_psia"] = mmp
    return out


# --------------------------
# UI: file uploader, run button, show results & download
# --------------------------
st.header("Input & Prediction")

st.markdown(
    """
**Important:** Input CSV must follow the positional column layout used by the original Shiny app.
At minimum it must have 12 columns (R 1-based):  
1:H2S-like fraction, 2:CO2 fraction, 3:N2 fraction, 4:C1, 5:C2, 6:C3, 7:C4, 8:C5, 9:C6, 10:C7+, 11:MW C7+, 12:Temperature (Tres)
(Use the example template for exact formatting.)
"""
)

uploaded = st.file_uploader("Upload the input CSV file here", type=["csv"])

if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded, header=None)  # positional columns, so no header
        st.success("CSV loaded. Preview (first 10 rows):")
        st.dataframe(df_in.head(10), use_container_width=True)
        st.session_state["input_df"] = df_in
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to enable prediction.")

run_clicked = st.button(
    "Run prediction", type="primary", disabled=("input_df" not in st.session_state)
)

if run_clicked:
    try:
        result_df = predict_mmp(st.session_state["input_df"], model_obj, model_type)
        st.session_state["result_df"] = result_df
        st.success("Prediction complete.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

if "result_df" in st.session_state:
    st.subheader("Results (first 20 rows)")
    st.dataframe(st.session_state["result_df"].head(20), use_container_width=True)

    # prepare download
    csv_bytes = st.session_state["result_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download the MMP results (CSV)",
        data=csv_bytes,
        file_name=f"MMP_Results-{pd.Timestamp.today().date()}.csv",
        mime="text/csv",
    )

# --------------------------
# Footer / reference
# --------------------------
st.markdown("---")
st.markdown(
    "**Reference:** Sinha, U., Dindoruk, B., & Soliman, M. (2021). Prediction of CO2 Minimum Miscibility Pressure Using an Augmented Machine-Learning-Based Model. SPE Journal, 1-13."
)
