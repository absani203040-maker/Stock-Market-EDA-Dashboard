import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Stock Market EDA Dashboard",
    page_icon="📈",
    layout="wide"
)

sns.set_theme(style="whitegrid")

st.title("📈 Stock Market EDA Dashboard")
st.write(
    "Explore stock market data through interactive charts, "
    "statistical summaries, price trends, and correlation analysis."
)

# ==================== 1. UPLOAD DATASET ====================

uploaded_file = st.sidebar.file_uploader(
    "Upload your stock market CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file using the sidebar to start the analysis.")
    st.markdown(
        """
        **Recommended columns:** Date, Open, High, Low, Close, Volume.

        You can also upload other CSV datasets for general exploratory data analysis.
        """
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

df = df.copy()

# ==================== 2. DATA CLEANING ====================

st.sidebar.header("Data Cleaning")

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows",
    value=True
)

duplicate_count = int(df.duplicated().sum())

if remove_duplicates:
    df = df.drop_duplicates().copy()

for col in df.columns:
    if df[col].dtype == "object":
        cleaned = (
            df[col]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        converted = pd.to_numeric(cleaned, errors="coerce")
        valid_count = int(df[col].notna().sum())

        if valid_count > 0:
            conversion_rate = int(converted.notna().sum()) / valid_count

            if conversion_rate >= 0.8:
                df[col] = converted

# ==================== 3. DETECT COLUMNS ====================

def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(dataframe, possible_names):
    normalized_columns = {
        normalize_column_name(col): col
        for col in dataframe.columns
    }

    for name in possible_names:
        key = normalize_column_name(name)

        if key in normalized_columns:
            return normalized_columns[key]

    return None


date_col = find_column(
    df, ["Date", "Datetime", "Timestamp", "Time"]
)

open_col = find_column(
    df, ["Open", "OpeningPrice"]
)

high_col = find_column(
    df, ["High", "HighestPrice"]
)

low_col = find_column(
    df, ["Low", "LowestPrice"]
)

close_col = find_column(
    df, ["Close", "ClosingPrice", "Adj Close", "Adjusted Close"]
)

volume_col = find_column(
    df, ["Volume", "TradingVolume"]
)

if date_col is not None:
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")

    if parsed_dates.notna().any():
        df[date_col] = parsed_dates
        df = df.sort_values(
            date_col,
            na_position="last"
        ).reset_index(drop=True)
    else:
        date_col = None

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# ==================== 4. DATASET OVERVIEW ====================

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Total Columns", f"{len(df.columns):,}")
col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
col4.metric("Duplicate Rows", f"{duplicate_count:,}")

tab1, tab2, tab3 = st.tabs(
    ["Data Preview", "Column Information", "Statistics"]
)

with tab1:
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("View last 10 rows"):
        st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    column_info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [
            int(df[col].isna().sum()) for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique()) for col in df.columns
        ]
    })

    st.dataframe(column_info, use_container_width=True)

with tab3:
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

st.download_button(
    label="Download Cleaned Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_stock_data.csv",
    mime="text/csv"
)

# ==================== 5. MISSING VALUES ====================

st.header("2. Missing Values Analysis")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    st.success("No missing values found.")
else:
    missing_info = missing.rename("Missing Count").to_frame()
    missing_info["Missing Percentage"] = (
        missing_info["Missing Count"] / len(df) * 100
    ).round(2)

    st.dataframe(missing_info, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    missing.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ==================== 6. NUMERICAL ANALYSIS ====================

st.header("3. Numerical Analysis")

if numeric_cols:
    selected_numeric = st.multiselect(
        "Select numerical columns",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    for start in range(0, len(selected_numeric), 2):
        selected_group = selected_numeric[start:start + 2]
        chart_columns = st.columns(len(selected_group))

        for index, col in enumerate(selected_group):
            with chart_columns[index]:
                fig, ax = plt.subplots(figsize=(6, 4))

                sns.histplot(
                    df[col].dropna(),
                    bins=30,
                    kde=True,
                    ax=ax
                )

                ax.set_title(f"Distribution of {col}")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

    with st.expander("View numerical boxplots"):
        for start in range(0, len(selected_numeric), 2):
            selected_group
