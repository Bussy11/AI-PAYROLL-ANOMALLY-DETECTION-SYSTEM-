import io
import os
import zipfile
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="IBS | AI Payroll Anomaly Detection & Forensic Audit",
    page_icon="🇳🇬",
    layout="wide",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9f8;
    }
    .nigeria-banner {
        background-color: #008751;
        height: 10px;
        border-radius: 4px;
        margin: 8px 0 16px 0;
    }
    .top-header {
        background-color: #ffffff;
        padding: 14px 18px;
        border-radius: 12px;
        border: 2px solid #008751;
        box-shadow: 0 4px 15px rgba(0, 135, 81, 0.15);
        text-align: center;
        margin: 12px 0 18px 0;
    }
    .top-title {
        color: #008751;
        font-weight: 900;
        font-size: 24px;
        text-transform: uppercase;
        margin: 4px 0;
    }
    .top-sub {
        color: #1b5e20;
        font-size: 14px;
        font-weight: 700;
        margin: 0;
    }
    .stButton > button {
        background-color: #008751;
        color: white;
        font-weight: bold;
        border: 1px solid #006b40;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #006b40;
        color: white;
        border: 1px solid #006b40;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_existing_image(candidates):
    """Return the first existing image filename from a list of candidates."""
    for filename in candidates:
        if os.path.isfile(filename):
            return filename
    return None


def display_image_or_placeholder(candidates, width, placeholder):
    """Display an image when available; otherwise display a clear placeholder."""
    image_path = find_existing_image(candidates)
    if image_path:
        st.image(image_path, width=width)
    else:
        st.info(placeholder)


def initialise_session_state():
    """Set safe defaults for the multi-page workflow."""
    defaults = {
        "auth_verified": False,
        "full_name": "",
        "address": "",
        "contact": "",
        "step": 1,
        "dev_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prepare_numeric_features(dataframe):
    """
    Prepare numeric model features while excluding likely identifier columns.
    Returns a cleaned feature table and a list of columns excluded from modelling.
    """
    numeric = dataframe.select_dtypes(include=[np.number]).copy()
    ignored_tokens = (
        "id", "employeeid", "staffid", "recordnumber",
        "bank_account", "accountnumber", "phone", "contact",
    )
    feature_columns = [
        col for col in numeric.columns
        if not any(token in str(col).lower().replace(" ", "").replace("_", "")
                   for token in ignored_tokens)
    ]
    numeric = numeric[feature_columns]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    numeric = numeric.fillna(0)
    numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1]
    return numeric


def build_sample_payroll():
    """Create a reproducible demonstration dataset with a few unusual records."""
    rng = np.random.default_rng(42)
    number_of_employees = 200

    data = pd.DataFrame(
        {
            "EmployeeID": [f"EMP{i:03d}" for i in range(1, number_of_employees + 1)],
            "Name": [f"Employee_{i:03d}" for i in range(1, number_of_employees + 1)],
            "State": rng.choice(["Kogi", "Kwara", "Ekiti"], number_of_employees),
            "LGA": rng.choice(["Okene", "Kabba", "Lokoja", "Ado"], number_of_employees),
            "BasicSalary": np.maximum(
                rng.normal(80000, 12000, number_of_employees), 30000
            ).round(2),
            "Allowances": np.maximum(
                rng.normal(15000, 4000, number_of_employees), 0
            ).round(2),
            "Deductions": np.maximum(
                rng.normal(5000, 1500, number_of_employees), 0
            ).round(2),
        }
    )
    data["NetSalary"] = (
        data["BasicSalary"] + data["Allowances"] - data["Deductions"]
    ).round(2)

    # Add demonstration anomalies for testing. These are not real payroll records.
    data.loc[10, "BasicSalary"] = 350000.00
    data.loc[10, "NetSalary"] = (
        data.loc[10, "BasicSalary"]
        + data.loc[10, "Allowances"]
        - data.loc[10, "Deductions"]
    )
    data.loc[25, "Allowances"] = 200000.00
    data.loc[25, "NetSalary"] = (
        data.loc[25, "BasicSalary"]
        + data.loc[25, "Allowances"]
        - data.loc[25, "Deductions"]
    )
    return data


def format_currency(value):
    """Format a numeric value as Nigerian naira for display."""
    try:
        return f"₦{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


# ============================================================
# INITIALISE SESSION STATE
# ============================================================
initialise_session_state()

# ============================================================
# PAGE HEADER: SCHOOL LOGO CENTRED; EFCC LEFT AND COAT OF ARMS RIGHT
# ============================================================
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# School logo is centred at the top.
school_left, school_center, school_right = st.columns([1, 2, 1])
with school_center:
    school_logo = find_existing_image(
        [
            "school_logo.jpeg", "school_logo.jpg", "school_logo.png",
            "logo.jpeg", "logo.jpg", "logo.png",
        ]
    )
    if school_logo:
        st.image(school_logo, width=170)
    else:
        st.markdown(
            "<div style='text-align:center;font-size:58px;'>🏫</div>"
            "<div style='text-align:center;font-weight:800;'>IBS SCHOOL LOGO</div>",
            unsafe_allow_html=True,
        )

# Government logos are immediately below the school logo:
# EFCC on the left, Coat of Arms on the right.
left_logo_col, middle_space, right_logo_col = st.columns([1, 0.35, 1])
with left_logo_col:
    st.markdown("<div style='display:flex;justify-content:center;'>", unsafe_allow_html=True)
    efcc_logo = find_existing_image(
        ["efcc_logo.png", "efcc_logo.jpeg", "efcc_logo.jpg", "EFCC_logo.png", "EFCC_logo.jpg"]
    )
    if efcc_logo:
        st.image(efcc_logo, width=135)
    else:
        st.markdown(
            "<div style='text-align:center;font-size:38px;'>🦅</div>"
            "<div style='text-align:center;font-weight:700;'>EFCC LOGO</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right_logo_col:
    st.markdown("<div style='display:flex;justify-content:center;'>", unsafe_allow_html=True)
    coat_logo = find_existing_image(
        [
            "coat_of_arms.png", "coat_of_arms.jpeg", "coat_of_arms.jpg",
            "coat-of-arms.png", "coat-of-arms.jpg",
        ]
    )
    if coat_logo:
        st.image(coat_logo, width=135)
    else:
        st.markdown(
            "<div style='text-align:center;font-size:38px;'>🇳🇬</div>"
            "<div style='text-align:center;font-weight:700;'>COAT OF ARMS</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="top-header">
        <p class="top-title">International Business School</p>
        <p class="top-sub">AI Payroll Anomaly Detection &amp; Forensic Audit Portal</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="nigeria-banner"></div>', unsafe_allow_html=True)

# ============================================================
# NAVIGATION MENU
# ============================================================
step_names = {
    1: "1. Authorization",
    2: "2. Data Input",
    3: "3. AI Analysis",
    4: "4. Visualizations",
    5: "5. Q&A & Kudos",
    6: "6. Developers",
}

st.markdown("### 📌 Navigation Menu")
navigation_columns = st.columns(6)
for step_number, step_label in step_names.items():
    with navigation_columns[step_number - 1]:
        button_label = (
            f"👉 [{step_number}]" if st.session_state.step == step_number
            else f"📍 {step_number}"
        )
        if st.button(
            button_label,
            use_container_width=True,
            key=f"nav_btn_{step_number}",
        ):
            if step_number > 1 and not st.session_state.auth_verified:
                st.warning(
                    "⛔ Access locked. Complete authorization on Step 1 first."
                )
                st.session_state.step = 1
            else:
                st.session_state.step = step_number
            st.rerun()

st.markdown(f"**Current Active Section:** `{step_names[st.session_state.step]}`")
st.divider()

# ============================================================
# PAGE 1: AUTHORIZATION & REGISTRATION
# ============================================================
if st.session_state.step == 1:
    st.header("🔑 Investigator Authorization & Registration")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Authorization Check")
        auth_choice = st.radio(
            "Do you have official authorization?",
            ["-- Please Select --", "YES", "NO"],
            index=0,
            key="authorization_choice",
        )
        if auth_choice == "YES":
            st.session_state.auth_verified = True
            st.success("✅ Authorization status confirmed by the user.")
        elif auth_choice == "NO":
            st.session_state.auth_verified = False
            st.error("⛔ Access denied. Official authorization is required.")
        else:
            st.session_state.auth_verified = False
            st.info("Please select your authorization status.")

    with col2:
        st.subheader("Investigator Details")
        st.session_state.full_name = st.text_input(
            "Full Name:", value=st.session_state.full_name
        )
        st.session_state.address = st.text_input(
            "Official Address:", value=st.session_state.address
        )
        st.session_state.contact = st.text_input(
            "Contact Number:", value=st.session_state.contact
        )

    if st.session_state.auth_verified and st.session_state.full_name.strip():
        st.success(
            f"Welcome, {st.session_state.full_name}. "
            "Your session details have been entered."
        )

    if st.button(
        "Continue to Data Input & Cleaning ➡️",
        use_container_width=True,
        key="continue_from_authorization",
    ):
        if not st.session_state.auth_verified:
            st.warning("Please confirm that you have official authorization.")
        elif not st.session_state.full_name.strip():
            st.warning("Please enter your full name.")
        else:
            st.session_state.step = 2
            st.rerun()

# ============================================================
# PAGE 2: DATA INPUT & CLEANING
# ============================================================
elif st.session_state.step == 2:
    st.header("📂 Payroll Data Input & Cleaning")
    data_choice = st.radio(
        "Select Dataset Source:",
        [
            "— Please Select —",
            "Generate Sample Payroll Dataset (200 Records)",
            "Upload External Payroll CSV File",
        ],
        horizontal=True,
        index=0,
        key="dataset_radio_choice",
    )

    dataframe = None
    if data_choice == "— Please Select —":
        st.info("Please select a dataset source above.")
        st.session_state.pop("df", None)

    elif "Generate" in data_choice:
        dataframe = build_sample_payroll()
        st.success(
            "Sample demonstration payroll dataset generated. "
            "It contains artificial test anomalies, not real employee data."
        )

    elif "Upload" in data_choice:
        uploaded_file = st.file_uploader(
            "Upload a payroll CSV file",
            type=["csv"],
            key="payroll_csv_uploader",
        )
        if uploaded_file is not None:
            try:
                dataframe = pd.read_csv(uploaded_file)
                if dataframe.empty:
                    st.error("The uploaded CSV contains no data rows.")
                    dataframe = None
                else:
                    st.success("CSV file loaded successfully.")
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
                st.error(f"Could not read the CSV file. Please check its format. Details: {error}")
                dataframe = None
        else:
            st.info("Awaiting CSV upload.")
            st.session_state.pop("df", None)

    if dataframe is not None:
        # Save a copy so later pages can use the selected dataset.
        st.session_state.df = dataframe.copy()
        st.subheader("Data Preview")
        st.caption(f"Rows: {len(dataframe):,} | Columns: {len(dataframe.columns):,}")
        st.dataframe(dataframe.head(10), use_container_width=True)

        with st.expander("Column names and missing-value overview"):
            overview = pd.DataFrame(
                {
                    "Column": dataframe.columns.astype(str),
                    "Data Type": [str(dtype) for dtype in dataframe.dtypes],
                    "Missing Values": dataframe.isna().sum().values,
                }
            )
            st.dataframe(overview, use_container_width=True)

    col_back, col_continue = st.columns(2)
    with col_back:
        if st.button("⬅️ Back to Authorization", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_continue:
        can_continue = (
            "df" in st.session_state
            and isinstance(st.session_state.df, pd.DataFrame)
            and not st.session_state.df.empty
        )
        if st.button(
            "Continue to AI Model Analysis ➡️",
            use_container_width=True,
            disabled=not can_continue,
        ):
            st.session_state.step = 3
            st.rerun()
        if not can_continue:
            st.caption("Select or generate a valid dataset to continue.")

# ============================================================
# PAGE 3: AI MODEL ANALYSIS
# ============================================================
elif st.session_state.step == 3:
    if not st.session_state.auth_verified:
        st.error("⛔ Access locked. Complete authorization on Step 1.")
        st.stop()

    st.header("🤖 AI Model Training & Anomaly Detection")
    if "df" not in st.session_state or st.session_state.df.empty:
        st.warning("Please complete Data Input (Step 2) first.")
        if st.button("⬅️ Return to Data Input"):
            st.session_state.step = 2
            st.rerun()
        st.stop()

    dataframe = st.session_state.df.copy()
    contamination_rate = st.slider(
        "Expected Anomaly Sensitivity Rate (%):",
        min_value=1,
        max_value=20,
        value=5,
        help="This is the estimated proportion of unusual records the model should flag. It is not a confirmed fraud rate.",
    ) / 100.0

    features = prepare_numeric_features(dataframe)
    if len(dataframe) < 10:
        st.error("At least 10 payroll records are required for analysis.")
    elif features.shape[1] == 0:
        st.error(
            "No suitable varying numeric columns were found. "
            "Ensure the CSV includes numeric payroll fields such as BasicSalary, "
            "Allowances, Deductions, or NetSalary."
        )
    else:
        if len(features) != len(dataframe):
            st.error("Feature preparation changed the row count. Please check the uploaded data.")
        else:
            try:
                # Split records for a training/test workflow.
                X_train, X_test = train_test_split(
                    features, test_size=0.20, random_state=42
                )
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                model = IsolationForest(
                    n_estimators=200,
                    contamination=contamination_rate,
                    random_state=42,
                )
                model.fit(X_train_scaled)

                # Score all records using the scaler fitted only on training data.
                all_scaled = scaler.transform(features)
                predictions = model.predict(all_scaled)
                anomaly_scores = model.decision_function(all_scaled)

                results = dataframe.copy()
                results["AnomalyStatus"] = np.where(
                    predictions == -1,
                    "Flagged for Review",
                    "No Anomaly Flag",
                )
                results["AnomalyScore"] = anomaly_scores
                flagged_records = results[
                    results["AnomalyStatus"] == "Flagged for Review"
                ].copy()

                st.session_state.results = results
                st.session_state.flagged_records = flagged_records
                st.session_state.model_features = list(features.columns)
                st.session_state.model_trained_at = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                st.success(
                    f"Model training completed using {len(X_train):,} training "
                    f"records and {len(X_test):,} test records."
                )
                st.warning(
                    "A flagged record is an unusual pattern for human review; "
                    "it is not proof of fraud or wrongdoing."
                )

                metric1, metric2, metric3, metric4 = st.columns(4)
                metric1.metric("Records Analysed", f"{len(results):,}")
                metric2.metric("Flagged for Review", f"{len(flagged_records):,}")
                metric3.metric("Not Flagged", f"{len(results) - len(flagged_records):,}")
                metric4.metric(
                    "Flagged Percentage",
                    f"{len(flagged_records) / len(results) * 100:.2f}%",
                )

                st.subheader("🚩 Flagged Payroll Records")
                if flagged_records.empty:
                    st.info("No records were flagged with the selected sensitivity setting.")
                else:
                    st.dataframe(flagged_records, use_container_width=True)

                st.subheader("Model Features Used")
                st.write(", ".join(features.columns.astype(str)))
            except (ValueError, TypeError) as error:
                st.error(
                    "The model could not analyse this dataset. Check that payroll "
                    f"columns contain usable numeric values. Details: {error}"
                )

    col_back, col_continue = st.columns(2)
    with col_back:
        if st.button("⬅️ Back to Data Input", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col_continue:
        if st.button("Continue to Visualizations & Reports ➡️", use_container_width=True):
            if "results" not in st.session_state:
                st.warning("Run a successful AI analysis before continuing.")
            else:
                st.session_state.step = 4
                st.rerun()

# ============================================================
# PAGE 4: VISUALIZATIONS & REPORT EXPORT
# ============================================================
elif st.session_state.step == 4:
    if not st.session_state.auth_verified:
        st.error("⛔ Access locked. Complete authorization on Step 1.")
        st.stop()

    st.header("📊 Payroll Analytics & Report Export")
    if "results" not in st.session_state:
        st.warning("Please run AI Model Analysis on Step 3 first.")
        if st.button("⬅️ Return to AI Analysis"):
            st.session_state.step = 3
            st.rerun()
    else:
        results = st.session_state.results
        flagged_records = st.session_state.flagged_records

        chart1, chart2, chart3 = st.columns(3)

        with chart1:
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            counts = results["AnomalyStatus"].value_counts().reindex(
                ["No Anomaly Flag", "Flagged for Review"], fill_value=0
            )
            bars = ax1.bar(
                counts.index,
                counts.values,
                color=["#008751", "#d32f2f"],
            )
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax1.set_title("Anomaly Status Count", fontweight="bold", fontsize=10)
            ax1.tick_params(axis="x", rotation=15)
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        with chart2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            pie_counts = results["AnomalyStatus"].value_counts().reindex(
                ["No Anomaly Flag", "Flagged for Review"], fill_value=0
            )
            ax2.pie(
                pie_counts.values,
                labels=pie_counts.index,
                colors=["#008751", "#d32f2f"],
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
            ax2.set_title("Anomaly Percentage Share", fontweight="bold", fontsize=10)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with chart3:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            if "State" in results.columns and "BasicSalary" in results.columns:
                state_salary = results.copy()
                state_salary["BasicSalary"] = pd.to_numeric(
                    state_salary["BasicSalary"], errors="coerce"
                )
                state_average = state_salary.groupby("State")["BasicSalary"].mean().dropna()
                if not state_average.empty:
                    bars3 = ax3.bar(state_average.index.astype(str), state_average.values,
                                    color="#1b5e20")
                    for bar in bars3:
                        height = bar.get_height()
                        ax3.text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f"₦{height:,.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                    ax3.tick_params(axis="x", rotation=20)
                    ax3.set_title("Average Basic Salary by State", fontweight="bold", fontsize=10)
                else:
                    ax3.text(0.5, 0.5, "No usable salary values", ha="center", va="center")
                    ax3.set_title("Average Basic Salary by State", fontweight="bold", fontsize=10)
            else:
                ax3.text(0.5, 0.5, "State or BasicSalary column unavailable",
                         ha="center", va="center", wrap=True)
                ax3.set_title("Salary by State", fontweight="bold", fontsize=10)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        st.divider()
        st.subheader("📦 Download Audit Package")
        summary = pd.DataFrame(
            {
                "ReportItem": [
                    "System",
                    "DateTime",
                    "Investigator",
                    "InvestigatorAddress",
                    "Contact",
                    "TotalRecords",
                    "FlaggedRecords",
                    "Model",
                    "FeaturesUsed",
                ],
                "Value": [
                    "IBS AI Payroll Anomaly Detection & Forensic Audit Portal",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    st.session_state.full_name,
                    st.session_state.address,
                    st.session_state.contact,
                    len(results),
                    len(flagged_records),
                    "Isolation Forest",
                    ", ".join(st.session_state.get("model_features", [])),
                ],
            }
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("payroll_audit_summary.csv", summary.to_csv(index=False))
            archive.writestr("full_results.csv", results.to_csv(index=False))
            archive.writestr("flagged_records.csv", flagged_records.to_csv(index=False))

        st.download_button(
            label="📥 Download Audit ZIP Package",
            data=zip_buffer.getvalue(),
            file_name=f"IBS_Payroll_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption(
            "Keep audit exports secure. They may contain personal or confidential payroll data."
        )

    col_back, col_continue = st.columns(2)
    with col_back:
        if st.button("⬅️ Back to AI Analysis", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col_continue:
        if st.button("Continue to Q&A & Kudos ➡️", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

# ============================================================
# PAGE 5: Q&A ASSISTANT & KUDOS
# ============================================================
elif st.session_state.step == 5:
    if not st.session_state.auth_verified:
        st.error("⛔ Access locked. Complete authorization on Step 1.")
        st.stop()

    st.header("❓ Project Q&A Assistant & User Kudos")
    question = st.text_input("Ask a question about the system:")

    if question:
        question_lower = question.lower()
        if "algorithm" in question_lower or "isolation" in question_lower:
            answer = (
                "The system uses Isolation Forest, an unsupervised machine-learning "
                "algorithm that identifies records that differ from common patterns."
            )
        elif "training" in question_lower or "test" in question_lower:
            answer = (
                "The data is split into training and test portions. The model is fitted "
                "on the training portion and then scores the payroll records for unusual patterns."
            )
        elif "report" in question_lower or "download" in question_lower:
            answer = (
                "The audit ZIP package includes a summary CSV, all analysed records, "
                "and a separate CSV containing records flagged for review."
            )
        elif "fraud" in question_lower or "anomaly" in question_lower:
            answer = (
                "An anomaly is a record that looks unusual compared with the patterns "
                "learned by the model. A flag is only a prompt for authorised human "
                "review; it does not establish fraud."
            )
        elif "data" in question_lower or "csv" in question_lower or "upload" in question_lower:
            answer = (
                "Choose the sample dataset or upload a CSV on Step 2. For useful payroll "
                "analysis, include numeric columns such as BasicSalary, Allowances, "
                "Deductions, and NetSalary."
            )
        else:
            answer = (
                "I can explain the Isolation Forest algorithm, data preparation, model "
                "training, anomaly flags, CSV upload, charts, and report downloads."
            )
        st.info("Assistant Response: " + answer)

    st.divider()
    st.subheader("🌟 Leave Your Kudos")
    with st.form("kudos_form"):
        kudos_name = st.text_input(
            "Your Name or Department:",
            value=st.session_state.full_name,
        )
        kudos_rating = st.select_slider(
            "Rate System Performance:",
            options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
            value="Excellent",
        )
        kudos_comment = st.text_area(
            "Your Kudos or Comments:",
            value="Great application!",
        )
        kudos_submitted = st.form_submit_button("Submit Kudos")
        if kudos_submitted:
            st.success(
                f"Thank you, {kudos_name or 'user'}! Your feedback was received "
                f"for this session. Rating: {kudos_rating}."
            )
            st.caption(
                "Note: feedback is displayed as a confirmation but is not saved to a permanent database."
            )

    col_back, col_continue = st.columns(2)
    with col_back:
        if st.button("⬅️ Back to Visualizations", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col_continue:
        if st.button("Continue to Developers ➡️", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

# ============================================================
# PAGE 6: DEVELOPERS TEAM SLIDESHOW
# ============================================================
elif st.session_state.step == 6:
    if not st.session_state.auth_verified:
        st.error("⛔ Access locked. Complete authorization on Step 1.")
        st.stop()

    st.header("👥 Meet the Development Team (12 Members)")
    st.write("Use the navigation buttons to browse the project team.")

    developers = [
        {
            "name": "A. O. Juwon",
            "address": "Ejuku, Yagba East LGA",
            "state": "Kogi State",
            "education": "NCE, B.Sc., M.Sc. (In View)",
            "role": "Team Lead and Project Problem Analyzer",
            "marital_status": "Married",
            "image": "developer 1.jpeg",
        },
        {
            "name": "ADEYEMI MARY MARVELOUS",
            "address": "ODO-ERE, YAGBA WEST LG",
            "state": "KOGI STATE",
            "education": "B. Sc",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "MARRIED",
            "image": "developer 2.jpeg",
        },
        {
            "name": "IBITOYE MAYOMIKUN JONATHAN",
            "address": "ISANLU, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B. ENG. ILORIN",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "Single",
            "image": "developer 3.jpeg",
        },
        {
            "name": "OLUKAIYE OLUGBENGA TIMOTHY",
            "address": "BOLORUNDURO KABBA, KABBA BUNU LG",
            "state": "KOGI STATE",
            "education": "B. ENG, CUSTECH Osara",
            "role": "PROBLEM SOLVING TEAM",
            "marital_status": "Single",
            "image": "developer 4.jpeg",
        },
        {
            "name": "KOMOLAFE BABATUNDE RICHARD",
            "address": "AYETORO GBEDE IJUMU LG",
            "state": "KOGI STATE",
            "education": "B. Sc.",
            "role": "DEMONSTRATION TEAM",
            "marital_status": "Single",
            "image": "developer 5.jpeg",
        },
        {
            "name": "ADEWUMI BISIRIYU TUNDE",
            "address": "IFE_OLUKOTUN, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B. Sc, CNA, NBA (in view)",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "MARRIED",
            "image": "developer 6.jpeg",
        },
        {
            "name": "BABALOLA MARVELOUS ITUNUOLUWA",
            "address": "EGBE, YAGBA WEST",
            "state": "KOGI STATE",
            "education": "B. Sc (NURSING) FED.UNI LOKOJA",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "Single",
            "image": "developer 7.jpeg",
        },
        {
            "name": "AKANKU TAIYE OLUWAPEMISIRE",
            "address": "ITEDO ISANLU, YAGBA EAST LG",
            "state": "KOGI STATE",
            "education": "B Sc. (Econs)",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "MARRIED",
            "image": "developer 8.jpeg",
        },
        {
            "name": "VICTOR IWALAYE",
            "address": "IYARA, IJUMU LG",
            "state": "KOGI STATE",
            "education": "NATIONAL DIPLOMA",
            "role": "AI TEAM",
            "marital_status": "Single",
            "image": "developer 9.jpeg",
        },
        {
            "name": "RACHEAL OLORUNMAIYE",
            "address": "KABBA BUNNU, LG",
            "state": "KOGI STATE",
            "education": "B.Sc",
            "role": "DATA COLLECTION TEAM",
            "marital_status": "N/A",
            "image": "developer 10.jpeg",
        },
        {
            "name": "ISREAL ADEOLA OLUMAIYEGUN",
            "address": "ODO ERE, YAGBA WEST LG",
            "state": "KOGI STATE",
            "education": "B. Sc. (Econs) KSU, Kabba",
            "role": "DEMONSTRATION TEAM MEMBER",
            "marital_status": "Single",
            "image": "developer 11.jpeg",
        },
        {
            "name": "Team Member 12",
            "address": "Address 12",
            "state": "State 12",
            "education": "Education 12",
            "role": "Role 12",
            "marital_status": "Single",
            "image": "developer12.jpeg",
        },
    ]

    # Keep the developer index inside the valid range.
    st.session_state.dev_index = int(st.session_state.dev_index) % len(developers)
    previous_col, counter_col, next_col = st.columns([1, 1, 1])

    with previous_col:
        if st.button("⬅️ Previous Developer", use_container_width=True):
            st.session_state.dev_index = (
                st.session_state.dev_index - 1
            ) % len(developers)
            st.rerun()

    with counter_col:
        st.markdown(
            f"<div style='text-align:center;padding-top:10px;font-weight:700;'>"
            f"Developer {st.session_state.dev_index + 1} of {len(developers)}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with next_col:
        if st.button("Next Developer ➡️", use_container_width=True):
            st.session_state.dev_index = (
                st.session_state.dev_index + 1
            ) % len(developers)
            st.rerun()

    current_developer = developers[st.session_state.dev_index]
    st.divider()
    image_column, details_column = st.columns([1, 2])

    with image_column:
        base_image = current_developer["image"]
        alternatives = [base_image]
        if base_image.lower().endswith(".jpeg"):
            alternatives.append(base_image[:-5] + ".jpg")
            alternatives.append(base_image[:-5] + ".png")
        elif base_image.lower().endswith(".jpg"):
            alternatives.append(base_image[:-4] + ".jpeg")
            alternatives.append(base_image[:-4] + ".png")
        elif base_image.lower().endswith(".png"):
            alternatives.extend([base_image[:-4] + ".jpeg", base_image[:-4] + ".jpg"])

        image_path = find_existing_image(alternatives)
        if image_path:
            st.image(image_path, use_container_width=True)
        else:
            st.info(
                f"Photo not found: {base_image}. Add the picture to the same "
                "folder as app.py, using the matching filename."
            )

    with details_column:
        st.subheader(current_developer["name"])
        st.markdown(f"📍 **Address:** {current_developer['address']}")
        st.markdown(f"🏛️ **State:** {current_developer['state']}")
        st.markdown(f"🎓 **Education:** {current_developer['education']}")
        st.markdown(f"💼 **Role:** {current_developer['role']}")
        st.markdown(f"💍 **Marital Status:** {current_developer['marital_status']}")

    if st.button("⬅️ Back to Q&A Assistant & Kudos", use_container_width=True):
        st.session_state.step = 5
        st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "International Business School (IBS) • AI Payroll Anomaly Detection & Forensic Audit Portal"
)
st.caption(
    "Demonstration software only. Anomaly flags require authorised human verification "
    "and must not be treated as proof of fraud."
)
