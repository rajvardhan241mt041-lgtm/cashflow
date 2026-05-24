"""
dashboard.py — Streamlit Collector Dashboard with SHAP Explainability
======================================================================
SAD V1 compliant. Features:
    - Full raw-feature invoice scoring form → POST /score-invoice
    - Prioritized invoice batch dataframe (mock batch for demonstration)
    - SHAP waterfall/bar chart — collectors see the "why" behind each prediction
    - Sidebar model info panel (version, MLflow run ID, source)
    - Colour-coded Next-Best-Action badges

Author: MLOps Factory
PEP-8 compliant.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AR Collections Intelligence — Cash Flow Factory",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-ish theme with accent colours
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Font import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3d59 0%, #17375e 100%);
        border-radius: 12px;
        padding: 16px;
        color: white !important;
    }
    [data-testid="metric-container"] label { color: #90caf9 !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color: white !important; }

    /* Divider */
    hr { border: 1px solid #e0e0e0; margin: 24px 0; }

    /* Action badge colours */
    .badge-finance   { background:#c62828; color:white; border-radius:6px; padding:4px 10px; font-weight:700; }
    .badge-manager   { background:#e65100; color:white; border-radius:6px; padding:4px 10px; font-weight:700; }
    .badge-call      { background:#f9a825; color:#333;  border-radius:6px; padding:4px 10px; font-weight:700; }
    .badge-wait      { background:#2e7d32; color:white; border-radius:6px; padding:4px 10px; font-weight:700; }

    /* Submit button */
    .stButton > button {
        background: linear-gradient(90deg, #1e3d59, #17b978);
        color: white;
        border: none;
        border-radius: 8px;
        height: 3rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# HELPERS
# ===========================================================================

def _action_badge(action: str) -> str:
    """Return an HTML-coloured badge for the action string."""
    css_map = {
        "FINANCE ESCALATION": "badge-finance",
        "MANAGER ESCALATION": "badge-manager",
        "CALL FOLLOW-UP":     "badge-call",
        "WAIT & OBSERVE":     "badge-wait",
    }
    css_class = css_map.get(action, "badge-wait")
    return f'<span class="{css_class}">{action}</span>'


def _call_score_invoice(payload: dict) -> dict:
    """POST the invoice payload to /score-invoice and return the JSON response."""
    url = f"{API_URL}/score-invoice"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_model_info() -> dict:
    """GET /model-info from the API."""
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _build_demo_batch(api_url: str) -> pd.DataFrame:
    """
    Score a small set of demonstration invoices in batch and return a
    prioritized dataframe. Cached for 5 minutes to avoid repeated API calls.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    demo_invoices = [
        {"Document_No": 91225037841.0, "Amount": 82000,  "Payment_Term": 5,  "Doc_Date": today, "Net_Due_Date": (datetime.utcnow() + timedelta(days=5)).strftime("%Y-%m-%d"),  "Age_Of_Customer_Months": 24,  "No_of_orders_by_customer": 10, "Rank_of_order_by_customer": 5,  "Payment_Method_description": "No Payment Method", "Region": "AA111", "City": "AA22", "Customer_Name": "AA00", "Zipcode": "AX0012", "Cust_Num": 5039221094},
        {"Document_No": 91225037842.0, "Amount": 12000,  "Payment_Term": 30, "Doc_Date": today, "Net_Due_Date": (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d"), "Age_Of_Customer_Months": 120, "No_of_orders_by_customer": 50, "Rank_of_order_by_customer": 45, "Payment_Method_description": "Direct Debits 1",  "Region": "AA112", "City": "AA23", "Customer_Name": "AA100","Zipcode": "AX0013", "Cust_Num": 5039221095},
        {"Document_No": 91225037843.0, "Amount": 210000, "Payment_Term": 60, "Doc_Date": today, "Net_Due_Date": (datetime.utcnow() + timedelta(days=60)).strftime("%Y-%m-%d"), "Age_Of_Customer_Months": 6,   "No_of_orders_by_customer": 2,  "Rank_of_order_by_customer": 2,  "Payment_Method_description": "No Payment Method", "Region": "AA113", "City": "AA24", "Customer_Name": "AA101","Zipcode": "AX0014", "Cust_Num": 5039221096},
        {"Document_No": 91225037844.0, "Amount": 3400,   "Payment_Term": 15, "Doc_Date": today, "Net_Due_Date": (datetime.utcnow() + timedelta(days=15)).strftime("%Y-%m-%d"), "Age_Of_Customer_Months": 60,  "No_of_orders_by_customer": 30, "Rank_of_order_by_customer": 25, "Payment_Method_description": "Direct Debits 2",  "Region": "AA114", "City": "AA25", "Customer_Name": "AA102","Zipcode": "AX0015", "Cust_Num": 5039221097},
        {"Document_No": 91225037845.0, "Amount": 67000,  "Payment_Term": 90, "Doc_Date": today, "Net_Due_Date": (datetime.utcnow() + timedelta(days=90)).strftime("%Y-%m-%d"), "Age_Of_Customer_Months": 36,  "No_of_orders_by_customer": 20, "Rank_of_order_by_customer": 18, "Payment_Method_description": "No Payment Method", "Region": "AA115", "City": "AA26", "Customer_Name": "AA103","Zipcode": "AX0016", "Cust_Num": 5039221098},
    ]

    rows = []
    for inv in demo_invoices:
        try:
            result = requests.post(f"{api_url}/score-invoice", json=inv, timeout=20).json()
            rows.append({
                "Priority 🔺":      result.get("Priority", 4),
                "Invoice #":        int(inv["Document_No"]),
                "Amount ($)":       f"${inv['Amount']:,.0f}",
                "Predicted Delay":  f"{result.get('Predicted_Delay', 0):.1f} days",
                "Expected Payment": result.get("Expected_Payment_Date", "N/A"),
                "Action":           result.get("Next_Best_Action", "N/A"),
            })
        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Priority 🔺")
    return df


# ===========================================================================
# SIDEBAR — Model info
# ===========================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Model Status")

    info = _get_model_info()
    if info:
        st.success(f"**Source:** {info.get('model_source', 'unknown').upper()}")
        st.info(f"**Version:** {info.get('model_version', 'N/A')}")
        st.caption(f"Run ID: `{info.get('run_id', 'N/A')}`")
        st.caption(f"Loaded at: {info.get('loaded_at', 'N/A')}")
        if info.get("model_source") == "mlflow":
            mlflow_url = f"{info.get('mlflow_tracking_uri', '#')}".replace("mlflow:5000", "localhost:5000")
            st.markdown(f"[Open MLflow UI ↗]({mlflow_url})")
    else:
        st.warning("Could not reach API.")

    st.divider()
    st.caption("SAD V1 — Cash Flow MLOps Factory")


# ===========================================================================
# MAIN PAGE — Header
# ===========================================================================

st.title("💸 AR Collections Intelligence Dashboard")
st.markdown(
    "Powered by **XGBoost + MLflow** · Explained by **SHAP** · "
    "Orchestrated by **Prefect** · Stored in **PostgreSQL**"
)
st.divider()

# ===========================================================================
# SECTION 1 — Priority Batch Queue
# ===========================================================================

st.subheader("📋 Prioritized Collections Queue")
st.caption(
    "Batch-scored invoices ranked by predicted overdue risk. "
    "Collectors should work top-to-bottom."
)

with st.spinner("Scoring demonstration invoices via API..."):
    batch_df = _build_demo_batch(API_URL)

if not batch_df.empty:
    # Colour-code the Action column
    def colour_action(val: str) -> str:
        colours = {
            "FINANCE ESCALATION": "background-color:#ffcdd2; color:#b71c1c; font-weight:700",
            "MANAGER ESCALATION": "background-color:#ffe0b2; color:#e65100; font-weight:700",
            "CALL FOLLOW-UP":     "background-color:#fff9c4; color:#f57f17; font-weight:700",
            "WAIT & OBSERVE":     "background-color:#c8e6c9; color:#1b5e20; font-weight:700",
        }
        return colours.get(val, "")

    styled = batch_df.style.applymap(colour_action, subset=["Action"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.warning(
        "Could not load batch scores. "
        "Ensure the API is running (`http://localhost:8000/health`)."
    )

st.divider()

# ===========================================================================
# SECTION 2 — Single Invoice Scoring Form
# ===========================================================================

st.subheader("🔍 Score a Single Invoice")

with st.form("score_form"):
    st.markdown("##### 📄 Invoice Details")
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Invoice Amount ($)", min_value=0.0, value=15000.0, step=100.0)
        doc_date = st.date_input("Document Date", value=datetime.utcnow().date())
        net_due_date = st.date_input("Net Due Date", value=(datetime.utcnow() + timedelta(days=30)).date())
        payment_term = st.number_input("Payment Term (days)", min_value=0.0, value=30.0, step=1.0)
        document_no = st.number_input("Document No.", value=91225037841.0, format="%.0f")
        payment_method = st.selectbox(
            "Payment Method",
            ["No Payment Method", "Direct Debits 1", "Direct Debits 2", "Regulatory 1"]
        )
        zipcode = st.text_input("Zipcode", value="AX0012")

    with col2:
        st.markdown("##### 👤 Customer Profile")
        cust_num = st.number_input("Customer Number", value=5039221094.0, format="%.0f")
        customer_name = st.text_input("Customer Name (code)", value="AA00")
        age_months = st.number_input("Customer Age (months)", min_value=0.0, value=24.0, step=1.0)
        no_orders = st.number_input("Total Orders (historical)", min_value=0.0, value=10.0, step=1.0)
        rank_order = st.number_input("Rank of This Invoice", min_value=0.0, value=5.0, step=1.0)
        region = st.text_input("Region", value="AA111")
        city = st.text_input("City", value="AA22")

    submitted = st.form_submit_button(
        "🔮 Score Invoice — Get Prediction & Action",
        use_container_width=True,
    )

if submitted:
    payload = {
        "Cust_Num": cust_num,
        "Document_No": document_no,
        "Amount": amount,
        "Age_Of_Customer_Months": age_months,
        "No_of_orders_by_customer": no_orders,
        "Rank_of_order_by_customer": rank_order,
        "Doc_Date": doc_date.strftime("%Y-%m-%d"),
        "Net_Due_Date": net_due_date.strftime("%Y-%m-%d"),
        "Payment_Method_description": payment_method,
        "Region": region,
        "City": city,
        "Customer_Name": customer_name,
        "Zipcode": zipcode,
        "Payment_Term": payment_term,
    }

    with st.spinner("Running inference via FastAPI..."):
        try:
            result = _call_score_invoice(payload)

            # ----------------------------------------------------------
            # KPI row
            # ----------------------------------------------------------
            st.success("✅ Inference successful!")
            k1, k2, k3 = st.columns(3)
            k1.metric("Predicted Overdue Delay", f"{result['Predicted_Delay']:.2f} days")
            k2.metric("Expected Payment Date", result["Expected_Payment_Date"])
            k3.metric("Priority", f"P{result['Priority']}")

            # ----------------------------------------------------------
            # Action badge
            # ----------------------------------------------------------
            action = result["Next_Best_Action"]
            st.markdown(
                f"**Recommended Action:** {_action_badge(action)}",
                unsafe_allow_html=True
            )
            st.caption(result.get("Reason", ""))

            st.divider()

            # ----------------------------------------------------------
            # SHAP Waterfall / Bar Chart
            # ----------------------------------------------------------
            shap_reasons = result.get("Top_SHAP_Reasons", [])
            if shap_reasons:
                st.subheader("📊 Why This Prediction? (SHAP Explainability)")
                st.caption(
                    "Positive bars = features that **increase** predicted delay. "
                    "Negative bars = features that **decrease** it."
                )

                features = [r["feature"] for r in shap_reasons]
                impacts = [r["impact_days"] for r in shap_reasons]
                colours = [
                    "#ef5350" if v > 0 else "#42a5f5"
                    for v in impacts
                ]

                fig = go.Figure(go.Bar(
                    x=impacts,
                    y=features,
                    orientation="h",
                    marker_color=colours,
                    text=[f"{v:+.2f} days" for v in impacts],
                    textposition="outside",
                ))
                fig.update_layout(
                    xaxis_title="Impact on Predicted Delay (days)",
                    yaxis_title="Feature",
                    height=max(300, len(features) * 80),
                    margin={"l": 10, "r": 10, "t": 20, "b": 20},
                    plot_bgcolor="#f8f9fa",
                    paper_bgcolor="#f8f9fa",
                    font={"family": "Inter, sans-serif"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SHAP explainability was not available for this prediction.")

            # ----------------------------------------------------------
            # Raw JSON expander
            # ----------------------------------------------------------
            with st.expander("Raw API Response (JSON)"):
                st.json(result)

        except requests.exceptions.HTTPError as http_exc:
            st.error(f"API Error {http_exc.response.status_code}: {http_exc.response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach API at `{API_URL}`. Is the API container running?")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
