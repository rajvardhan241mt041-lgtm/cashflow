import streamlit as st
import pandas as pd
import requests
import os
import joblib

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Load encoders if available to populate UI selectboxes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "encoders.joblib")

encoders = {}
if os.path.exists(ENCODERS_PATH):
    try:
        encoders = joblib.load(ENCODERS_PATH)
    except Exception as e:
        pass

def get_choices(col_name: str, default_choices: list) -> list:
    if col_name in encoders:
        return sorted(list(encoders[col_name].classes_))
    return default_choices

st.set_page_config(page_title="Cash Flow Forecasting Dashboard", layout="wide")

# Modern aesthetic styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
    .stButton>button:hover {
        background-color: #17b978;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💸 Intelligent Cash Flow Forecasting & Recommendation Factory")
st.markdown("---")

# Header KPIs and Description
st.markdown("""
    Welcome to the **Cash Flow Forecasting Control Center**. Expose raw invoice data to predict overdue delays 
    using real-time XGBoost ML inference, explain key predictions with SHAP values, and receive automated 
    Next-Best-Action recovery strategies.
""")

with st.form("predict_form"):
    st.markdown("### 📝 Enter Raw Invoice & Customer Profile Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### **Invoice Core Details**")
        amount = st.number_input("Invoice Amount ($)", min_value=0.0, value=15000.0, step=100.0)
        doc_date = st.date_input("Document Date (Doc_Date)", value=pd.to_datetime("2026-05-24"))
        net_due_date = st.date_input("Net Due Date (Net_Due_Date)", value=pd.to_datetime("2026-06-24"))
        payment_term = st.number_input("Payment Term (Days)", min_value=0.0, value=30.0, step=1.0)
        document_no = st.number_input("Document Number (Document_No)", min_value=0.0, value=91225037841.0, format="%.0f")
        payment_method = st.selectbox("Payment Method Description", get_choices('Payment_Method_description', ['No Payment Method', 'Direct Debits 1']))
        zipcode = st.selectbox("Zipcode", get_choices('Zipcode', ['AX0012', 'AX0013']))

    with col2:
        st.markdown("##### **Customer Profile & History**")
        cust_num = st.number_input("Customer Number (Cust_Num)", min_value=0.0, value=5039221094.0, format="%.0f")
        customer_name = st.selectbox("Customer Name (Code)", get_choices('Customer_Name', ['AA00', 'AA100']))
        age_of_customer_months = st.number_input("Age of Customer Relationship (Months)", min_value=0.0, value=24.0, step=1.0)
        no_of_orders = st.number_input("Total Number of Orders (Historical)", min_value=0.0, value=10.0, step=1.0)
        rank_of_order = st.number_input("Rank of Order (This Invoice)", min_value=0.0, value=5.0, step=1.0)
        region = st.selectbox("Region", get_choices('Region', ['AA111', 'AA112']))
        city = st.selectbox("City", get_choices('City', ['AA22', 'AA23']))
        
    submit_btn = st.form_submit_button("🔮 Predict Overdue Delay & Get Recommendation", use_container_width=True)

if submit_btn:
    payload = {
        "Cust_Num": cust_num,
        "Document_No": document_no,
        "Amount": amount,
        "Age_Of_Customer_Months": age_of_customer_months,
        "No_of_orders_by_customer": no_of_orders,
        "Rank_of_order_by_customer": rank_of_order,
        "Doc_Date": doc_date.strftime("%Y-%m-%d"),
        "Net_Due_Date": net_due_date.strftime("%Y-%m-%d"),
        "Payment_Method_description": payment_method,
        "Region": region,
        "City": city,
        "Customer_Name": customer_name,
        "Zipcode": zipcode,
        "Payment_Term": payment_term
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            st.success("Inference generated successfully!")
            data = response.json()
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("#### **Predictions & Actions**")
                predicted_val = data["Predicted_Delay"]
                st.metric(label="Predicted Overdue Delay", value=f"{predicted_val:.2f} days")
                
                action = data["Next_Best_Action"]
                if "ESCALATION" in action or "URGENT" in action:
                    st.error(f"🚨 **Next Best Action:** {action}")
                elif "FOLLOW-UP" in action or "CALL" in action:
                    st.warning(f"⚠️ **Next Best Action:** {action}")
                else:
                    st.success(f"✅ **Next Best Action:** {action}")
            
            with res_col2:
                st.markdown("#### **SHAP Driver Breakdown**")
                for i, reason in enumerate(data.get("Top_SHAP_Reasons", []), 1):
                    if "error" in reason:
                        st.write("Explainability driver breakdown unavailable.")
                    else:
                        impact_val = reason['impact']
                        direction = "increased" if impact_val > 0 else "decreased"
                        st.write(f"{i}. **{reason['feature']}** {direction} the predicted delay by **{abs(impact_val):.2f} days**")
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to predictive API: {e}")

