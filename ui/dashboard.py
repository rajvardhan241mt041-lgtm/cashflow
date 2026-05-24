import streamlit as st
import pandas as pd
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Cash Flow Forecasting Dashboard", layout="wide")

st.title("💸 Cash Flow Forecasting & Recommendation Engine")
st.markdown("Automated predictions and Next-Best-Action recommendations based on the Accounts Receivable dataset.")

# For demonstration, we simulate loading some new invoices that need predicting.
# In a real environment, you might fetch these from a database or an API endpoint.
st.subheader("Invoice Prediction Panel")

with st.form("predict_form"):
    st.markdown("### Test Invoice Prediction")
    amount = st.number_input("Amount", min_value=0.0, value=15000.0)
    age_of_customer_months = st.number_input("Age of Customer (Months)", min_value=0.0, value=24.0)
    payment_term = st.number_input("Payment Term (Days)", min_value=0.0, value=30.0)
    no_of_orders = st.number_input("Number of Orders", min_value=0.0, value=10.0)
    rank_of_order = st.number_input("Rank of Order", min_value=0.0, value=5.0)
    
    submit_btn = st.form_submit_button("Get Prediction & Action")

if submit_btn:
    payload = {
        "Amount": amount,
        "Age_Of_Customer_Months": age_of_customer_months,
        "Payment_Term": payment_term,
        "No_of_orders_by_customer": no_of_orders,
        "Rank_of_order_by_customer": rank_of_order,
        "features": {
            "Amount": amount,
            "Age_Of_Customer_Months": age_of_customer_months,
            "Payment_Term": payment_term,
            "No_of_orders_by_customer": no_of_orders,
            "Rank_of_order_by_customer": rank_of_order
        }
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="Predicted Overdue Delay (Days)", value=round(data["Predicted_Delay"], 2))
                
                action = data["Next_Best_Action"]
                if "ESCALATION" in action:
                    st.error(f"🚨 **Action:** {action}")
                elif "FOLLOW-UP" in action:
                    st.warning(f"⚠️ **Action:** {action}")
                else:
                    st.success(f"✅ **Action:** {action}")
            
            with col2:
                st.markdown("#### Top Drivers (SHAP Values)")
                reasons = data["Top_SHAP_Reasons"]
                for i, r in enumerate(reasons):
                    if "error" in r:
                        st.error(f"SHAP Explainer Error: {r['error']}")
                    else:
                        st.write(f"{i+1}. **{r['feature']}**: {round(r['impact'], 2)} days")
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {e}")
