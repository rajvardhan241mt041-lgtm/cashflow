def get_recommendation(predicted_delay: float, amount: float, age_of_customer_months: float) -> str:
    """
    Business rules engine for Cash Flow Forecasting Next-Best-Action.
    
    Rule 1: If Predicted_Delay > 30 days AND Amount > 50000 -> "FINANCE ESCALATION: High-value severe delay."
    Rule 2: If Predicted_Delay > 30 days (but lower amount) -> "MANAGER ESCALATION: Severe delay."
    Rule 3: If Predicted_Delay is between 15 and 30 days -> "CALL FOLLOW-UP: Moderate risk."
    Rule 4: If Predicted_Delay < 15 days -> "WAIT & OBSERVE / AUTOMATED EMAIL."
    """
    if predicted_delay > 30:
        if amount > 50000:
            return "FINANCE ESCALATION: High-value severe delay."
        else:
            return "MANAGER ESCALATION: Severe delay."
    elif 15 <= predicted_delay <= 30:
        return "CALL FOLLOW-UP: Moderate risk."
    else:
        return "WAIT & OBSERVE / AUTOMATED EMAIL."
