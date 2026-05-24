"""
rules_engine.py — Business Rules Engine for Next-Best-Action Recommendations
==============================================================================
SAD V1 compliant. Maps (predicted_delay, invoice_amount) to a concrete
collections action recommendation.

Threshold Matrix (SAD V1 spec):
    ┌─────────────────────────────────────────────┬──────────────────────────┐
    │ Condition                                   │ Recommendation           │
    ├─────────────────────────────────────────────┼──────────────────────────┤
    │ Delay > 30 days  AND  Amount > $50,000      │ FINANCE ESCALATION       │
    │ Delay > 30 days  (any amount)               │ MANAGER ESCALATION       │
    │ 15 days <= Delay <= 30 days                 │ CALL FOLLOW-UP           │
    │ Delay < 15 days                             │ WAIT & OBSERVE           │
    └─────────────────────────────────────────────┴──────────────────────────┘

Author: MLOps Factory
PEP-8 compliant.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants — single source of truth, easy to update
# ---------------------------------------------------------------------------
HIGH_DELAY_THRESHOLD_DAYS: float = 30.0    # Days above which escalation is triggered
MEDIUM_DELAY_THRESHOLD_DAYS: float = 15.0  # Days above which a call is required
HIGH_VALUE_THRESHOLD_AMOUNT: float = 50_000.0  # Amount above which finance is involved

# Human-readable action labels
ACTION_FINANCE_ESCALATION: str = "FINANCE ESCALATION"
ACTION_MANAGER_ESCALATION: str = "MANAGER ESCALATION"
ACTION_CALL_FOLLOW_UP: str = "CALL FOLLOW-UP"
ACTION_WAIT_OBSERVE: str = "WAIT & OBSERVE"


# ===========================================================================
# DATA CLASS — Structured recommendation output
# ===========================================================================

@dataclass
class Recommendation:
    """
    Structured output from the rules engine.

    Attributes:
        action: The recommended collections action string.
        priority: Numeric priority (1=highest, 4=lowest) for queue ordering.
        reason: Human-readable explanation of why this action was assigned.
    """
    action: str
    priority: int
    reason: str


# ===========================================================================
# CORE RULES FUNCTION
# ===========================================================================

def get_recommendation(
    predicted_delay: float,
    amount: float,
    age_of_customer_months: Optional[float] = None,
) -> Recommendation:
    """
    Apply the SAD V1 business rules to generate a Next-Best-Action
    recommendation for a collections agent.

    Args:
        predicted_delay: Days overdue predicted by the XGBoost model.
                         Negative values mean the invoice is expected to be
                         paid early (no action needed).
        amount:          Invoice amount in currency units (e.g., USD).
        age_of_customer_months: Optional customer tenure. Not used in SAD V1
                                threshold logic but retained for future rules.

    Returns:
        Recommendation: Dataclass with action, priority, and reason fields.

    Examples:
        >>> r = get_recommendation(45, 75000)
        >>> r.action
        'FINANCE ESCALATION'

        >>> r = get_recommendation(35, 20000)
        >>> r.action
        'MANAGER ESCALATION'

        >>> r = get_recommendation(20, 5000)
        >>> r.action
        'CALL FOLLOW-UP'

        >>> r = get_recommendation(5, 1000)
        >>> r.action
        'WAIT & OBSERVE'
    """
    # Guard: treat negative delays (early payment) as zero-risk
    effective_delay = max(predicted_delay, 0.0)

    # ------------------------------------------------------------------
    # Rule 1: HIGH delay + HIGH value → Finance team must be involved
    # ------------------------------------------------------------------
    if (
        effective_delay > HIGH_DELAY_THRESHOLD_DAYS
        and amount > HIGH_VALUE_THRESHOLD_AMOUNT
    ):
        return Recommendation(
            action=ACTION_FINANCE_ESCALATION,
            priority=1,
            reason=(
                f"Invoice predicted {effective_delay:.1f} days overdue "
                f"on a high-value invoice of ${amount:,.0f} — "
                "requires immediate finance team escalation."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 2: HIGH delay (any value) → Manager must intervene
    # ------------------------------------------------------------------
    if effective_delay > HIGH_DELAY_THRESHOLD_DAYS:
        return Recommendation(
            action=ACTION_MANAGER_ESCALATION,
            priority=2,
            reason=(
                f"Invoice predicted {effective_delay:.1f} days overdue "
                "— manager intervention required to recover payment."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 3: MEDIUM delay → Collector call required
    # ------------------------------------------------------------------
    if effective_delay >= MEDIUM_DELAY_THRESHOLD_DAYS:
        return Recommendation(
            action=ACTION_CALL_FOLLOW_UP,
            priority=3,
            reason=(
                f"Invoice predicted {effective_delay:.1f} days overdue "
                "— proactive call follow-up recommended."
            ),
        )

    # ------------------------------------------------------------------
    # Rule 4: LOW delay (or early) → No action needed
    # ------------------------------------------------------------------
    return Recommendation(
        action=ACTION_WAIT_OBSERVE,
        priority=4,
        reason=(
            f"Invoice predicted {effective_delay:.1f} days overdue "
            "— within acceptable range. Monitor and observe."
        ),
    )


def get_recommendation_str(predicted_delay: float, amount: float) -> str:
    """
    Convenience wrapper returning just the action string.
    Maintained for backward compatibility with older code paths.

    Args:
        predicted_delay: Days overdue predicted by the model.
        amount:          Invoice amount.

    Returns:
        str: Action string (e.g., 'FINANCE ESCALATION').
    """
    return get_recommendation(predicted_delay, amount).action
