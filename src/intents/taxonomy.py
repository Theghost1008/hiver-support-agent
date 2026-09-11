"""
Intent taxonomy for the AppleSupport agent.

Derived from manual reading of a random sample of 60 real
(customer message, Apple reply) pairs from the dataset — see
data/taxonomy_reading_sample.csv for the underlying reading notes.

Updated after labeling the full 60-row sample: added `billing_payment`
to cover payment-method / billing-info requests, which didn't fit
cleanly under any existing bucket (e.g. a customer confirming they
want to update their payment info, with Apple linking a "how to
change payment info" guide).
"""

INTENT_TAXONOMY = {
    "update_installation": "Update won't install, update errors, new bugs appearing right after an update",
    "battery_performance": "Battery draining fast, general performance/slowness complaints",
    "bug_report": "A specific feature is broken (e.g. Siri, contacts, settings not saving), not tied to a recent update",
    "account_security": "Login issues, hacking concerns, two-factor authentication, phishing/fraud reports",
    "icloud_sync": "iCloud or cross-device data/sync issues",
    "feature_howto": "Customer is asking how to do something, or how a feature/setting works, not reporting a problem",
    "hardware_issue": "Physical device or accessory problem (battery service, charger, cables, camera, screen)",
    "support_escalation": "Customer has already been in contact before and remains unresolved/frustrated",
    "billing_payment": "Payment method, billing, or subscription/payment-info questions",
    "other_unclear": "Doesn't clearly fit another category, or too little context to tell (vague question, resolved issue, short reply with no standalone context)",
}

def get_intent_labels() -> list[str]:
    return list(INTENT_TAXONOMY.keys())