"""
Intent taxonomy for the AppleSupport agent.

Derived from manual reading of a random sample of 60 real
(customer message, Apple reply) pairs from the dataset — see
data/taxonomy_reading_sample.csv for the underlying reading notes.
"""

INTENT_TAXONOMY = {
    "update_installation": "Update won't install, update errors, new bugs appearing right after an update",
    "battery_performance": "Battery draining fast, general performance/slowness complaints",
    "bug_report": "A specific feature is broken (e.g. Siri, contacts, settings not saving), not tied to a recent update",
    "account_security": "Login issues, hacking concerns, two-factor authentication",
    "icloud_sync": "iCloud or cross-device data/sync issues",
    "feature_howto": "Customer is asking how to do something, not reporting a problem",
    "hardware_issue": "Physical device or accessory problem (battery service, charger, cables)",
    "support_escalation": "Customer has already been in contact before and remains unresolved/frustrated",
    "other_unclear": "Doesn't clearly fit another category",
}

def get_intent_labels() -> list[str]:
    """Returns just the label names, e.g. for use in a classifier prompt."""
    return list(INTENT_TAXONOMY.keys())