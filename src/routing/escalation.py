from langdetect import detect, LangDetectException
from src.intents.classifier import classify_intent, load_few_shot_examples
from src.retrieval.retriever import load_index, retrieve_similar
import re

SHORT_MSG_WORD_THRESHOLD = 5
WEAK_GROUNDING_SIMILARITY = 0.55

ALWAYS_ESCALATE_INTENTS = {
    "account_security","billing_payment","other_unclear","support_escalation"
}

SAFETY_KEYWORDS = [
    r'\bfire\b', r'\bsmoke\b', r'\bexplod', r'\bburn(ed|ing|t)?\b',
    r'\binjur(y|ed|ies)\b', r'\bshock(ed)?\b', r'\bcaught fire\b',
]

SEVERE_IMPACT_PHRASES = [
    r'\ball my (photos|data|files|contacts)\b',
    r'\blost everything\b', r'\blost all\b', r'\beverything.{0,15}gone\b',
    r'\bgone forever\b',
]

PROFANITY_PATTERN = re.compile(
    r'\b(fuck|shit|damn|ass|bitch|crap)\w*\b', re.IGNORECASE
)

PRIOR_ATTEMPT_PHRASES = [
    r'\balready tried\b', r'\btried everything\b', r'\bfactory reset\b',
    r'\brestored (it |the phone )?to factory\b', r'\bstill (not|doesn\'?t|isn\'?t|won\'?t)\b',
]


def is_safety_critical(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in SAFETY_KEYWORDS)

def has_severe_impact_language(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in SEVERE_IMPACT_PHRASES)

def has_hostile_tone(text: str) -> bool:
    has_profanity = bool(PROFANITY_PATTERN.search(text))
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    is_shouting = caps_ratio > 0.6 and len(letters) > 15  # avoid false positives on short messages
    return has_profanity or is_shouting

def has_prior_failed_attempt(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in PRIOR_ATTEMPT_PHRASES)

def is_non_eng(text:str)->bool:
    try:
        return detect(text)!="en"
    except LangDetectException:
        return False

def decide_escalation(message: str, intent:str, grounded_examples: list[dict])->dict:
    word_count = len(message.split())
    if is_safety_critical(message):
        return {"decision": "escalate", "reason": "Message contains safety-critical language (e.g. fire, injury) requiring urgent human handling."}

    if has_severe_impact_language(message):
        return {"decision": "escalate", "reason": "Message describes severe impact (e.g. total data loss) warranting careful human review."}

    if has_hostile_tone(message):
        return {"decision": "escalate", "reason": "Message shows signs of hostile or highly frustrated tone (profanity or shouting)."}

    if has_prior_failed_attempt(message):
        return {"decision": "escalate", "reason": "Message indicates a prior troubleshooting attempt already failed, suggesting a deeper issue."}

    if intent in ALWAYS_ESCALATE_INTENTS:
        return {"decision": "escalate", "reason": f"Intent '{intent}' is always escalated (sensitive or already-unresolved category)."}
    if intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "decision": "escalate",
            "reason":f"Intent: '{intent}' is always escalated (sensitive or already-unresovled category)"
        }
    if is_non_eng(message):
        return {
            "decision":"escalate",
            "reason": "Message appears to be Non-English; system is scoped to English support"
        }
    if word_count< SHORT_MSG_WORD_THRESHOLD:
        return {
            "decision":"escalate",
            "reason":f"Message is very short({word_count} words) and likely lacks standalone context(a known failure mode observed during classifier evaluation)."
        }
    if not grounded_examples:
        return {
            "decision":"escalate",
            "reason": "No historical examples cleared the similarity threshold; no strong grounding available for a reply"
        }
    avg_similarity = sum(ex["similarity"] for ex in grounded_examples) / len(grounded_examples)
    if avg_similarity < WEAK_GROUNDING_SIMILARITY:
        return {
            "decision":"escalate",
            "reason":f"Grounding is weak (avg similarity {avg_similarity:.2f}); low confidence in retrieved historical precedent"
        }
    return {
        "decision":"auto_handle",
        "reason":f"Intent '{intent}' is not in the always escalate set, message has sufficient context, and grounding is strong (avg similarity {avg_similarity:.2f})."
    }

if __name__=="__main__":
    index=load_index()
    few_shot = load_few_shot_examples()

    test_message = "my phone screen went completely black and won't respond to anything"
    intent = classify_intent(test_message,few_shot)
    retrieved = retrieve_similar(test_message,index,top_k=3)
    grounded = [ex for ex in retrieved if ex["similarity"]>=0.5]

    result = decide_escalation(test_message,intent,grounded)
    print(f"Message: {test_message}")
    print(f"Intent: {intent}")
    print(f"Decision: {result["decision"]}")
    print(f"Reason: {result["reason"]}")