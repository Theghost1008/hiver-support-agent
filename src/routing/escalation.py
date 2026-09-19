from langdetect import detect, LangDetectException
from src.intents.classifier import classify_intent, load_few_shot_examples
from src.retrieval.retriever import load_index, retrieve_similar

SHORT_MSG_WORD_THRESHOLD = 5
WEAK_GROUNDING_SIMILARITY = 0.55

ALWAYS_ESCALATE_INTENTS = {
    "account_security","billing_payment","other_unclear","support_escalation"
}

def is_non_eng(text:str)->bool:
    try:
        return detect(text)!="en"
    except LangDetectException:
        return False

def decide_escalation(message: str, intent:str, grounded_examples: list[dict])->dict:
    word_count = len(message.split())
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