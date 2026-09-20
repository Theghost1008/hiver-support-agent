from src.intents.classifier import classify_intent,load_few_shot_examples
from src.retrieval.retriever import load_index,retrieve_similar
from src.reply.generator import generate_reply,SIMILARITY_THRESHOLD
from src.routing.escalation import decide_escalation

_index = None
_few_shot = None

def _get_resources():
    global _index, _few_shot
    if _index is None:
        _index=load_index()
    if _few_shot is None:
        _few_shot = load_few_shot_examples()
    return _index,_few_shot

def process_message(message: str)-> dict:
    index,few_shot = _get_resources()
    intent = classify_intent(message,few_shot)

    retrieved = retrieve_similar(message,index,top_k=3)
    grounded_examples = [ex for ex in retrieved if ex["similarity"]>=SIMILARITY_THRESHOLD]

    reply_result = generate_reply(message,index)
    escalation_result = decide_escalation(message,intent,grounded_examples)

    return {
        "message":message,
        "intent":intent,
        "reply":reply_result["reply"],
        "escalation_decision":escalation_result["decision"],
        "escalation_reason": escalation_result["reason"],
        "num_grounded_examples": reply_result["num_grounded_examples"],
        "grounding_similarities": reply_result["grounding_similarities"],
    }

if __name__=="__main__":
    test_messages = [
        "my phone screen went completely black and won't respond to anything",
        "I upgraded my iPhone and now it crashes at 32% battery, please help",
        "my iphone 5C caught fire in the classroom, what should I do?",
    ]
    for msg in test_messages:
        result = process_message(msg)
        print(f"Message: {result['message']}")
        print(f"Intent: {result['intent']}")
        print(f"Decision: {result['escalation_decision']} — {result['escalation_reason']}")
        print(f"Reply: {result['reply']}")
        print()