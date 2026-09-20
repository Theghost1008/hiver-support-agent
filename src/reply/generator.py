import os
import re
from groq import Groq
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve_similar
from src.retrieval.retriever import load_index

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

SIMILARITY_THRESHOLD = 0.5

def strip_urls(texts: str)->str:
    return re.sub(r'https?://\S+', '',texts).strip()

def build_reply_prompt(message: str, grounded_examples: list[dict])->str:
    if grounded_examples:
        example_block = "\n\n".join(
            f'Customer: "{ex["historical_customer_texts"]}"\n'
            f'Apple\'s actual reply: "{strip_urls(ex["historical_apple_reply"])}"'
            for ex in grounded_examples
        )
        grounding_instructions = (
            f"Here are real examples of how AppleSupport had handled similar "
            f"issues in the past: \n\n{example_block}"
            f"Draft a reply to the new message below, in a similar tone and "
            f"style to these real examples."
        )
        grounding_instructions += ( 
            "\n\nDo NOT include any links or URLs in your reply, since you do "
            "not have a real one to provide. If the historical examples "
            "mention DM, just say to continue over DM without including a link."
        )
    else:
        grounding_instructions=(
            "No closely similar histoical examples were found. Draft a "
            "reasonable, brief, professional support reply in AppleSupport's "
            "typical style: acknowledge the issue, and if it requires "
            "account-specific troubleshooting, offer to continue over DM"
        )
    return f"""You are drafting a public Twitter reply as AppleSupport 
    {grounding_instructions}
    New customer message: {message}
    Write ONLY the reply text, nothing else. Keep it brief, consistent with a real Twitter support reply(typicall under 200 characters)."""

def generate_reply(message: str,index:dict)->dict:
    retrieved = retrieve_similar(message,index,top_k=3)
    grounded_examples = [ex for ex in retrieved if ex['similarity']>= SIMILARITY_THRESHOLD]
    prompt = build_reply_prompt(message, grounded_examples)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=300,
        reasoning_effort="low",
    )
    raw_reply = response.choices[0].message.content
    reply_text = raw_reply.strip() if raw_reply else ""

    return {
        "reply":reply_text,
        "num_grounded_examples":len(grounded_examples),
        "grounding_similarities":[ex["similarity"] for ex in grounded_examples]
    }

if __name__=="__main__":
    index = load_index()
    test_message = "my phone screen went completely black and won't respond to anything"
    result = generate_reply(test_message,index)
    print(f"Message: {test_message}")
    print(f"Grounded on {result['num_grounded_examples']} historical examples"
          f"(similarities: {result['grounding_similarities']})")
    print(f"Generated reply: {result['reply']}")
