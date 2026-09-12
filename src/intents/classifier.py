import os
import json
from groq import Groq
from dotenv import load_dotenv
from src.intents.taxonomy import INTENT_TAXONOMY
import pandas as pd
from pathlib import Path

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL = "openai/gpt-oss-120b"

def build_classification_prompt(message: str, few_shot_examples: list[dict])->str:
    taxonomy_block = "\n".join(f"- {label}:{desc}" for label,desc in INTENT_TAXONOMY.items())
    examples_block = "\n".join(f'Message- "{ex["text"]}"\nIntent: "{ex["intent"]}"\n' for ex in few_shot_examples)
    return f"""You are classifying customer support messages sent to Apple Support on Twitter.

Available intent categories:
{taxonomy_block}

Examples of correctly labeled messages:
{examples_block}

Now classify this new message. Respond with ONLY the intent label, nothing else.

Message: "{message}"
Intent:"""

def classify_intent(message: str, few_shot_examples: list[dict])->str:
    prompt = build_classification_prompt(message,few_shot_examples)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=300,
        reasoning_effort="low"
    )
    raw_output = response.choices[0].message.content
    if raw_output is None:
        return "other_unclear"
    return _parse_intent_label(raw_output.strip())

def _parse_intent_label(raw_output: str)->str:
    """
    Defends against the model adding commentary around the label
    instead of returning it cleanly. Falls back to 'other_unclear'
    if nothing recognizable is found, rather than crashing.
    """
    lowered = raw_output.lower()
    for label in INTENT_TAXONOMY:
        if label in lowered:
            return label
    return "other_unclear"

def load_few_shot_examples(example_per_label: int = 1)->list[dict]:
    """Pulls a small number of real labeled examples per intent category
    from the manually labeled taxonomy samples"""
    df = pd.read_csv(PROJECT_ROOT/"data"/"taxonomy_reading_sample_filled.csv")
    df = df.dropna(subset=["rough_bucket"])
    examples = []
    for label in INTENT_TAXONOMY:
        matches = df[df["rough_bucket"]==label].head(example_per_label)
        for _, row in matches.iterrows():
            examples.append({"text":row["text_customer"],"intent":label})
    return examples

if __name__ == "__main__":
    examples = load_few_shot_examples(example_per_label=1)
    test_msg = "My phone screen went completely black and won't respond to anything"
    predicted = classify_intent(test_msg,examples)
    print(f"Message: {test_msg}")
    print(f"Predicted intent: {predicted}")