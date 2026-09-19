import os
import time
import json
from groq import Groq, RateLimitError
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

def classify_intent(message: str, few_shot_examples: list[dict],max_retries: int=3)->str:
    prompt = build_classification_prompt(message,few_shot_examples)
    for attempt in range(max_retries):
        try:
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
        except RateLimitError as e:
            wait_time = 100
            print(f"Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError(f"Failed after {max_retries} retries due to persistent rate limiting")
    

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

def classify_intent_batch(messages: list[str], few_shot_examples: list[dict], max_retries: int = 3) -> list[str]:
    taxonomy_block = "\n".join(f"- {label}: {desc}" for label, desc in INTENT_TAXONOMY.items())
    examples_block = "\n".join(
        f'Message: "{ex["text"]}"\nIntent: {ex["intent"]}\n' for ex in few_shot_examples
    )
    numbered_messages = "\n".join(f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages))

    prompt = f"""You are classifying customer support messages sent to Apple Support on Twitter.

    Available intent categories:
    {taxonomy_block}

    Examples of correctly labeled messages:
    {examples_block}

    Classify each of the following {len(messages)} messages. Respond with ONLY a
    JSON array of {len(messages)} intent labels, in the same order as the messages,
    nothing else. Example format: ["bug_report", "battery_performance", ...]

    Messages:
    {numbered_messages}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
                reasoning_effort="low",
            )
            raw_output = response.choices[0].message.content
            return _parse_batch_labels(raw_output, expected_count=len(messages))

        except RateLimitError:
            wait_time = 100
            print(f"Batch rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
            time.sleep(wait_time)

    
    return ["other_unclear"] * len(messages)


def _parse_batch_labels(raw_output: str | None, expected_count: int) -> list[str]:
    """
    Defensively parses the model's JSON array response. Falls back to
    'other_unclear' for individual entries that are missing, malformed,
    or not a recognized label, rather than failing the whole batch.
    """
    if raw_output is None:
        return ["other_unclear"] * expected_count

    try:
        # model may wrap the JSON in commentary or code fences; extract the array
        start = raw_output.index("[")
        end = raw_output.rindex("]") + 1
        labels = json.loads(raw_output[start:end])
    except (ValueError, json.JSONDecodeError):
        return ["other_unclear"] * expected_count

    # Pad or truncate to match expected count, validating each label
    result = []
    for i in range(expected_count):
        if i < len(labels) and labels[i] in INTENT_TAXONOMY:
            result.append(labels[i])
        else:
            result.append("other_unclear")
    return result

if __name__ == "__main__":
    examples = load_few_shot_examples(example_per_label=1)
    test_msg = "My phone screen went completely black and won't respond to anything"
    predicted = classify_intent(test_msg,examples)
    print(f"Message: {test_msg}")
    print(f"Predicted intent: {predicted}")