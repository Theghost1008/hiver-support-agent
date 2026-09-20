"""
LLM-as-judge: scores each generated reply as PASS/FAIL against the
human-written good_reply_criteria for that row, with a stated reason.
"""
import os
import json
import time
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

def build_judge_prompt(message: str, criteria: str, reply: str) -> str:
    return f"""You are evaluating whether a customer support reply meets
specific quality criteria. Be strict but fair.

Customer message: "{message}"

Criteria the reply should meet: {criteria}

Generated reply to evaluate: "{reply}"

First, identify the distinct individual requirements within the criteria
(e.g. "ask device model" and "offer troubleshooting" count as 2 separate
requirements). Then evaluate how many the reply actually satisfies.

Respond with ONLY a JSON object in this exact format, nothing else:
{{"verdict": "PASS" or "FAIL", "criteria_met": <number>, "criteria_total": <number>, "reason": "one brief sentence"}}

"verdict" is "PASS" only if criteria_met equals criteria_total (ALL
requirements satisfied). "criteria_met" and "criteria_total" should
reflect partial credit even when verdict is FAIL."""


def judge_reply(message: str, criteria: str, reply: str, max_retries: int = 3) -> dict:
    prompt = build_judge_prompt(message, criteria, reply)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                reasoning_effort="low",
            )
            raw_output = response.choices[0].message.content
            return _parse_judge_response(raw_output)

        except RateLimitError:
            wait_time = 100
            print(f"Judge rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
            time.sleep(wait_time)

    return {"verdict": "FAIL", "reason": "Judge call failed after retries"}


def _parse_judge_response(raw_output: str | None) -> dict:
    default = {"verdict": "FAIL", "criteria_met": 0, "criteria_total": 1, "reason": "Empty or unparseable judge response"}
    if raw_output is None:
        return default
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        parsed = json.loads(raw_output[start:end])

        verdict = parsed.get("verdict", "FAIL")
        if verdict not in ("PASS", "FAIL"):
            verdict = "FAIL"

        criteria_met = int(parsed.get("criteria_met", 0))
        criteria_total = int(parsed.get("criteria_total", 1))
        criteria_total = max(criteria_total, 1)  # guard against division by zero later
        criteria_met = min(criteria_met, criteria_total)  # sanity guard

        return {
            "verdict": verdict,
            "criteria_met": criteria_met,
            "criteria_total": criteria_total,
            "reason": parsed.get("reason", ""),
        }
    except (ValueError, json.JSONDecodeError, TypeError):
        return default