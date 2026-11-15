import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["zero", "few"], default="zero")
parser.add_argument("--case", default="dev-0001")
args = parser.parse_args()

data_dir = Path(__file__).parent / "data"


def load_records(filename):
    with (data_dir / filename).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def user_message(record):
    return {"role": "user", "content": json.dumps(record["input"], ensure_ascii=False)}


def score_reply(reply, target):
    try:
        prediction = json.loads(reply)
    except json.JSONDecodeError:
        return "FAIL: reply is not valid standalone JSON"

    if target["type"] == "tool_call":
        if prediction == target:
            return "PASS: exact tool call"
        return "FAIL: JSON differs from the expected tool call"

    if not isinstance(prediction, dict) or set(prediction) != {"type", "message"}:
        return "FAIL: expected an object with exactly type and message"
    if prediction["type"] != target["type"]:
        return f"FAIL: expected response type {target['type']}"
    if not isinstance(prediction["message"], str) or not prediction["message"].strip():
        return "FAIL: expected a nonempty message"

    return "REVIEW: structure passed. Check the language and meaning of the message."


cases = {record["id"]: record for record in load_records("dev.jsonl")}
if args.case not in cases:
    parser.error(f"Unknown case {args.case}. Available cases: {', '.join(cases)}")
case = cases[args.case]

instructions = (
    "Return one JSON object only. For a supported request, use type "
    "'tool_call', name, and arguments. For an ambiguous or unsupported "
    "request, use type 'clarification' or 'unsupported' and a Bangla message."
)
messages = [{"role": "system", "content": instructions}]

if args.mode == "few":
    training = {record["id"]: record for record in load_records("train.jsonl")}
    for example_id in ("train-0001", "train-0002"):
        example = training[example_id]
        messages.append(user_message(example))
        messages.append({
            "role": "assistant",
            "content": json.dumps(example["target"], ensure_ascii=False),
        })

messages.append(user_message(case))
payload = {
    "model": "qwen2.5:1.5b",
    "messages": messages,
    "stream": False,
    "options": {"temperature": 0},
}

request = Request(
    "http://127.0.0.1:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=180) as response:
    reply = json.load(response)["message"]["content"]

print(f"Case: {args.case} | Mode: {args.mode}")
print("Raw model reply:")
print(reply)
print(score_reply(reply, case["target"]))