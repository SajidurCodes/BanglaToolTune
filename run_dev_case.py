import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["zero", "few"], default="zero")
args = parser.parse_args()

data_dir = Path(__file__).parent / "data"


def load_records(filename):
    with (data_dir / filename).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def user_message(record):
    return {"role": "user", "content": json.dumps(record["input"], ensure_ascii=False)}


case = next(record for record in load_records("dev.jsonl") if record["id"] == "dev-0001")

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

print(f"Case: {case['id']} | Mode: {args.mode}")
print("Raw model reply:")
print(reply)

try:
    prediction = json.loads(reply)
except json.JSONDecodeError:
    print("FAIL: reply is not valid standalone JSON")
else:
    if prediction == case["target"]:
        print("PASS: exact tool call")
    else:
        print("FAIL: JSON differs from target")
        print("Expected:", json.dumps(case["target"], ensure_ascii=False))