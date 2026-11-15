import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def score_reply(reply: str, target: dict) -> str:
    try:
        prediction = json.loads(reply)
    except json.JSONDecodeError:
        return "FAIL: invalid standalone JSON"

    if target["type"] == "tool_call":
        return "PASS: exact tool call" if prediction == target else "FAIL: wrong call"

    if not isinstance(prediction, dict) or prediction.get("type") != target["type"]:
        return f"FAIL: expected {target['type']}"

    if set(prediction) != {"type", "message"}:
        return "FAIL: expected exactly type and message"

    message = prediction["message"]
    if not isinstance(message, str) or not message.strip():
        return "FAIL: empty or invalid message"

    return "REVIEW: check the Bangla message against the question"


@torch.inference_mode()
def generate_reply(model, tokenizer, inputs) -> str:
    output = model.generate(**inputs, do_sample=False, max_new_tokens=160)
    answer = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)


parser = argparse.ArgumentParser()
parser.add_argument("--cases", nargs="+", default=["dev-0001"])
parser.add_argument("--compare", action="store_true")
args = parser.parse_args()

project = Path(__file__).resolve().parent
adapter_dir = Path.home() / "Downloads" / "banglatooltune-pilot-7steps-081c0f9"

with (project / "data" / "train_chat.jsonl").open(encoding="utf-8") as file:
    system_message = json.loads(file.readline())["messages"][0]

with (project / "data" / "dev.jsonl").open(encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]
cases = {record["id"]: record for record in records}

for case_id in args.cases:
    if case_id not in cases:
        parser.error(f"Unknown case {case_id}. Available: {', '.join(cases)}")

config = PeftConfig.from_pretrained(adapter_dir)
print("Loading on CPU:", config.base_model_name_or_path, flush=True)
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    dtype=torch.float32,
    device_map={"": "cpu"},
)
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

for case_id in args.cases:
    case = cases[case_id]
    messages = [
        system_message,
        {"role": "user", "content": json.dumps(case["input"], ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    print(f"\n{case_id} | {case['input']['question']}", flush=True)

    if args.compare:
        print("Base:", flush=True)
        with model.disable_adapter():
            reply = generate_reply(model, tokenizer, inputs)
        print(reply)
        print(score_reply(reply, case["target"]))

    print("Adapter:", flush=True)
    reply = generate_reply(model, tokenizer, inputs)
    print(reply)
    print(score_reply(reply, case["target"]))