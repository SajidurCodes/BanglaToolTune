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


def user_message(record: dict) -> dict:
    return {
        "role": "user",
        "content": json.dumps(record["input"], ensure_ascii=False),
    }


@torch.inference_mode()
def generate_reply(model, tokenizer, inputs) -> str:
    output = model.generate(**inputs, do_sample=False, max_new_tokens=160)
    answer = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(answer, skip_special_tokens=True)


parser = argparse.ArgumentParser()
parser.add_argument("--cases", nargs="+", default=["dev-0001"])
modes = parser.add_mutually_exclusive_group()
modes.add_argument("--compare", action="store_true")
modes.add_argument("--two-shot", action="store_true")
args = parser.parse_args()

project = Path(__file__).resolve().parent
data_dir = project / "data"
adapter_dir = Path.home() / "Downloads" / "banglatooltune-pilot-7steps-081c0f9"

with (data_dir / "train_chat.jsonl").open(encoding="utf-8") as file:
    system_message = json.loads(file.readline())["messages"][0]

with (data_dir / "dev.jsonl").open(encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]
cases = {record["id"]: record for record in records}

for case_id in args.cases:
    if case_id not in cases:
        parser.error(f"Unknown case {case_id}. Available: {', '.join(cases)}")

prefix = [system_message]
if args.two_shot:
    with (data_dir / "train.jsonl").open(encoding="utf-8") as file:
        examples = [json.loads(line) for line in file if line.strip()]
    training = {record["id"]: record for record in examples}

    for example_id in ("train-0001", "train-0002"):
        example = training[example_id]
        prefix.append(user_message(example))
        prefix.append({
            "role": "assistant",
            "content": json.dumps(example["target"], ensure_ascii=False),
        })

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
    messages = [*prefix, user_message(case)]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    print(f"\n{case_id} | {case['input']['question']}", flush=True)

    if args.compare or args.two_shot:
        print("Base (two-shot):" if args.two_shot else "Base (zero-shot):", flush=True)
        with model.disable_adapter():
            reply = generate_reply(model, tokenizer, inputs)
        print(reply)
        print(score_reply(reply, case["target"]))

    if not args.two_shot:
        print("Adapter:", flush=True)
        reply = generate_reply(model, tokenizer, inputs)
        print(reply)
        print(score_reply(reply, case["target"]))