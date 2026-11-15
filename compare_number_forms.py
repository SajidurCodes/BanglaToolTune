import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

data_dir = Path(__file__).parent / "data"
checkpoint = Path.home() / "Downloads" / "banglatooltune-8steps-b451e41"

with (data_dir / "train.jsonl").open(encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]
tools = next(r["input"]["available_tools"] for r in records if r["id"] == "train-0007")

with (data_dir / "train_chat.jsonl").open(encoding="utf-8") as file:
    system_message = json.loads(file.readline())["messages"][0]

config = PeftConfig.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("Loading checkpoint:", checkpoint.name, flush=True)
base = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, dtype=torch.float32, device_map={"": "cpu"}
)
model = PeftModel.from_pretrained(base, checkpoint)
model.eval()

dataset_id = "77777777-7777-4777-8777-777777777777"
expected = {
    "type": "tool_call",
    "name": "preview_dataset",
    "arguments": {"dataset_id": dataset_id, "limit": 4},
}


@torch.inference_mode()
def run_and_score(label, inputs):
    output = model.generate(**inputs, do_sample=False, max_new_tokens=160)
    reply = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"{label}: {reply}")
    try:
        prediction = json.loads(reply)
    except json.JSONDecodeError:
        print("FAIL: invalid standalone JSON")
    else:
        print("PASS: exact call" if prediction == expected else "FAIL: wrong call")


for number in ("চারটা", "৪টা"):
    question = f"হাজিরার ফাইলটা একটু দেখি। শুরুর {number} রেকর্ড দেখাবে?"
    case_input = {
        "question": question,
        "dataset_id": dataset_id,
        "columns": ["student_id", "class_date", "attendance_status"],
        "available_tools": tools,
    }
    messages = [
        system_message,
        {"role": "user", "content": json.dumps(case_input, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

    print(f"\nQuestion: {question}", flush=True)
    with model.disable_adapter():
        run_and_score("Base", inputs)
    run_and_score("Eight-update adapter", inputs)