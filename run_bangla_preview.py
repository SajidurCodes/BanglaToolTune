import asyncio
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parent
DATABRIDGE = Path(r"C:\Github 2\databridge-mcp")
MODEL_PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
CHECKPOINT = Path.home() / "Downloads" / "banglatooltune-8steps-b451e41"


def predict() -> None:
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    case_input = json.loads(sys.stdin.buffer.read())
    with (PROJECT / "data" / "train_chat.jsonl").open(encoding="utf-8") as file:
        system_message = json.loads(file.readline())["messages"][0]

    config = PeftConfig.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        dtype=torch.float32,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(base, CHECKPOINT)
    model.eval()

    messages = [
        system_message,
        {"role": "user", "content": json.dumps(case_input, ensure_ascii=False)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=160)

    reply = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    sys.stdout.buffer.write(reply.encode("utf-8"))


async def run_preview() -> None:
    from mcp import Client, StdioServerParameters

    sys.path.insert(0, str(DATABRIDGE))
    from databridge.storage import DatasetStore

    with (PROJECT / "data" / "train.jsonl").open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    tools = next(r["input"]["available_tools"] for r in records if r["id"] == "train-0007")

    columns = ["student_id", "class_date", "attendance_status"]
    rows = [
        ["S01", "2026-11-15", "present"],
        ["S02", "2026-11-15", "absent"],
        ["S03", "2026-11-15", "present"],
        ["S04", "2026-11-15", "late"],
        ["S05", "2026-11-15", "present"],
    ]
    csv_bytes = "\n".join(",".join(row) for row in [columns, *rows]).encode("utf-8")

    with TemporaryDirectory() as temp_dir:
        database = Path(temp_dir) / "datasets.sqlite3"
        dataset = DatasetStore(database).add_csv("attendance.csv", csv_bytes)
        case_input = {
            "question": "হাজিরার ফাইলটা একটু দেখি। শুরুর ৪টা রেকর্ড দেখাবে?",
            "dataset_id": dataset.dataset_id,
            "columns": columns,
            "available_tools": tools,
        }

        print("Selected dataset:", dataset.dataset_id, flush=True)
        print("Generating the tool call on CPU...", flush=True)
        process = subprocess.run(
            [str(MODEL_PYTHON), str(Path(__file__).resolve()), "--predict"],
            input=json.dumps(case_input, ensure_ascii=False).encode("utf-8"),
            stdout=subprocess.PIPE,
            check=True,
        )
        reply = process.stdout.decode("utf-8")
        print("Raw model reply:", reply, sep="\n", flush=True)

        try:
            call = json.loads(reply)
        except json.JSONDecodeError:
            raise SystemExit("FAIL: invalid standalone JSON. No tool was executed.")

        expected = {
            "type": "tool_call",
            "name": "preview_dataset",
            "arguments": {"dataset_id": dataset.dataset_id, "limit": 4},
        }
        if call != expected or type(call["arguments"]["limit"]) is not int:
            raise SystemExit("FAIL: prediction differs from the expected call. No tool was executed.")

        server = StdioServerParameters(
            command=sys.executable,
            args=[str(DATABRIDGE / "tests" / "mcp_test_server.py"), str(database)],
        )
        async with asyncio.timeout(45):
            async with Client(server) as client:
                result = await client.call_tool(call["name"], call["arguments"])

        assert result.is_error is False, result.content
        preview = result.structured_content
        assert preview is not None
        assert preview["columns"] == columns
        assert preview["rows"] == rows[:4]
        assert preview["total_rows"] == 5 and preview["returned_rows"] == 4
        assert preview["has_more_rows"] is True
        print("PASS: Qwen generated the call; MCP returned S01 through S04 in order.")


if __name__ == "__main__":
    if sys.argv[1:] == ["--predict"]:
        predict()
    else:
        asyncio.run(run_preview())