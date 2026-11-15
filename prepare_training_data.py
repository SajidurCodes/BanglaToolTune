import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCE = DATA_DIR / "train.jsonl"
OUTPUT = DATA_DIR / "train_chat.jsonl"

# Keep the instruction identical to the one used for local evaluation.
SYSTEM_PROMPT = (
    "Return one JSON object only. For a supported request, use type "
    "'tool_call', name, and arguments. For an ambiguous or unsupported "
    "request, use type 'clarification' or 'unsupported' and a Bangla message."
)


def main() -> None:
    conversations = []

    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(record["input"], ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(record["target"], ensure_ascii=False)},
        ]
        conversations.append(json.dumps({"messages": messages}, ensure_ascii=False))

    if not conversations:
        raise ValueError(f"No training records found in {SOURCE}")

    OUTPUT.write_text("\n".join(conversations) + "\n", encoding="utf-8")
    print(f"Wrote {len(conversations)} conversations to {OUTPUT.name}")


if __name__ == "__main__":
    main()