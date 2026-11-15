# BanglaToolTune

BanglaToolTune explores whether a small language model can turn a Bangla question about a CSV into the right action: call a DataBridge tool, ask a clarification question in Bangla, or explain that the available tools cannot fulfill the request.

I fine-tuned Qwen2.5-1.5B-Instruct with QLoRA on a Colab T4, ran the adapter on a Windows laptop, and compared it with zero-shot and two-shot prompting. The adapter ran successfully end to end, but two-shot prompting earned the highest score on the three held-out questions.

## A verified local demo

> হাজিরার ফাইলটা একটু দেখি। শুরুর ৪টা রেকর্ড দেখাবে?

The demo creates a five-row attendance CSV in a temporary DataBridge store. It gives the model the Bangla question, column names, available tool descriptions, and the dataset ID generated for that run. In the verified run, Qwen produced:

```json
{
  "type": "tool_call",
  "name": "preview_dataset",
  "arguments": {
    "dataset_id": "a226cacb7bf9451d8c7691deafe4982e",
    "limit": 4
  }
}
```

The demo checked the prediction against the expected call, then passed the model-generated arguments to a local MCP client over stdio. DataBridge returned students `S01` through `S04` in order. The fifth row stayed outside the preview. This demo verifies one fixed request; it does not establish accuracy on arbitrary CSV questions.

## Training and evaluation

The final adapter used eight labeled training conversations and eight optimizer updates. Training used a 4-bit NF4 base model and rank-8 LoRA adapters on the query and value projections. Only the assistant’s JSON response and end marker contributed to training loss.

The held-out comparison used the same Windows CPU, FP32 base model, greedy decoding, and three library-loan questions for all approaches. The library-loan schema was reserved from training. The two-shot prompt included one tool call and one clarification example.

| Method | Held-out cases passed |
|---|---:|
| Zero-shot base | 0/3 |
| Two-shot base | 1/3 |
| Eight-update QLoRA adapter | 0/3 |

Two-shot prompting returned the same valid `profile_dataset` call three times. It answered the missing-return-date question correctly, then missed both the ambiguous date question and the unsupported frequency question. The adapter showed no accuracy gain on this set. All three questions were assistant-authored and used one schema. [Per-case results](RESULTS.md)

## Run the verified Windows demo

The commands below assume Python 3.11, a BanglaToolTune virtual environment at `.venv`, and a separate [DataBridge MCP checkout](https://github.com/SajidurCodes/Databridge-MCP) at `C:\Github 2\databridge-mcp` with its own working `.venv`.

The model runner expects the extracted eight-update adapter and tokenizer at `%USERPROFILE%\Downloads\banglatooltune-8steps-b451e41`. The adapter was exported from Colab separately. A fresh clone of this repository alone does not supply those weights.

From the BanglaToolTune repository root:

```cmd
"C:\Github 2\databridge-mcp\.venv\Scripts\python.exe" check_mcp_preview.py
"C:\Github 2\databridge-mcp\.venv\Scripts\python.exe" -u run_bangla_preview.py
```

The first command checks DataBridge with a fixed call. The second starts local Qwen inference, validates the generated call, executes it, and checks the returned rows.

To run the held-out comparison with the same checkpoint:

```cmd
.venv\Scripts\python.exe -u run_heldout.py --compare
.venv\Scripts\python.exe -u run_heldout.py --two-shot
```

## Repository guide

| Path | Purpose |
|---|---|
| `data/train.jsonl` | Labeled Bangla inputs, targets, and evaluation notes |
| `prepare_training_data.py` | Converts training records into chat conversations |
| `data/dev.jsonl` and `data/test.jsonl` | Development cases and reserved evaluation cases |
| `run_heldout.py` | Runs and scores the three held-out approaches |
| `compare_number_forms.py` | Compares the Bangla word `চারটা` with the digit form `৪টা` |
| `check_mcp_preview.py` | Checks a fixed tool call through the real MCP client |
| `run_bangla_preview.py` | Runs the model-generated attendance preview end to end |