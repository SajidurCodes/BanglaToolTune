# BanglaToolTune pilot results

## Held-out evaluation

Three assistant-authored Bangla questions used a library-loan CSV schema reserved from training and few-shot examples. All three approaches used Qwen2.5-1.5B-Instruct on Windows CPU with FP32 weights and greedy decoding. The two-shot prompt included one tool-call example and one clarification example. The QLoRA adapter was trained on eight examples for eight optimizer updates.

| Held-out task | Zero-shot base | Two-shot base | QLoRA adapter |
|---|---|---|---|
| Count missing return dates | Correct tool and ID, invalid JSON | Pass | Asked for clarification |
| Clarify which of three dates the user meant | Called profile_dataset | Called profile_dataset | Called profile_dataset |
| Explain why the available tool cannot identify the most frequent title | Called profile_dataset | Called profile_dataset | Asked for clarification |
| **Cases passed** | **0/3** | **1/3** | **0/3** |

Two-shot prompting produced the same valid tool call for all three questions. It passed the one case where that call was appropriate. The adapter showed no accuracy gain on this test set. Three questions on one schema are too few to estimate broader Bangla routing accuracy.

## Live MCP demo

In a separate Windows CPU run, the eight-update adapter selected preview_dataset, copied a freshly generated dataset ID, and requested a limit of 4. The demo validated the call and executed it through a local MCP client. DataBridge returned the first four rows of a five-row attendance CSV in order.