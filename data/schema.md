# Label schema

Multiclass classification. Three classes:

| id | class | description | typical sources |
|----|-------|-------------|-----------------|
| 0 | `benign` | No attempt to manipulate system instructions. Includes mundane and hard/confusing benign. | Alpaca/Dolly (mundane), XSTest, OR-Bench, benign portion of WildGuardMix / ToxicChat |
| 1 | `prompt_injection` | Attempt to override/manipulate the system instructions or exfiltrate the system prompt ("ignore previous instructions", "show your system prompt"). | deepset/prompt-injections, Gandalf |
| 2 | `jailbreak` | Attempt to bypass safety policies to obtain prohibited content (role-play, DAN, personas, framing). | In-The-Wild, JailbreakBench, HackAPrompt |

## Labeling decision: direct harmful requests without framing

Closed: folded into `jailbreak` (same guardrail action: block), no 4th `harmful_direct` class. Implemented
in `data/build_dataset.py` for: all of HarmBench, XSTest's `unsafe` split, and OR-Bench's `or-bench-toxic`
config. WildGuardMix's `harmful`-labeled prompts fold the same way (it's a general harm annotation, not
injection-specific).

## Notes

- **Deduplicate across sources before splitting** — AdvBench appears inside both HarmBench and JailbreakBench.
- Hold out a set of **unseen attacks** (attack styles not present in training) to measure generalization.
- Watch the benign mix: mostly *mundane* benign, minority *hard* benign, so the model learns
  "benign = no manipulation attempt", not "benign = sounds weird".

## Sources (HF Hub repo ids, verified)

Confirmed against the HF Hub API (`gated`/`license` fields) and datasets-server. Used by `data/download.py`.

| Source | Repo id | Config(s) | License | Gated | Notes |
|---|---|---|---|---|---|
| JailbreakBench | `JailbreakBench/JBB-Behaviors` | `behaviors`, `judge_comparison` | MIT | No | Official repo. |
| HarmBench | `walledai/HarmBench` | default | MIT | Yes (auto) | No official HF repo from the authors (GitHub CSV only); this is a same-license mirror. |
| HackAPrompt | `hackaprompt/hackaprompt-dataset` | default | MIT | Yes (auto) | ~600k rows; `dataset` column splits `playground_data` vs `submission_data`. May contain offensive content. |
| Gandalf (Lakera) | `Lakera/gandalf_ignore_instructions` | default | MIT | No | 1000 rows, has train/val/test. |
| deepset/prompt-injections | `deepset/prompt-injections` | default | Apache-2.0 | No | train 546 / test 116. |
| WildGuardMix | `allenai/wildguardmix` | `wildguardtrain`, `wildguardtest` | odc-by | Yes (auto, AI2 responsible-use terms) | Contains explicitly disturbing content by design; don't train on harmful examples alone. |
| XSTest | `Paul/XSTest` | default | CC-BY-4.0 | No | Use this, **not** `walledai/XSTest` (that mirror is gated). |
| OR-Bench | `bench-llms/or-bench` | `or-bench-80k`, `or-bench-hard-1k`, `or-bench-toxic` | CC-BY-4.0 | No | Note the "s" in `bench-llms` (org the paper cites); `bench-llm/or-bench` is an unofficial duplicate. |
| ToxicChat | `lmsys/toxic-chat` | `toxicchat0124` | **CC-BY-NC-4.0** | No | Non-commercial license — fine for this portfolio/demo, not for a revenue product. |

**Gated sources** (HarmBench, HackAPrompt, WildGuardMix) need an HF account that has clicked "Agree and
access repository" on each dataset page, plus `HF_TOKEN` set in the environment before running
`data/download.py`.
