# Label schema

Multiclass classification. Three classes:

| id | class | description | typical sources |
|----|-------|-------------|-----------------|
| 0 | `benign` | No attempt to manipulate system instructions. Includes mundane and hard/confusing benign. | Alpaca/Dolly (mundane), XSTest, OR-Bench, benign portion of WildGuardMix / ToxicChat |
| 1 | `prompt_injection` | Attempt to override/manipulate the system instructions or exfiltrate the system prompt ("ignore previous instructions", "show your system prompt"). | deepset/prompt-injections, Gandalf |
| 2 | `jailbreak` | Attempt to bypass safety policies to obtain prohibited content (role-play, DAN, personas, framing). | In-The-Wild, JailbreakBench, HackAPrompt |

## Open labeling decision

Direct harmful requests without adversarial framing (plain HarmBench, e.g. "how to build ransomware"
with no disguise): **recommended to fold into `jailbreak`** (same guardrail action: block), or add a 4th
`harmful_direct` class only if that granularity is wanted. Decide when writing `build_dataset.py`.

## Notes

- **Deduplicate across sources before splitting** — AdvBench appears inside both HarmBench and JailbreakBench.
- Hold out a set of **unseen attacks** (attack styles not present in training) to measure generalization.
- Watch the benign mix: mostly *mundane* benign, minority *hard* benign, so the model learns
  "benign = no manipulation attempt", not "benign = sounds weird".
