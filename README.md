# Guardrail

A specialized **jailbreak / prompt-injection classifier** that screens prompts before they reach an
LLM — trained, evaluated and (as MLOps) deployed on AWS SageMaker. This README is written as a small
research report: question → hypothesis → method → results → discussion → conclusions.

🇬🇧 **English** (below) · 🇲🇽 [**Español**](#-guardrail-español)

---

## Research question

Modern LLM apps need a guardrail that decides, on the hot path, whether an incoming prompt is a
**jailbreak** or a **prompt-injection** attempt before the model ever sees it. There are two obvious
ways to build one:

1. Fine-tune a **specialized classifier** on known attacks.
2. Prompt a **general-purpose LLM** to judge each request.

**The question:** for jailbreak / prompt-injection detection, which approach is better — and *better
on which axis*? Recall, false positives, latency, cost, and (the interesting one) **generalization to
attack styles never seen in training**.

## Hypothesis

> A specialized classifier fine-tuned on attack data will **outperform a general-purpose LLM guardrail**
> — higher recall, lower false positives, and orders of magnitude cheaper and faster per request.

Implicit sub-hypothesis worth testing on its own: **a bigger model generalizes better.** If true, the
1.5B LLM-as-classifier should beat the small encoders on unseen attacks.

## Method

- **Task.** Multiclass classification: `benign` (0) / `prompt_injection` (1) / `jailbreak` (2).
- **Data.** 9 public benchmarks (JailbreakBench, HarmBench, HackAPrompt, Gandalf, deepset,
  WildGuardMix, XSTest, OR-Bench, ToxicChat), mapped to the 3 labels, deduped across sources, topped
  up with mundane benign prompts (Alpaca) so the benign class isn't only "scary-but-safe" examples.
  See [`data/schema.md`](data/schema.md).
- **Generalization split.** **HackAPrompt is held out entirely** as `unseen_attacks` — a stylistically
  distinct attack the models never train on. This is the axis the whole experiment hinges on.
- **Candidate classifiers**, all fine-tuned identically (shared code, class-weighted loss for the
  ~0.85% `prompt_injection` class):
  - **DistilBERT** (66M) — small/fast reference.
  - **DeBERTa-v3-base** (184M) — stronger encoder.
  - **Qwen2.5-1.5B** (1.5B) — a small LLM fine-tuned as a classifier.
- **Baseline.** A **general-purpose LLM prompted as a guardrail**, no fine-tuning, via Amazon Bedrock:
  **Haiku 4.5** (cheap/fast) and **Sonnet 4.6** (capable-LLM ceiling), thinking disabled to match a
  realistic hot-path config.
- **Metrics.** macro-F1, per-class recall, false-positive rate, batch=1 latency (p50/p99), and
  **binary attack detection** (`pred ≠ benign`) — the metric that actually matters for a block/allow
  guardrail, since flagging an attack as the *wrong subtype* still blocks it.
- **Infra.** Data prep as SageMaker Processing Jobs; training and evaluation as SageMaker jobs on
  `ml.g5.xlarge` (all models on the same GPU so latency is comparable). Total project cost ≈ **$35**.

## Results

**The central result — binary attack detection (`pred ≠ benign`):**

| Model | `test` (in-distribution) | `unseen` (novel attack) |
|---|---|---|
| DistilBERT (66M, fine-tuned) | 0.909 | 0.308 |
| **DeBERTa-v3 (184M, fine-tuned)** | **0.946** | 0.433 |
| Qwen2.5 (1.5B, fine-tuned) | 0.929 | 0.280 |
| Haiku 4.5 (generic LLM) | 0.816 | 0.576 |
| **Sonnet 4.6 (generic LLM)** | 0.842 | **0.662** |

**In-distribution detail (`test` split):**

| Model | macro-F1 | recall_injection | recall_jailbreak | FP rate | p50 latency | training cost |
|---|---|---|---|---|---|---|
| DistilBERT | 0.938 | 0.921 | 0.905 | 0.022 | **3.2 ms** | $1.49 |
| DeBERTa | 0.960 | 0.937 | 0.946 | 0.015 | 16.5 ms | $2.54 |
| Qwen | 0.960 | 0.944 | 0.928 | 0.016 | 29.7 ms | $22.52 |
| Haiku 4.5 | 0.780 | 0.700 | 0.806 | 0.059 | API call | per-call |
| Sonnet 4.6 | 0.821 | 1.000 | 0.817 | 0.048 | API call | per-call |

- **In-distribution:** the fine-tuned classifiers win clearly (0.91–0.95 detection vs 0.82–0.84 for
  the LLMs), at **~1000× lower cost/latency** per inference (3–30 ms on-device vs a paid API call).
- **Unseen attacks:** the ranking **flips** — the generic LLMs generalize far better (0.58–0.66 vs
  0.28–0.43). On `unseen`, the LLMs label most HackAPrompt prompts as `injection` (Sonnet 576/1000),
  so their binary detection stays high even though exact-subtype recall is low (~4%).
- **Size does not buy generalization:** the 1.5B Qwen is the **worst** generalizer (0.280), below both
  encoders. No monotonic trend with model size.
- **Quality gate:** the gate (set *before* seeing results) requires ≥0.85 recall on unseen jailbreaks.
  **No model passes** — DeBERTa passes every in-distribution threshold but, like all of them, falls
  far short on generalization.

## Discussion

The hypothesis is **half right, and the wrong half is the interesting one.**

- *"Specialized classifier > generic LLM"* holds **in-distribution** — and by a lot, once you factor in
  cost and latency. For known attack patterns, a fine-tuned encoder is the obvious choice.
- It **fails on generalization.** Against a genuinely novel attack style, the classifiers collapse
  (they memorized training-distribution patterns) while the un-fine-tuned LLMs reason about *intent*
  and hold up much better. Each approach wins a different regime.
- The sub-hypothesis (*bigger = better generalization*) is **false here.** Paying 10× the training cost
  and ~9× the latency for Qwen bought the worst generalization of the three. Scale alone didn't help.
- **Honest caveat:** `unseen_attacks` is a single held-out source (HackAPrompt), so it measures
  generalization to *that* style, not in the abstract. The in-distribution → unseen gap is the finding;
  the absolute unseen numbers are directional.

This isn't a failed experiment — it's a more useful answer than the hypothesis. The two approaches are
**complementary**, which points directly at a **cascade** architecture.

## Conclusions

1. **For known attacks, ship the specialized classifier.** **DeBERTa-v3** is the pick: best
   in-distribution detection (0.946), best macro-F1, passes every in-distribution gate, and 5× faster
   than Qwen at a fraction of the cost.
2. **No model is robust to novel attacks on its own** — the best (Sonnet 4.6, 0.662) still misses ~34%.
3. **Bigger is not automatically better.** The 1.5B model was the most expensive *and* the weakest
   generalizer.
4. **The design that follows from the data is a cascade:** the fast/cheap classifier handles the bulk
   of known traffic, with a general-purpose LLM as fallback for the uncertain / novel-looking cases
   where the classifier is weak.

---

## Repo layout

```
data/         # download + build dataset (SageMaker Processing Jobs; output to S3)
training/     # per-model training code (distilbert / deberta / qwen) + shared code
evaluation/   # metrics, unseen-attacks eval, quality gate, LLM baseline
pipelines/    # SageMaker Pipeline definition
infra/        # Terraform (SageMaker, endpoint, API Gateway + Lambda, IAM, monitoring)
inference/    # Lambda handler for the public demo
.github/      # CI/CD workflows (infra + ML)
```

---
---

# 🇲🇽 Guardrail (Español)

Un **clasificador especializado de jailbreak / prompt-injection** que filtra los prompts antes de que
lleguen a un LLM — entrenado, evaluado y desplegado (MLOps) en AWS SageMaker. Este README está escrito
como un pequeño reporte de investigación

## Planteamiento del problema

Toda app con LLM necesita un guardrail que decida, en caliente, si un prompt entrante es un intento de
**jailbreak** o de **prompt-injection** u otros antes de que el modelo lo vea. Hay dos formas obvias de
construirlo:

1. Fine-tunear un **clasificador especializado** sobre ataques conocidos.
2. Pedirle a un **LLM de propósito general** que juzgue cada request.

**La pregunta:** para detección de jailbreak / prompt-injection, ¿qué enfoque es mejor — y *mejor en
qué eje*? Recall, falsos positivos, latencia, costo y (el interesante) **generalización a estilos de
ataque nunca vistos en el entrenamiento**.

## Hipótesis

> Un clasificador especializado fine-tuneado sobre datos de ataque **supera a un guardrail basado en un
> LLM genérico** — mejor recall, menos falsos positivos, y órdenes de magnitud más barato y rápido por
> request.

Sub-hipótesis que vale la pena probar aparte: **un modelo más grande generaliza mejor.** Si es cierta,
el LLM-como-clasificador de 1.5B debería ganarle a los encoders pequeños en ataques no vistos.

## Método

- **Tarea.** Clasificación multiclase: `benign` (0) / `prompt_injection` (1) / `jailbreak` (2).
- **Datos.** 9 benchmarks públicos (JailbreakBench, HarmBench, HackAPrompt, Gandalf, deepset,
  WildGuardMix, XSTest, OR-Bench, ToxicChat), mapeados a las 3 etiquetas, deduplicados entre fuentes y
  complementados con prompts benignos mundanos (Alpaca) para que la clase benigna no sea solo ejemplos
  "que suenan peligrosos pero son inofensivos". Ver [`data/schema.md`](data/schema.md).
- **Split de generalización.** **HackAPrompt se aparta por completo* como `unseen_attacks` — un estilo
  de ataque distinto que los modelos nunca ven en entrenamiento. Es el eje del que depende todo el
  experimento.
- **Clasificadores candidatos**, todos fine-tuneados igual (código compartido, loss ponderado por clase
  para el ~0.85% de `prompt_injection`):
  - **DistilBERT** (66M) — referencia pequeña/rápida.
  - **DeBERTa-v3-base** (184M) — encoder más fuerte.
  - **Qwen2.5-1.5B** (1.5B) — un LLM pequeño fine-tuneado como clasificador.
- **Baseline.** Un **LLM genérico usado como guardrail**, sin fine-tuning, vía Amazon Bedrock:
  **Haiku 4.5** (barato/rápido) y **Sonnet 4.6** (techo de LLM capaz), con thinking apagado para imitar
  una config realista de hot-path.
- **Métricas.** macro-F1, recall por clase, tasa de falsos positivos, latencia batch=1 (p50/p99), y
  **detección binaria de ataque** (`pred ≠ benigno`) — la métrica que de verdad importa para un
  guardrail de bloquear/permitir, porque marcar un ataque con el *subtipo equivocado* igual lo bloquea.
- **Infra.** Preparación de datos como SageMaker Processing Jobs; entrenamiento y evaluación como jobs
  de SageMaker en `ml.g5.xlarge` (todos en la misma GPU para que la latencia sea comparable).

## Resultados

**El resultado central — detección binaria de ataque (`pred ≠ benigno`):**

| Modelo | `test` (in-distribution) | `unseen` (ataque nuevo) |
|---|---|---|
| DistilBERT (66M, fine-tuned) | 0.909 | 0.308 |
| **DeBERTa-v3 (184M, fine-tuned)** | **0.946** | 0.433 |
| Qwen2.5 (1.5B, fine-tuned) | 0.929 | 0.280 |
| Haiku 4.5 (LLM genérico) | 0.816 | 0.576 |
| **Sonnet 4.6 (LLM genérico)** | 0.842 | **0.662** |

**Detalle in-distribution (split `test`):**

| Modelo | macro-F1 | recall_injection | recall_jailbreak | tasa FP | latencia p50 | costo entrenamiento |
|---|---|---|---|---|---|---|
| DistilBERT | 0.938 | 0.921 | 0.905 | 0.022 | **3.2 ms** | $1.49 |
| DeBERTa | 0.960 | 0.937 | 0.946 | 0.015 | 16.5 ms | $2.54 |
| Qwen | 0.960 | 0.944 | 0.928 | 0.016 | 29.7 ms | $22.52 |
| Haiku 4.5 | 0.780 | 0.700 | 0.806 | 0.059 | llamada API | por llamada |
| Sonnet 4.6 | 0.821 | 1.000 | 0.817 | 0.048 | llamada API | por llamada |

- **In-distribution:** los clasificadores fine-tuneados ganan claro (detección 0.91–0.95 vs 0.82–0.84
  de los LLMs), a **~1000× menos costo/latencia** por inferencia (3–30 ms on-device vs una llamada API
  de pago).
- **Ataques no vistos:** el ranking **se invierte** — los LLMs genéricos generalizan mucho mejor
  (0.58–0.66 vs 0.28–0.43). En `unseen`, los LLMs etiquetan la mayoría de HackAPrompt como `injection`
  (Sonnet 576/1000), así que su detección binaria se mantiene alta aunque el recall de subtipo exacto
  sea bajo (~4%).
- **El tamaño no compra generalización:** el Qwen de 1.5B es el **peor** generalizador (0.280), por
  debajo de ambos encoders. No hay tendencia monótona con el tamaño.
- **Quality gate:** el gate (definido *antes* de ver resultados) exige ≥0.85 de recall en jailbreaks no
  vistos. **Ningún modelo pasa** — DeBERTa pasa todos los umbrales in-distribution pero, como todos, se
  queda muy corto en generalización.

## Discusión

La hipótesis es **mitad correcta, y la mitad equivocada es la interesante.**

- *"Clasificador especializado > LLM genérico"* se cumple **in-distribution** — y por mucho, una vez que
  cuentas costo y latencia. Para patrones de ataque conocidos, un encoder fine-tuneado es la elección
  obvia.
- **Falla en generalización.** Contra un estilo de ataque genuinamente nuevo, los clasificadores
  colapsan (memorizaron patrones de la distribución de entrenamiento) mientras los LLMs sin fine-tuning
  razonan sobre la *intención* y aguantan mucho mejor. Cada enfoque gana en un régimen distinto.
- La sub-hipótesis (*más grande = mejor generalización*) es **falsa aquí.** Pagar 10× el costo de
  entrenamiento y ~9× la latencia por Qwen compró la peor generalización de los tres. La escala por sí
  sola no ayudó.
- **Aclaración honesta:** `unseen_attacks` es una sola fuente apartada (HackAPrompt), así que mide
  generalización a *ese* estilo, no en abstracto. La brecha in-distribution → unseen es el hallazgo; los
  números absolutos de unseen son direccionales.

No es un experimento fallido — es una respuesta más útil que la hipótesis. Los dos enfoques son
**complementarios**, lo que apunta directo a una arquitectura en **cascada**.

## Conclusiones

1. **Para ataques conocidos, despliega el clasificador especializado.** La elección es **DeBERTa-v3**:
   mejor detección in-distribution (0.946), mejor macro-F1, pasa todos los gates in-distribution, y 5×
   más rápido que Qwen a una fracción del costo.
2. **Ningún modelo es robusto a ataques nuevos por sí solo** — el mejor (Sonnet 4.6, 0.662) aún deja
   pasar ~34%.
3. **Más grande no es automáticamente mejor.** El modelo de 1.5B fue el más caro *y* el peor
   generalizador.
4. **El diseño que se deriva de los datos es una cascada:** el clasificador rápido/barato maneja el
   grueso del tráfico conocido, con un LLM genérico de fallback para los casos inciertos / de aspecto
   novedoso donde el clasificador es débil.
