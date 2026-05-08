<div align="center">

# Popularity-Aware Machine Unlearning in Large Language Models

**A research narrative: from a hypothesis about fact popularity to a benchmark that tests it — to an algorithm designed from its insights**

</div>

---

## The Hypothesis

LLM unlearning methods treat every fact in the forget set as equally hard to remove. But this assumption is almost certainly wrong.

Facts encountered more frequently during pretraining are encoded more redundantly across layers and attention heads. A model doesn't store "the capital of Poland is Warsaw" in one place — it reinforces that association across millions of training tokens, each nudging the same parameters. A rare entity like the founder of an obscure brand appears far fewer times and leaves a narrower parametric footprint. If memorisation depth scales with training-corpus frequency, then applying the same gradient force to every forget-set example will systematically **under-erase popular facts and over-erase rare ones**.

Existing methods calibrate the unlearning signal from the model's current output distribution — log-probabilities, confidence scores. But confidence is a local, surface-level signal that changes throughout training. It does not reflect how deeply a fact is encoded across the model's parameters. A model can assign high confidence to a surface pattern even when the corresponding fact is not deeply embedded, and vice versa.

This motivates a concrete research question:

> *Does fact popularity — measured externally and independently of the model — predict unlearning difficulty? And if so, how?*

---

## The Benchmark: DUET

To test the hypothesis, we need a benchmark that explicitly controls for popularity and enables controlled comparison. No existing benchmark does this. TOFU uses synthetic data with narrow, artificial distributions. RWKU and WMDP do not stratify by fact prevalence. So we build one.

**DUET** (*Dual Unlearning Evaluation across Training Stages*) is constructed from 57k Wikidata-derived factual QA pairs spanning 25 semantic topics. For each fact, we compute a **popularity score** (`pop_sum`) as the sum of Wikipedia sitelinks of its subject and object entities — a stable, corpus-agnostic proxy for pretraining frequency that correlates with model-perceived salience at >80%.

<div align="center">

| Property | Value |
|----------|-------|
| Total QA pairs | 28,600 |
| Topics | 25 (city-dominant) |
| Popularity range (city domain) | 44 – 3,763 (median ≈ 1,090) |
| Forget splits | Rare 1 / 5 / 10%, Popular 1 / 5 / 10% |
| Fast retain sets | 500 / 1,500 samples |
| Validation | BERT cosine sim > 0.6 with LLaMA-3.1-8B |

</div>

DUET also adds a second axis: the **training paradigm**. Most unlearning studies either work on pretrained checkpoints or SFT checkpoints, rarely contrasting the two. DUET enables controlled comparison of both, using the same architecture and the same forget set.

The dataset is publicly available: `SwetieePawsss/DUET`. Code: `github.com/Anya-wUw/DUET`.

---

## What the Benchmark Reveals

We run three standard algorithms — Gradient Ascent (GA), Gradient Difference (GD), and Negative Preference Optimization (NPO) — across a grid of learning rates and both model types. The results confirm the hypothesis and reveal a second, unexpected pattern.

### Insight 1: Popular facts resist unlearning — and pretrained models resist it doubly

For rare facts, both pretrained and SFT models behave conventionally: ROUGE-L on the forget set decays as learning rate increases. For popular facts, the pretrained model does something counter-intuitive: ROUGE-L on the forget set **increases** across all tested learning rates and epochs. The model treats the gradient ascent signal as additional fine-tuning on familiar knowledge.

The SFT model behaves as expected on popular facts: forgetting progresses, though it requires stronger signals and causes earlier retention collapse.

<div align="center">
<p><em>Figure: Unlearning landscape across fact popularity and model training type. The top-right quadrant (popular facts, pretrained models) is the hardest regime and the least studied.</em></p>
</div>

> **Takeaway 1:** For popular facts, a preliminary SFT step is a prerequisite for any meaningful unlearning to occur.

### Insight 2: Popularity determines retention risk, not just forgetting speed

On the retain set, the asymmetry inverts. Pretrained models are relatively stable on retain regardless of which facts are forgotten. SFT models are more robust overall — roughly half the catastrophic forgetting risk — but specifically *when forgetting rare facts*. Unlearning popular facts with an SFT model causes the largest and earliest retention collapses.

> **Takeaway 2:** SFT models are more robust on retain for rare-fact removal but fragile when erasing popular facts.

### Insight 3: The asymmetry is parametric, not surface-level

Token-level and hidden-state analyses confirm the hypothesis at the representation level. Unlearning rare facts induces large rank shifts (∆rank ≈ +220 in pretrained LLaMA under GA) while changing log-probabilities modestly. Unlearning popular facts produces large log-probability shifts but barely moves the token's rank in the distribution.

<div align="center">

| Algo | Phase | Popularity | ∆ log P | rank_base | rank_unl | ∆rank |
|------|-------|-----------|---------|-----------|----------|-------|
| GA | pretrain | popular | 3.88 | 34.2 | 41.8 | +7.5 |
| GA | pretrain | rare | 2.07 | 8.8 | 230.4 | +221.6 |
| GA | SFT | popular | 10.94 | 59.9 | 55.2 | −4.6 |
| GA | SFT | rare | 2.72 | 111.3 | 49.7 | −61.6 |

*Token-level diagnostics (DUET, city domain, N=5%, lr=2×10⁻⁵).*

</div>

For popular facts: gradient ascent moves probability mass around but doesn't disrupt the underlying factual encoding. The fact remains extractable under paraphrase or adversarial queries. This is not forgetting — it is surface suppression.

Hidden-state similarity tells the same story: SFT models show concentrated, selective representation changes for the targeted fact type, while pretrained models show limited internal adaptation regardless of which facts are unlearned.

### Insight 4: General capability is not the bottleneck

Worst-case MMLU and HellaSwag deviation stays within 3% across all tested configurations and algorithms. The challenge in LLM unlearning is not avoiding collateral capability damage — it is achieving deep parametric erasure of specific facts in the first place.

---

## The Design Conclusions

DUET's findings translate directly into design requirements for a better unlearning algorithm:

1. **The unlearning signal must be calibrated to memorisation depth, not to model confidence.** Confidence is local and changes throughout training. Popularity is global, fixed, and correlates with how broadly a fact is encoded.

2. **Popular and rare facts need different gradient regimes.** For rare facts, concentrate gradient on high-confidence tokens (self-limiting, prevents over-erasure). For popular facts, distribute gradient broadly across all answer tokens (overcomes wide parametric encoding).

3. **The forget-retain tradeoff should be controlled automatically.** A fixed retain coefficient requires dataset-specific grid search. As forget pressure increases (especially for popular facts), the retain penalty must adapt.

4. **Paraphrase generalisation is the test of genuine erasure.** An algorithm that passes surface ROUGE-L but fails on paraphrased queries has not truly forgotten anything.

---

## The Algorithm: AdaPop

AdaPop (*Adaptive Popularity*) is designed from these four requirements.

<div align="center">
  <img src="img/crop_adapop_page-0001.jpg" width="80%" alt="AdaPop overview"/>
  <p><em>Figure: Overview of AdaPop. The popularity exponent β<sub>i</sub> reweights per-fact token weights (requirement 1 & 2); the dual-ascent controller updates α each epoch to keep retain loss within tolerance ε (requirement 3).</em></p>
</div>

### Component 1 — Popularity-Dependent Exponent

Each fact is assigned an exponent β_i derived from its external popularity score s_i via a power-law mapping that mirrors how parametric memorisation scales with corpus frequency:

```
β_i = clip(a · s_i^{-b},  β_min, β_max)
```

The sign of β_i − 1 determines the qualitative gradient regime:
- **β_i > 1** (rare facts): effective per-token gradient weight *decreases* as the token becomes less confident. The update self-limits once the fact is erased — prevents over-erasure.
- **β_i < 1** (popular facts): effective weight *increases* on tokens that resist erasure. Sustains gradient pressure throughout training, covering the broad parametric footprint.
- **β_i = 1**: equivalent to standard uniform gradient ascent.

The coefficients are derived analytically from two anchor constraints on the DUET popularity distribution (s_r ≈ 100, s_p ≈ 3000), yielding **a = 58.7, b = 0.796**, clipped to [0.05, 2.0].

```
w_{i,t} = stopgrad(p_{i,t})^{β_i}
L_f = -(1/|Ω_F|) Σ w_{i,t} · NLL_{i,t}
```

### Component 2 — Dual-Ascent Retain Controller

The retain coefficient α is treated as a Lagrange multiplier enforcing the constraint that retain-loss drift δ_k stays below tolerance ε:

```
λ_{k+1} = proj_{[0, λ_max]}(λ_k + η_λ · (δ_k − ε))
α_{k+1} = α_0 + λ_{k+1}
```

Updates run at epoch granularity (not per-step) to reduce oscillation from noisy batch-level retain losses. The controller uses the popularity-weighted forget loss as its feedback signal — aligning the feedback with the actual gradient mass distribution. No dataset-specific hyperparameter search required.

**Default hyperparameters:** α_0 = 0.5, ε = 0.1, η_λ = 0.1, λ_max = 5.0.

### Why Both Components Are Necessary

The ablation isolates the two failure modes independently:

| Config | Popular paraphrase ROUGE-L ↓ | Retain ROUGE-L ↑ |
|--------|------------------------------|------------------|
| α=fixed, β=fixed (WGA-like) | 0.194 (Qwen) | collapses at moderate LR |
| α=fixed, β=popularity | **highest of all** — worst config | unstable |
| α=controller, β=fixed | moderate | ≥ 0.927 |
| **α=controller, β=popularity (AdaPop)** | **0.055 (Qwen)** | **≥ 0.724** |

The popularity exponent with a fixed α is the *worst* configuration: for popular facts, β_i < 1 distributes gradient mass broadly, producing weak per-token signals that cannot overcome memorisation depth without the controller allowing the forget signal to strengthen. The two components are complementary — the exponent calibrates the direction of the gradient; the controller determines how aggressively the forget signal acts.

---

## Results: Does It Work?

All experiments use LoRA fine-tuning (rank 32, α 64, batch size 1, 32 gradient accumulation steps) at lr = 10⁻⁴, three model families (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-7B-it), two benchmarks (DUET and RWKU).

### Surface Forgetting and Retention

<div align="center">
  <img src="img/pop_sum_diversity_comparison.png" width="72%" alt="pop_sum distributions"/>
  <p><em>Figure: Wikidata pop_sum distributions for DUET and RWKU. DUET spans nearly two orders of magnitude — the direct test of popularity-sensitive methods.</em></p>
</div>

**Llama-3.1-8B-Instruct**

| Benchmark | Algorithm | ROUGE F↓ | ROUGE R↑ | Cos Sim F↓ | Cos Sim R↑ |
|-----------|-----------|----------|----------|------------|------------|
| DUET | **AdaPop** | **0.043** ±.007 | 0.959 ±.019 | **0.369** | 0.971 |
| DUET | WGA | 0.036 ±.002 | **0.995** ±.003 | 0.442 | **0.996** |
| DUET | NPO | 0.670 ±.095 | **0.996** ±.005 | 0.684 | **0.998** |
| RWKU | **AdaPop** | **0.078** ±.050 | 0.972 ±.008 | **0.125** | 0.977 |
| RWKU | WGA | 0.095 ±.009 | **0.977** ±.003 | 0.247 | **0.984** |
| RWKU | NPO | 0.540 ±.034 | 0.957 ±.005 | 0.403 | 0.967 |

On surface ROUGE-L, WGA and AdaPop are competitive on DUET. The gap opens on cosine similarity — a signal of representation-level change.

### The Real Test: Paraphrase Generalisation

This is requirement 4 from the design conclusions. If forgetting is genuine, ROUGE-L should stay low when the same question is paraphrased. If forgetting is surface suppression, paraphrase ROUGE-L will jump back up — especially for popular facts.

| | Llama Pop | Llama Rare | Qwen Pop | Qwen Rare | Gemma Pop | Gemma Rare |
|---|---|---|---|---|---|---|
| **AdaPop** | **0.040** | 0.049 | **0.055** | 0.092 | **0.028** | 0.026 |
| WGA | 0.067 | **0.017** | 0.194 | **0.014** | 0.096 | **0.003** |
| NPO | 0.854 | 0.538 | 0.778 | 0.432 | 0.796 | 0.339 |

WGA's tier asymmetry is exactly what DUET predicted: near-perfect erasure on rare paraphrases (0.003–0.017) but persistent recall on popular paraphrases (0.067–0.194). Confidence-weighted ascent efficiently disrupts the narrow encoding of rare facts but cannot cover the broad parametric footprint of popular ones. AdaPop's popularity-dependent β closes this gap.

**RWKU Level-3 adversarial robustness** (nine prompt manipulation strategies — prefix injection, role-playing, cross-lingual reformulations):

| | Llama | Qwen | Gemma |
|---|---|---|---|
| **AdaPop** | **0.262** | **0.144** | **0.195** |
| WGA | 0.396 | 0.211 | 0.239 |
| NPO | 0.657 | 0.400 | 0.485 |

### Internal Metrics: Confirming Parametric Erasure

The ∆Rank metric is the most diagnostic. A large positive ∆Rank means the gold answer token falls far down the model's ranked distribution — genuine parametric suppression. WGA's ∆Rank on DUET is **−3,714**: the gold token actually moves *up* in rank after unlearning, even as its absolute log-probability falls. This is the same surface-suppression pattern identified in DUET's token-level analysis. AdaPop's ∆Rank is **+53,760**.

| | ∆LP F↓ | ∆Rank F↑ | Hid.Cos F↓ | KL F↑ |
|---|---|---|---|---|
| **AdaPop** (DUET) | **−160** | **+53,760** | **0.702** | **42.5** |
| NPO (DUET) | −42 | +8,824 | 0.861 | 5.4 |
| WGA (DUET) | −52 | −3,714 | 0.729 | 16.2 |

*Averaged across Llama, Qwen, Gemma at lr=10⁻⁴.*

<div align="center">
  <img src="img/internal_metrics.jpg" width="90%" alt="Internal representation metrics"/>
  <p><em>Figure: Internal representation metrics across models and benchmarks. AdaPop (purple) consistently achieves the strongest forget-split shifts while keeping retain-split metrics near the base model.</em></p>
</div>

### Learning-Rate Stability

<div align="center">
  <img src="img/merged_metrics/duet_merged/duet_merged_Llama-3.1-8B-Instruct.png" width="100%" alt="ROUGE-L across learning rates, DUET, Llama"/>
  <p><em>Figure: ROUGE-L Recall and Cosine Similarity on DUET across five learning rates, Llama-3.1-8B-Instruct. AdaPop (purple) occupies a better forget–retain frontier across the full sweep — unlike WGA, which requires per-model grid search, and NPO, which barely responds to learning rate on DUET at all.</em></p>
</div>

AdaPop is the only method for which forget quality improves monotonically with learning rate while retain quality remains controlled. The dual-ascent controller compensates automatically — no grid search, no early stopping heuristics.

### General Capability

| Algorithm | Llama MMLU | Llama HS | Qwen MMLU | Qwen HS | Gemma MMLU | Gemma HS |
|-----------|-----------|---------|---------|--------|----------|--------|
| Orig | 0.65 | 0.73 | 0.71 | 0.68 | 0.47 | 0.64 |
| **AdaPop** | 0.65 | 0.76 | 0.71 | 0.68 | 0.52 | 0.62 |
| WGA | 0.66 | 0.76 | 0.71 | 0.70 | 0.52 | 0.68 |
| NPO | 0.65 | 0.72 | 0.70 | 0.68 | 0.51 | 0.63 |
| GA | ~0.23 | ~0.33 | ~0.23 | ~0.43 | ~0.25 | ~0.26 |

Consistent with DUET's finding: the bottleneck is deep erasure, not capability preservation. AdaPop, WGA, and NPO all remain within 0.05 MMLU of the base checkpoint.

---

## Summary of the Research Chain

```
Hypothesis
  "Popular facts are harder to forget because memorisation depth
   scales with pretraining frequency — existing methods don't account for this"
         ↓
DUET Benchmark
  Confirms: popular facts resist all gradient-based unlearning methods
  Reveals: pretrained models cannot unlearn popular facts at all;
           SFT models can, but at the cost of earlier retention collapse
  Diagnoses: the failure is parametric (∆Rank, hidden-state similarity),
             not a surface measurement artefact
         ↓
Design Requirements
  1. Calibrate gradient to external popularity, not model confidence
  2. Different regimes for rare (self-limiting) and popular (pressure-sustaining) facts
  3. Automated forget-retain balance — no grid search
  4. Paraphrase generalisation as the measure of genuine erasure
         ↓
AdaPop Algorithm
  Popularity-dependent exponent β_i: closes the popular/rare gap
  Dual-ascent retain controller: eliminates dataset-specific hyperparameter search
  Result: deeper parametric erasure (∆Rank +53,760 vs WGA −3,714),
          paraphrase ROUGE-L ≤ 0.060 on popular facts (vs WGA 0.067–0.194),
          monotonically stable across learning rates
```

---

## Datasets

| Resource | Link | Description |
|----------|------|-------------|
| DUET dataset | `SwetieePawsss/DUET` | 28.6k QA pairs with pop_sum annotations |
| RWKU + pop_sum | `SwetieePawsss/exp_r` | RWKU with Wikidata citelink scores |

---

## Citation

**DUET:**

```bibtex
@misc{borisiuk2026anatomyunlearningdualimpact,
      title={Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning}, 
      author={Anna Borisiuk and Andrey Savchenko and Alexander Panchenko and Elena Tutubalina},
      year={2026},
      eprint={2602.19612},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.19612}, 
}
```

**AdaPop:** preprint in preparation.

---

*MIT License.*
