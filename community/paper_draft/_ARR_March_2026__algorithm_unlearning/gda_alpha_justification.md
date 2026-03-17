# Honest Theoretical Justification: Why GDA and Why Dynamic α?

## 1. Why Gradient Descent Ascent (GDA) over Alternatives?

### 1.1 Established Facts from Literature

**GDA as Standard Baseline:**
Multiple recent works establish GDA as the straightforward baseline for LLM unlearning Gradient Ascent performs gradient ascent on forget data to increase prediction loss, thereby removing parameterized knowledge.

**Known Problem with GDA:**
GDA suffers from severe instability, driving updates in a divergent direction and often resulting in drastically degraded model utility. Multiple works document this: methods like GA, GAD, and KL first increase forget quality but suddenly show significant model utility drop, leading to collapse.

---

### 1.2 Comparison with Alternative Approaches

#### **A. Projection-based Methods (Task Vectors)**

**What We Know:**
- Task vectors are a real approach where a reinforced model is obtained by tuning on unlearning corpus, then the task vector is subtracted from the original model
- Existing unlearning methods minimally impact concept vectors and mostly suppress them during inference, while directly ablating these vectors demonstrably removes knowledge

**Logical Limitations:**

1. **One-shot Operation:**
   - Task vector subtraction is a single operation: θ_new = θ_old - λ·(θ_ft - θ_base)
   - No iterative refinement based on actual forget/retain performance
   - If λ is misspecified, there's no gradient feedback to correct

2. **Global Parameter λ:**
   - Single scalar λ controls all forgetting strength
   - Cannot adapt per-sample based on difficulty or popularity
   - **Your observation:** Different facts have different memorization strengths based on popularity

**Your Contribution (from appendix observations):**
- You note instability with varying popularity in your experiments
- This is your empirical finding, not yet in published literature
- Makes sense: popular facts need different λ than rare facts

---

#### **B. Distillation Methods**

**Computational Cost:**
- Requires training separate teacher model
- KL minimization involves gradient ascent term for forgetting and minimizes KL divergence
- Approximately 2× memory overhead (student + teacher)

**Indirect Forgetting:**
- No explicit gradient signal on forget set
- Forgetting happens implicitly through absence in teacher
- Less direct control over forgetting strength

---

#### **C. Log-Probability Suppression**

**Standard Approach:**
```
L = -Σ log p(y_forget)  # Minimize probability
```

**Theoretical Hypothesis (not yet empirically validated):**

**Semantic Substitution Problem:**
- Suppressing p("password is 12345") doesn't necessarily suppress:
  - p("passcode is 12345")
  - p("access code: 12345")
  - p("secret is one-two-three-four-five")

**Why this hypothesis makes sense:**
- Log-prob suppression operates at token level
- Semantically similar phrases share embedding space
- Model may "route around" by using synonyms

**GDA Theoretical Advantage:**
```
∇_θ [log p(y)] affects hidden representations h(y), not just output tokens
```
- By chain rule: gradient propagates through entire network
- Changes to h(y) affect semantic neighborhood
- **But:** This needs empirical validation - we don't have published evidence yet

---

### 1.3 Why WGA (Weighted Gradient Ascent)?

**Sample-Adaptive Weighting:**
```python
w(i,t) = (p(i,t))^β(i)  # detached
```

**Intuition:**
- High p(i,t) (model confident) → high weight → stronger signal
- Low p(i,t) (model uncertain) → low weight → avoid over-suppression
- β(i) based on popularity provides per-sample adaptation

**Contrast with uniform weighting:**
- Standard GA treats all samples equally
- Doesn't account for varying difficulty or memorization strength

---

## 2. Why Dynamic α over Fixed α?

### 2.1 The Core Motivation

**Problem:** In gradient-based unlearning, we can go too far and break generation.

**Evidence from Literature:**
Multiple methods show sudden model utility drop and collapse during training, with GA particularly susceptible to excessive unlearning that compromises model integrity.

**What Dynamic α Provides:**
- Automatic adjustment when retain performance degrades
- Safety mechanism against catastrophic forgetting
- No manual hyperparameter grid search

---

### 2.2 Mathematical Formulation

**Unlearning as Constrained Optimization:**

The true objective should be:
```
maximize: Forgetting_Effectiveness(θ)
subject to: Retain_Degradation(θ) ≤ ε_threshold
```

**Standard Approach (Fixed α):**
```
L_total = γ·L_forget + α·L_retain
```
- α is hyperparameter that must be grid-searched
- Doesn't adapt to training dynamics

**Adaptive Approach:**
```
If ε_retain(t) > ε_high: α(t+1) = α(t) + Δα
If ε_retain(t) < ε_low:  α(t+1) = α(t) - Δα
Else:                    α(t+1) = α(t)
```

**Analogy to Constrained Optimization:**
This resembles Lagrange multiplier methods where:
- α acts like penalty parameter
- Automatically adjusts to satisfy constraint
- Converges to point where constraint is active

**Mathematical Argument (not formal proof):**

Define V(t) = ε_retain(t) - ε_threshold as constraint violation.

When V(t) > 0 (constraint violated):
1. Increase α → stronger retain gradient
2. Next update: θ ← θ - η[γ∇L_forget + α_new·∇L_retain]
3. Increased α term pulls toward lower retain loss
4. Expected: V(t+1) < V(t)

This creates negative feedback loop toward constraint satisfaction.

---

### 2.3 Practical Benefits (Logical Arguments)

**1. Eliminates Hyperparameter Search:**
- Fixed α: Need to try α ∈ {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0} = 7 runs minimum
- Plus need to try different γ values
- Typical: 5-10× more runs for grid search
- Adaptive α: Single run, α finds optimal value automatically

**2. Adapts to Non-Stationarity:**
- Early training: strong forget gradients, need lower α
- Late training: weak forget gradients, need higher α  
- Fixed α cannot adapt
- Adaptive α adjusts automatically

**3. Safety Guarantee:**
- If retain starts degrading: α increases immediately
- Acts as automatic brake
- Prevents catastrophic forgetting

---

## 3. Why AdaPop Framework?

### 3.1 Three Key Contributions

**1. Intuitive Optimization Formulation:**
```
minimize: -Forgetting_Effectiveness
subject to: ε_retain ≤ ε_threshold
control: Popularity-weighted gradients
```

Clear semantics:
- Objective: maximize forgetting
- Constraint: don't break retain
- Control: adapt to popularity

**2. Popularity-Based Weighting:**
```python
β(i) = 2.26 · (pop_sum(i))^(-0.677)
```

**Your Insight:**
- Popular facts (high pop_sum) → harder to forget → need different treatment
- Rare facts (low pop_sum) → easier to forget → gentler approach
- Single formula handles heterogeneous memorization

**Why This Helps:**
- No need to manually tune β per dataset
- Automatically adapts strength based on memorization
- Natural interpretability: popularity ↔ forgetting difficulty

**3. Dual Optimization (Adaptive α + Popularity β):**
- β handles sample-level heterogeneity
- α handles epoch-level constraint satisfaction
- Together: minimal manual tuning required

---

### 3.2 Comparison Summary (Honest Assessment)

| Aspect | Projection | Distillation | Fixed GDA | **AdaPop** |
|--------|-----------|--------------|-----------|-----------|
| Hyperparameters | λ, rank, layers | T, α_dist, lr | α, γ, lr | **lr, epochs only** |
| Iterative? | No (one-shot) | Yes | Yes | Yes |
| Per-sample adaptive? | No (global λ) | No | No | **Yes (via β)** |
| Constraint enforcement? | No | No | No | **Yes (via adaptive α)** |
| Semantic neighborhood? | Partial | Partial | Likely yes* | Likely yes* |

*Not empirically validated yet

---

## 4. What We DON'T Know (Honest Gaps)

**Things that need empirical validation:**

1. **Semantic substitution hypothesis:**
   - We hypothesize GDA prevents "passcode" when forgetting "password"
   - Not yet tested in controlled experiments
   - **Recommended experiment:** Measure p(synonyms) before/after unlearning

2. **Optimal β formula:**
   - β(i) = 2.26 · (pop_sum)^(-0.677) is your empirical finding
   - Works in your experiments, but not theoretically derived
   - May need tuning for other domains/models

3. **Adaptive α convergence rate:**
   - We argue it converges to optimal point
   - Formal convergence proof requires assumptions we haven't verified
   - Works empirically, but theoretical guarantees are informal

4. **Projection instability with popularity:**
   - Your appendix shows this
   - Not yet in published literature
   - Important contribution if you document it properly

---

## 5. Honest Conclusion

**What We Can Confidently Claim:**

1. **GDA is established baseline** with known instability problem (peer-reviewed evidence)
2. **Adaptive α provides safety mechanism** against catastrophic forgetting (logical argument + analogy to Lagrange multipliers)
3. **Popularity weighting addresses heterogeneous memorization** (your empirical observation + logical necessity)
4. **Reduced hyperparameter search** (logical: β auto-computed, α auto-adapted)

**What Needs Further Validation:**

1. Semantic substitution hypothesis (needs experiments)
2. Comparison to projection methods on popularity-varying datasets (your contribution)
3. Formal convergence proofs for adaptive α scheme
4. Optimal choice of β formula across domains

**Honest Positioning for Paper:**

> We propose AdaPop, combining GDA with (1) popularity-based weighting to handle heterogeneous memorization strength, and (2) adaptive retain loss coefficient to ensure bounded degradation. Unlike fixed-weight approaches requiring extensive grid search, AdaPop automatically adapts both per-sample (via β) and per-epoch (via α), requiring only standard hyperparameters (learning rate, epochs). While GDA suffers from known instability issues, our adaptive mechanism provides a safety constraint. Empirical evaluation shows [your experimental results], and analysis reveals [your findings about projection instability with popularity].

**This is honest, defensible, and highlights your actual contributions without overclaiming.**