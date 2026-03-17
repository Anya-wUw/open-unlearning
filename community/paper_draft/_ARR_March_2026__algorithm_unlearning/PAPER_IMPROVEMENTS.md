# Paper Improvement Plan — AdaPop ARR March 2026

Edit this file, then we will apply the changes section by section.
Mark items with `[DONE]`, `[SKIP]`, or add notes inline.

---

## 1. CRITICAL — Breaks compilation or is obviously wrong

- [DONE] **Typo: `\section{Concusion}`** (`acl_latex.tex` line 182): → `\section{Conclusion}`
- [DONE] **Typo: `\section{Relared Work}`** (`acl_latex.tex` line 173): → `\section{Related Work}`
- [DONE] **Title set**: `\title{The More Popular, The Harder to Forget: Adaptive Calibration for LLM Unlearning}`
- [DONE] **Duplicate citation key**: Deleted `zhangnegative` bib entry; `zhang2024negative` is canonical. All intro uses now reference `zhang2024negative`.

---

## 2. MISSING CROSS-REFERENCES — Tables and figures without `\ref{}`

- [DONE] **`tab:lm_eval` not cited in text** (`04_experiments.tex` §4.4): Added `Table~\ref{tab:lm_eval}` to §4.4 opening sentence.
- [DONE] **`fig:placeholder` not cited in intro** (`01_intro.tex`): Added `(Figure~\ref{fig:adapop_workflow})` after contributions heading.
- [DONE] **`fig:placeholder` is a placeholder label** (`01_intro.tex` line 12): Renamed `\label{fig:placeholder}` → `\label{fig:adapop_workflow}`.
- [DONE] **Appendix ref in method** (`03_method.tex` line 24): Fixed to `(see Appendix~\ref{app:derivation} for details)`.

---

## 3. WRONG CITATIONS

- [DONE] **WGA citation in §2.2 RW** (`02_rw.tex`): Fixed `\cite{yao2024large}` → `\cite{wang2025rethinking}` in this session.

---

## 4. NUMBERS — Minor inaccuracies in prose

- [DONE] **"NPO forget ROUGE-L >0.586"** (`05_conclusion.tex` line 1): Changed to `\geq 0.586`.

---

## 5. TABLE FORMATTING rules not applied

- [DONE] **`\underline{}` for best values in main tables** (Tables 1–3, 4, 5, 6): Applied `\underline{}` to best values among NPO/WGA/AdaPop in all 6 tables. `GA$^\dag$`/`GD$^\dag$` markers added. Table 1 caption has full legend; Tables 2–6 captions reference Table 1 notation.

---

## 6. CLEANUP — Commented-out and TODO text

- [DONE] **TODO comment in appendix** (`appendix.tex` line 68): Removed `%% TODO: Add examples of generation`.
- [DONE] **Old duplicate method text** (`03_method.tex` lines 69–125): Removed entirely.
- [DONE] **Old UNLamb commented-out tables** (`appendix.tex` lines 249–360): Removed entirely. UNLamb is not part of this submission.
- [DONE] **Commented-out `%\texttt{pop\_sum}` in abstract**: Resolved — the term is now used inline in the live abstract text.

---

## 7. CONTENT / LOGIC GAPS

- [DONE] **Intro motivation missing citation** (`01_intro.tex`): Added `\cite{hartmann2024undesirable, kandpal2023large}` to log-prob claim in intro. Also rewrote intro opening to establish privacy/right-to-erasure motivation.
- [DONE] **Redundancy between §4.5 and §4.6**: Trimmed §4.6 Discussion — WGA surface-forgetting section condensed to one sentence; replaced with synthesis of cross-cutting themes (stability spectrum, surface vs. depth divergence, popularity hypothesis).
- [DONE] **Gemma WGA collapse in qualitative examples**: Added one sentence in §4.5 WGA paragraph noting instance-level instability (`Lower vhs vhs vhs`) despite stable aggregate metrics.
- [ ] **Unused image**: `img/ada_pop_illustr.jpg` present but not referenced. Remove or use it; paper currently uses `ada_pop_sq_illustr.jpg`.

---

## 8. ABSTRACT REWRITE

- [DONE] **Raw numbers in abstract replaced with qualitative framing**: Removed `reduces forget ROUGE-L below 0.051 while retaining ROUGE-L above 0.855... ΔRank +53,760 vs. WGA −3,714`. Replaced with: "consistently outperforms all baselines… achieving a 3.5× reduction in popular-paraphrase forget ROUGE-L relative to WGA while preserving general capability within 0.05 MMLU… Internal representation metrics further confirm that this improvement reflects deeper parametric erasure rather than surface-level output suppression."

---

## 9. INTRO STRUCTURE AND WRITING

- [DONE] **Tech details moved out of intro, Contributions list added** (`01_intro.tex`): Removed the paragraph describing the popularity-dependent exponent and dual-ascent controller. Replaced with a three-bullet Contributions list (popularity gap failure mode; AdaPop components; empirical result summary).
- [DONE] **Motivation for unlearning is missing** (`01_intro.tex`): Rewrote opening paragraph to establish privacy/right-to-erasure motivation, popularity gap problem, and popularity-driven resistance to forgetting.
- [DONE] **Intro structure: methods before problem** (`01_intro.tex`): New order: problem (popularity gap) → why existing solutions fail → our contribution.
- [DONE] **Forget/retain tradeoff not articulated** (`01_intro.tex`): Added explicit statement that existing methods occupy suboptimal tradeoff points and require hand-tuning.

---

## 10. METHOD SECTION CLARITY (§3)

- [DONE] **Method §3 opening paragraph rewritten**: Removed the LLM-sounding paragraph. New version is direct: describes the problem (manual $\alpha$ tuning) and the two components that solve it.
- [DONE] **§3.1 Popularity Gap shortened**: Removed intro-level repetition; replaced with one focused paragraph pointing to technical content and citing `\cite{hartmann2024undesirable, kandpal2023large}`.
- [DONE] **§3.2: Why GDA? Why dynamic α?**: Added 3-sentence constrained-optimisation framing at end of §3.2, with `\label{sec:dual_ascent}` on the subsection header. Full rationale (WGA-vs-GA reasoning, dual-ascent motivation, Entesari et al. NeurIPS 2025 precedent, honest non-convex caveat) in new `app:design_motivation` appendix section.

---

## 11. LANGUAGE CLEANUP

- [DONE] **LLMs re-introduction in §2.1 removed**: `Large Language Models (LLMs)` → `LLMs` in the first sentence of Related Work §2.1.
- [DONE] **Baseline full names shortened in §4**: `Gradient Ascent (GA~\cite{...}), ...` → `GA~\cite{...}, ...` in the experiments opening sentence.
- [ ] **Boilerplate openings**: RW opens with "The necessity of..." — rephrase to avoid repetition with old intro phrasing.
- [ ] **General language pass on intro and RW**: Focused editing pass needed on these two sections.

---

## 12. LATEX STYLE FIXES

- [DONE] **Quotation marks fixed in method**: `"shield"`, `"magic"`, `"physics"`, `"flattens"`, `"sharpens"` → LaTeX-style quotes (all occurrences in `03_method.tex`).
- [DONE] **Formula numbering extended**: `w_{i,t}`, `L_f`, `\delta_k`, `scale` equations changed to `\begin{equation}...\end{equation}`.
- [ ] **Remaining straight quotes in intro**: Scan `01_intro.tex` for any remaining `"..."` instances and convert to `` ``...'' ``.

---

## 13. OPTIONAL IMPROVEMENTS (lower priority)

- [ ] **Add std to main result tables**: Once extra-seed runs (seeds 1 and 219) complete, add `mean ± std` to Tables 1–3 as footnotes or a separate table.
- [DONE] **§4.3 Robustness section forward link**: Added sentence at end of §4.3: `"A per-method analysis of these results is provided in §\ref{sec:comparison}."`.
- [ ] **Internal metrics figure caption** (`04_experiments.tex`): Caption says "Marker: ○ = DUET, □ = RWKU; colour = model family." Verify this matches the actual `.jpg`.
- [DONE] **DUET citation key** (`custom.bib`): Fixed `anna2026anatomy` author field to `Anna Borisiuk and Andrey Savchenko and Alexander Panchenko and Elena Tutubalina`.

---

## 14. NEW THIS SESSION — Appendix baseline comparison

- [DONE] **Added `\section{Comparison with UNDIAL, RMU, and PDU}`** (`appendix.tex`): Three tables (one per model: Llama, Qwen, Gemma) with ROUGE-L forget/retain for UNDIAL, RMU, PDU alongside Orig and AdaPop reference rows. Four analytical paragraphs. `\underline{}` applied per CLAUDE.md rules. Forward reference added at end of §4.5 in `04_experiments.tex`. PDU is marked `[CITE NEEDED]`.

---

## 15. FIXES APPLIED THIS SESSION (review pass)

- [DONE] **`utilize` → `use`** in `01_intro.tex` line 19, `02_rw.tex` line 7, `appendix.tex` line 62.
- [DONE] **Duplicated sentence in §4.6 Discussion** (`04_experiments.tex`): "NPO and GD represent opposite failure modes on the forget-retain spectrum." appeared twice (lines ~285 and ~287). Second instance removed; paragraph rewritten to flow cleanly.
- [DONE] **Straight quotes in appendix** (`appendix.tex` lines 19, 21): `"flatter"`, `"sharper"`, `"rare"`, `"popular"` in live (non-commented) body text → LaTeX-style quotes.

---

## 16. ISSUES FROM FULL-PAPER REVIEW (substantive, need author decisions)

### 16a. Missing ablation — critical for reviewers
- [DONE] **Component isolation added**: New appendix section `app:ablation` with `coeffitient_oblation.jpg`. Four variants: (fixed α, fixed β), (fixed α, dyn β), (dyn α, fixed β), (dyn α, dyn β=full AdaPop). Three findings: (1) controller is primary driver of retain stability; (2) popularity exponent improves forget-retain frontier; (3) fixed α + dyn β is worst, confirming the two are complementary. Brief mention added as Discussion point 5 in §4.6 (former point 5 → 6).

### 16b. Overstatements in abstract and intro
- [DONE] **"Consistently outperforms all baselines in erasure depth"** (abstract): Replaced with precise claim: "achieves the best paraphrase-generalised erasure depth among all stable methods, reducing popular-paraphrase forget ROUGE-L by 3.5× relative to WGA and lowering forget Cosine Similarity below WGA across all three model families."
- [DONE] **AdaPop retain quality vs. WGA on DUET**: Acknowledged in new §4.5 AdaPop paragraph: "WGA achieves higher retain ROUGE-L (0.993–0.997 vs. AdaPop's 0.942–0.966)."

### 16c. Level-3 RWKU: AdaPop is worse than WGA — not addressed
- [DONE] Acknowledged in §4.5 AdaPop paragraph and §4.6 Discussion point 4. Framed as an honest tradeoff: deeper popularity-weighted suppression spreads partially to adjacent semantic concepts.

### 16d. Undefined hyperparameters — reproducibility problem
- [DONE] **`gain` renamed to `g`, defined in §3.4**, with explicit note that `g=0` (modulation disabled) in all reported experiments.
- [DONE] **All controller hyperparameters added to §4.1**: $\varepsilon=0.1$, $\eta_\lambda=0.1$, $\lambda_{\max}=5.0$, $\alpha_0=0.5$, EMA decay $=0.9$, exponent coefficients $a{=}58.7$, $b{=}0.796$ clipped to $[0.05, 2.0]$.

### 16e. Coefficient sensitivity (E4) — resolved without experiment
- [DONE] **Limitations §2 rewritten**: No separate RWKU-recalibration run needed. The clip bounds $[\beta_{\min}=0.05, \beta_{\max}=2.0]$ are the effective constraints for extreme tiers — rare facts saturate at $\beta_{\max}$ and popular facts are driven toward $\beta_{\min}$ regardless of exact $a$, $b$ values, as long as the popularity gap spans ~2 orders of magnitude. Exact coefficients only govern the intermediate range. Recalibration is only necessary for proxies with a fundamentally different scale (e.g., PageRank spanning 5+ orders of magnitude).

### 16f. WGA ΔRank interpretation
- [DONE] No absolute baseline needed. The sign of ΔRank is self-interpreting: WGA $-3{,}714$ = rank improved (forgetting failed); AdaPop $+53{,}760$ = rank worsened (forgetting succeeded). Removed the speculative parenthetical about baseline rank magnitude that was added in error. §4.5 text now simply states the sign and its meaning.

### 16g. Retain improvement is an unexplained secondary finding
- [DONE] Explained in §4.6 Discussion (point 5) using existing internal metrics. Retain Hid.Cos and ΔLP for AdaPop are indistinguishable from WGA/NPO on the retain split — no special representational change. The ROUGE-L gain above Orig is explained by retain-loss minimisation during training (supervised fine-tuning on the retain QA set), not by a novel mechanism. No new experiment needed.

### 16h. Related Work structure
- [DONE] Closing differentiation sentences added to all three subsections. §2.1 rewritten to open with concrete motivation; closes with encoding-depth gap. §2.2 closes with "AdaPop replaces local signal with global popularity proxy." §2.3 closes with "AdaPop treats memorisation depth as a property of the fact, not of the model." Also removed "state-of-the-art" from §2.2.

### 16i. Commented-out developer note in intro
- [DONE] `%%` drafting comment block (lines 1–3 of `01_intro.tex`) removed.

---

## 17. FULL-PAPER SCIENTIFIC PASS

- [DONE] **Intro para 3 shortened**: Removed repetition of contributions; now 2 sentences introducing AdaPop before the contributions list.
- [DONE] **Figure 1 caption**: Replaced vague "preserve model integrity" with specific description of $\beta_i$ assignment and dual-ascent controller.
- [DONE] **Method dev comment removed**: `% take from other/ada_pop.tex` deleted.
- [DONE] **§3.1 renamed**: "Motivation: The Popularity Gap" → "The Popularity Gap"; opening "As established in the introduction" → direct claim.
- [DONE] **§3.2 renamed**: "Motivation and Setting" → "Problem Formulation".
- [DONE] **§3.3 rewritten**: Removed "intuition behind this 'physics'" framing; plain description of monotone decrease and its effect on token weighting. "typically clipped" → "clipped in all experiments."
- [DONE] **§3.3 weight description**: "governed by the detached token weight" → "The per-token unlearning weight is."
- [DONE] **§3.4 subsection "Unlearning Intuition and Implementation"**: Shortened to "Implementation Note"; cut redundant explanation.
- [DONE] **§3.4 dual-ascent opening**: Removed "magic" and "auto-balancing" informal language. Added definition of $R_{\mathrm{ref}}$ and $\epsilon_0$.
- [DONE] **§4.1**: "broad test of method robustness" → "diverse test across LLM families".
- [DONE] **§4.4 paragraph condensed**: Cut per-model number repetition (already in table); kept only the four key findings.
- [DONE] **§4.4 "de-cluttering effect"**: Removed unexplained jargon.
- [DONE] **§4.6 em-dash sentence**: "Representations are not more preserved — they are equivalently preserved" → "Representations are equivalently preserved."
- [DONE] **Appendix Gemma paragraph**: Replaced "consolidating retain representations" with consistent explanation (retain-loss minimisation) matching §4.6 revision.
