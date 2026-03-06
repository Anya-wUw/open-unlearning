import json
from pathlib import Path
import pandas as pd
import re
import numpy as np

def get_data(bench):
    base_dir = Path(f"/mnt/extremessd10tb/borisiuk/new_MU_exps/open-unlearning/saves/unlearn/{bench}")
    
    def get_model(run_name):
        if "Llama-3.1-8B" in run_name: return "Llama"
        if "gemma-7b-it" in run_name: return "Gemma"
        if "Qwen2.5-7B-Instruct" in run_name: return "Qwen"
        return "unknown"

    data_map = {} # (algo, model, is_bugfix) -> data

    # 1. Delta LP/Rank/KL/HiddenCos from FORGET_METRICS_SUMMARY.json
    for p in base_dir.glob("**/FORGET_METRICS_SUMMARY.json"):
        try:
            d = json.loads(p.read_text())
            run_dir = p.parents[2]
            algo_dir = run_dir.parent
            algo_name = algo_dir.name.lower()
            is_bugfix = "NEW_BUGFIX_" in algo_name.upper()
            clean_algo = algo_name.replace("new_bugfix_", "")
            
            run_name = run_dir.name
            model = get_model(run_name)
            
            m = re.search(r"_lr([^_]+)", run_name)
            lr = m.group(1) if m else ""
            if lr != "1e-3": continue
            
            key = (clean_algo, model, is_bugfix)
            if key not in data_map: data_map[key] = {}
            
            f_split = [k for k in d.keys() if "forget" in k][0]
            r_split = [k for k in d.keys() if "retain" in k or "neighbor" in k or "holdout" in k][0]
            
            data_map[key].update({
                "f_lp": d[f_split].get("delta_logprob_mean"),
                "r_lp": d[r_split].get("delta_logprob_mean"),
                "f_rank": d[f_split].get("delta_rank_mean"),
                "r_rank": d[r_split].get("delta_rank_mean"),
                "f_kl": d[f_split].get("kl_mean"),
                "r_kl": d[r_split].get("kl_mean"),
                "f_hcos": d[f_split].get("hidden_cos_mean"),
                "r_hcos": d[r_split].get("hidden_cos_mean"),
            })
        except: continue

    final_rows = []
    for (algo, model, bugfix), vals in data_map.items():
        final_rows.append({"algo": algo, "model": model, "is_bugfix": bugfix, **vals})
    
    if not final_rows: return pd.DataFrame()
    df = pd.DataFrame(final_rows)
    df = df.sort_values("is_bugfix", ascending=False).drop_duplicates(["algo", "model"])
    return df

def format_val(val):
    if val is None or np.isnan(val): return "---"
    return f"{val:.2f}"

def gen_2x4_latex(df_duet, df_rwku, algo="ada_pop"):
    # Rows: DUET, RWKU
    # Columns: LogProb, Rank, KL, Hidden Cos
    # Cell: Llama / Qwen / Gemma (Forget split)
    
    metrics = [("lp", "$\Delta$ LP"), ("rank", "$\Delta$ Rank"), ("kl", "KL Div"), ("hcos", "Hidden Cos")]
    benches = [("DUET", df_duet), ("RWKU", df_rwku)]
    models = ["Llama", "Qwen", "Gemma"]
    
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Internal Unlearning Metrics (Forget Split) for \textbf{AdaPop} across Benchmarks and Models.}")
    print(r"\label{tab:internal_metrics}")
    print(r"\footnotesize")
    print(r"\begin{tabularx}{\linewidth}{|l|c|c|c|c|}")
    print(r"\hline")
    print(r"\textbf{Benchmark} & \textbf{$\Delta$ Log-Prob} $\downarrow$ & \textbf{$\Delta$ Rank} $\uparrow$ & \textbf{KL Div} $\uparrow$ & \textbf{Hidden Cos} $\downarrow$ \\ \hline")
    
    for b_name, df in benches:
        row_str = f"\\textbf{{{b_name}}} "
        for m_key, m_name in metrics:
            vals = []
            for model in models:
                subset = df[(df["algo"] == algo) & (df["model"] == model)]
                if not subset.empty:
                    vals.append(format_val(subset.iloc[0][f"f_{m_key}"]))
                else:
                    vals.append("---")
            row_str += f" & {' / '.join(vals)}"
        row_str += r" \\ \hline"
        print(row_str)
        
    print(r"\end{tabularx}")
    print(r"\end{table*}")

df_duet = get_data("duet")
df_rwku = get_data("rwku")

print("--- 2x4 Internal Metrics Table ---")
gen_2x4_latex(df_duet, df_rwku)
