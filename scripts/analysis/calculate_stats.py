from datasets import load_dataset
import numpy as np

def get_pop_sums(dataset_name, subset=None, split=None):
    ds = load_dataset(dataset_name, subset, split=split)
    return [float(x['pop_sum']) for x in ds if x['pop_sum'] is not None]

# DUET
duet_rare = get_pop_sums("SwetieePawsss/DUET", split="city_forget_rare_10")
duet_popular = get_pop_sums("SwetieePawsss/DUET", split="city_forget_popular_10")
duet_merged = duet_rare + duet_popular

# RWKU
rwku = get_pop_sums("SwetieePawsss/exp_r", "forget_level2", split="test")

def print_stats(name, data):
    print(f"\nStats for {name}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Min: {np.min(data):.2f}")
    print(f"  Max: {np.max(data):.2f}")
    print(f"  P10: {np.percentile(data, 10):.2f}")
    print(f"  P90: {np.percentile(data, 90):.2f}")

print_stats("DUET (Merged)", duet_merged)
print_stats("RWKU (forget_level2)", rwku)
