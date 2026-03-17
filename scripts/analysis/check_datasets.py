from datasets import load_dataset

try:
    print("Checking DUET...")
    duet_forget_rare = load_dataset("SwetieePawsss/DUET", split="city_forget_rare_10")
    duet_popular_rare = load_dataset("SwetieePawsss/DUET", split="city_forget_popular_10")
    
    print(f"DUET city_forget_rare_10 columns: {duet_forget_rare.column_names}")
    print(f"DUET city_forget_popular_10 columns: {duet_popular_rare.column_names}")
    
    if "pop_sum" in duet_forget_rare.column_names:
        print(f"Example pop_sum from DUET (rare): {duet_forget_rare[0]['pop_sum']}")
    if "pop_sum" in duet_popular_rare.column_names:
        print(f"Example pop_sum from DUET (popular): {duet_popular_rare[0]['pop_sum']}")
except Exception as e:
    print(f"Error loading DUET: {e}")

try:
    print("\nChecking RWKU (exp_r)...")
    # Based on config: path="SwetieePawsss/exp_r", name=${forget_split}, split="test"
    rwku = load_dataset("SwetieePawsss/exp_r", "forget_level2", split="test")
    print(f"RWKU forget_level2 columns: {rwku.column_names}")
    if "pop_sum" in rwku.column_names:
        print(f"Example pop_sum from RWKU: {rwku[0]['pop_sum']}")
except Exception as e:
    print(f"Error loading RWKU: {e}")
