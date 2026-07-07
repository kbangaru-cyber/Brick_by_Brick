#!/usr/bin/env python3
"""
inspect_stage6.py — inspect stage6 full + noreason outputs
"""
import json, sys, re
from pathlib import Path
from collections import defaultdict, Counter

def main():
    if len(sys.argv) < 2:
        print("Usage: inspect_stage6.py <out_dir>")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    show_idx = int(sys.argv[2]) if len(sys.argv) > 2 else -1  # -1 = summary only

    full_files = sorted(out_dir.glob("stage6_full.shard*.jsonl"))
    nr_files = sorted(out_dir.glob("stage6_noreason.shard*.jsonl"))

    if not full_files:
        # Maybe not sharded
        if (out_dir / "stage6_full.jsonl").exists():
            full_files = [out_dir / "stage6_full.jsonl"]
            nr_files = [out_dir / "stage6_noreason.jsonl"]
        else:
            print(f"No stage6_full files found in {out_dir}")
            sys.exit(1)

    full_samples, nr_samples = [], []
    for f in full_files:
        with open(f) as fh:
            for line in fh:
                full_samples.append(json.loads(line))
    for f in nr_files:
        with open(f) as fh:
            for line in fh:
                nr_samples.append(json.loads(line))

    print(f"Full samples:     {len(full_samples)}")
    print(f"Noreason samples: {len(nr_samples)}")
    print(f"Unique structures: {len(set(s['metadata']['struct_hash'] for s in full_samples))}")
    print()

    # Per-tier breakdown
    tier_counts = Counter(s["metadata"]["tier"] for s in full_samples)
    print("Per-tier counts (full):")
    for t in ["T1", "T2", "T3", "T4"]:
        print(f"  {t}: {tier_counts.get(t, 0)}")

    # Token length stats per tier
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            "unsloth/Qwen2.5-32B-Instruct",
            cache_dir="/ocean/projects/cis260075p/bangarug/brickagent/hf-cache/hub",
            trust_remote_code=True,
        )
        print("\nToken length by tier (full variant):")
        by_tier = defaultdict(list)
        for s in full_samples[:100]:  # sample 100 for speed
            text = tok.apply_chat_template(s["messages"], tokenize=False)
            n = len(tok(text)["input_ids"])
            by_tier[s["metadata"]["tier"]].append(n)
        for t in ["T1", "T2", "T3", "T4"]:
            if by_tier[t]:
                mn, mx = min(by_tier[t]), max(by_tier[t])
                avg = sum(by_tier[t]) // len(by_tier[t])
                print(f"  {t}: avg={avg} min={mn} max={mx} (n={len(by_tier[t])})")
    except Exception as e:
        print(f"(token check skipped: {e})")

    # Leak checks
    print("\n=== T2 aesthetic leaks (numbers/count words) ===")
    leak_re = re.compile(r'\b(brick|layer|z=\d)\b', re.IGNORECASE)
    t2_leaks = 0
    for s in full_samples:
        if s["metadata"]["tier"] != "T2": continue
        prompt = s["messages"][1]["content"]
        if leak_re.search(prompt):
            t2_leaks += 1
    print(f"  T2 prompt leaks: {t2_leaks}/{tier_counts['T2']}")

    # Check per-tier reasoning depth
    print("\n=== Think block length by tier ===")
    think_re = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    for t in ["T1", "T2", "T3", "T4"]:
        samples_t = [s for s in full_samples if s["metadata"]["tier"] == t]
        if not samples_t: continue
        lens = []
        for s in samples_t[:20]:
            m = think_re.search(s["messages"][2]["content"])
            if m: lens.append(len(m.group(1)))
        if lens:
            avg = sum(lens) // len(lens)
            print(f"  {t}: avg think chars = {avg} (n={len(lens)})")

    # Show a specific sample
    if show_idx >= 0 and show_idx < len(full_samples):
        s = full_samples[show_idx]
        print(f"\n{'='*72}\nSAMPLE {show_idx} — {s['metadata']['tier']} {s['metadata']['variant']}")
        print(f"Object: {s['metadata']['object_class']}  Bricks: {s['metadata']['brick_count']}")
        print(f"{'='*72}")
        print(f"\n--- USER ---\n{s['messages'][1]['content']}")
        print(f"\n--- ASSISTANT ---\n{s['messages'][2]['content']}")

if __name__ == "__main__":
    main()
