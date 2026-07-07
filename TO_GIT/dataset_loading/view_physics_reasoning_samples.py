#!/usr/bin/env python3
"""View one sample per tier from a stage6 JSONL file."""
import json, sys, argparse, textwrap

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="Path to stage6_full.shard00.jsonl or similar")
    p.add_argument("--tiers", nargs="+", default=["T1", "T2", "T3", "T4"])
    p.add_argument("--full", action="store_true", help="Print full assistant text, not truncated")
    args = p.parse_args()

    seen = {}
    with open(args.jsonl) as f:
        for line in f:
            d = json.loads(line)
            t = d["metadata"]["tier"]
            if t in args.tiers and t not in seen:
                seen[t] = d
            if len(seen) == len(args.tiers):
                break

    for tier in args.tiers:
        if tier not in seen:
            print(f"\n{'='*80}\n{tier}: NOT FOUND in file\n{'='*80}")
            continue
        d = seen[tier]
        m = d["metadata"]
        msgs = d["messages"]
        print(f"\n{'='*80}")
        print(f"  {tier}  |  object: {m['object_class']}  |  "
              f"bricks: {m['brick_count']}  layers: {m['layer_count']}  "
              f"stable: {m['stability_pct']}%")
        print(f"  hash: {m['struct_hash']}  variant: {m['variant']}")
        print('='*80)
        print(f"\n--- USER PROMPT ---\n{msgs[1]['content']}")
        print(f"\n--- ASSISTANT ---")
        body = msgs[2]['content']
        if args.full:
            print(body)
        else:
            for block in ["<think>", "<plan>", "<build>", "<review>"]:
                if block in body:
                    end = body.find(block.replace("<", "</"))
                    snippet = body[body.find(block):end + len(block)+1]
                    if len(snippet) > 600:
                        snippet = snippet[:600] + "\n  ...[truncated]..."
                    print(snippet + "\n")

if __name__ == "__main__":
    main()

