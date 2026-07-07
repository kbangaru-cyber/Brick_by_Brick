import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/stage6_test_full.jsonl"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

with open(path) as f:
    for i, line in enumerate(f):
        if i == idx:
            d = json.loads(line)
            break

print("=" * 80)
print(f"SAMPLE {idx} — full content")
print("=" * 80)

print("\n\n### SYSTEM PROMPT ###\n")
print(d["messages"][0]["content"])

print("\n\n### USER PROMPT ###\n")
print(d["messages"][1]["content"])

print("\n\n### ASSISTANT RESPONSE ###\n")
print(d["messages"][2]["content"])

print("\n\n### METADATA ###\n")
print(json.dumps(d["metadata"], indent=2))
