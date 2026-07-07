import json
import argparse
import sys

def view_jsonl_interactive(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())

                print(f"\n{'='*80}")
                print(f" SAMPLE {i + 1} ".center(80, '='))
                print(f"{'='*80}\n")

                # Print Metadata for quick context
                meta = data.get("metadata", {})
                if meta:
                    print(f"--- METADATA ---")
                    print(f"Class:      {meta.get('object_class', 'N/A')}")
                    print(f"Bricks:     {meta.get('brick_count', 'N/A')}")
                    print(f"Stability:  {meta.get('stability_pct', 'N/A')}%")
                    compliance = meta.get("compliance", {})
                    if compliance:
                        comp_str = ", ".join([f"{k}: {'✓' if v else '✗'}" for k, v in compliance.items()])
                        print(f"Compliance: {comp_str}")
                    print("-" * 80 + "\n")

                # Print the actual conversation
                for msg in data.get("messages", []):
                    role = msg.get("role", "UNKNOWN").upper()
                    content = msg.get("content", "")
                    print(f"=== {role} ===")
                    print(f"{content}\n")

                # Pause and wait for user
                print(f"{'='*80}")
                user_input = input("Press [Enter] to see the next sample, or type 'q' to quit: ").strip().lower()
                if user_input == 'q':
                    print("Exiting viewer.")
                    break

    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'. Make sure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Interactive viewer for JSONL conversation data.")
    
    # Add the file argument with a shorthand (-f) and a default fallback
    parser.add_argument(
        '--file', 
        '-f', 
        type=str, 
        default="data/stage5_v7.jsonl", 
        help="Path to the JSONL file you want to view."
    )
    
    # Parse the arguments from the terminal
    args = parser.parse_args()

    # Run the viewer with the captured file path
    view_jsonl_interactive(args.file)
