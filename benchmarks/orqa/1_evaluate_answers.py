#!/usr/bin/env python3
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL file for matching answers"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file"
    )
    args = parser.parse_args()

    total = 0
    matches = 0

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                # Parse JSON line
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                    continue

                target = data["TARGET_ANSWER"]
                llm_str = data["LLM_ANSWER"]

                # Convert LLM_ANSWER string to integer
                try:
                    llm = int(llm_str)
                except (ValueError, TypeError):
                    llm = None

                total += 1
                if llm == target:
                    matches += 1

    except FileNotFoundError:
        print(f"File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if total == 0:
        print("No valid entries found.")
    else:
        percentage = (matches / total) * 100
        print(f"Matches: {matches}/{total} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
