"""Small regression check on examples supplied by the project owner (never train data)."""

from __future__ import annotations

import argparse
import json

from classifier import Classifier

CASES = [
    ("Пошел нахуй ебаный петух.", "VIOLATION"),
    ("Свн шлюхи.", "VIOLATION"),
    ("Свн шьюхи.", "VIOLATION"),
    ("Спасибо за помощь, хорошего дня!", "SAFE"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    args = parser.parse_args()
    classifier = Classifier(args.model_dir)
    results = []
    for text, expected in CASES:
        actual = classifier.classify(text)
        passed = actual == expected
        results.append({"text": text, "expected": expected, "actual": actual, "passed": passed})
        print(f"{'PASS' if passed else 'CHECK'}: {text!r}: expected {expected}, got {actual}")
    report_path = f"{args.model_dir}/smoke_test.json"
    with open(report_path, "w", encoding="utf-8") as report:
        json.dump(results, report, ensure_ascii=False, indent=2)
        report.write("\n")
    if not all(result["passed"] for result in results):
        print("Some owner-provided regression phrases did not match; inspect smoke_test.json.")


if __name__ == "__main__":
    main()
