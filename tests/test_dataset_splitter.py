import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dataset_splitter import split_csv


class SplitTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / "data.csv"
        self.output = self.root / "split"
        lines = ["id,label"] + [f"{index},{'A' if index < 8 else 'B'}" for index in range(12)]
        self.source.write_text("\n".join(lines), encoding="utf-8")

    def read_partition(self, name):
        with (self.output / name).open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))

    def test_split_is_deterministic_and_each_row_is_written_once(self):
        second = self.root / "second"
        for target in (self.output, second):
            split_csv(self.source, target, Fraction(1, 4), 15)
        for filename in ("train.csv", "test.csv", "split.json"):
            self.assertEqual((self.output / filename).read_bytes(), (second / filename).read_bytes())
        train = [int(row["id"]) for row in self.read_partition("train.csv")]
        test = [int(row["id"]) for row in self.read_partition("test.csv")]
        self.assertEqual((len(train), len(test)), (9, 3))
        self.assertEqual(sorted(train + test), list(range(12)))
        self.assertEqual(train, sorted(train))
        self.assertEqual(test, sorted(test))

    def test_stratification_keeps_both_labels_and_exact_test_count(self):
        result = split_csv(self.source, self.output, Fraction(1, 4), 42, stratify="label")
        self.assertEqual(result["classes"], [
            {"label": "A", "train": 6, "test": 2}, {"label": "B", "train": 3, "test": 1},
        ])
        self.assertEqual({row["label"] for row in self.read_partition("test.csv")}, {"A", "B"})

    def test_impossible_stratification_writes_no_output(self):
        with self.assertRaisesRegex(ValueError, "test oranıyla"):
            split_csv(self.source, self.output, Fraction(1, 12), 42, stratify="label")
        self.assertFalse(self.output.exists())

    def test_invalid_input_and_existing_output_are_untouched(self):
        self.source.write_text("id,label\n1,A\n2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Satır 3"):
            split_csv(self.source, self.output, Fraction(1, 2), 42)
        self.assertFalse(self.output.exists())
        original = self.source.read_bytes()
        with self.assertRaises(FileExistsError):
            split_csv(self.source, self.source, Fraction(1, 2), 42)
        self.assertEqual(self.source.read_bytes(), original)

    def test_cli_uses_default_fraction_and_writes_metadata(self):
        script = Path(__file__).resolve().parents[1] / "dataset_splitter.py"
        run = subprocess.run(
            [sys.executable, str(script), str(self.source), str(self.output), "--seed", "7"],
            capture_output=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        metadata = json.loads((self.output / "split.json").read_text(encoding="utf-8"))
        self.assertEqual((metadata["train_rows"], metadata["test_rows"], metadata["seed"]), (10, 2, 7))
