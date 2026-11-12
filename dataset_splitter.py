"""Split a CSV into reproducible train and test partitions."""

import argparse
from collections import defaultdict
from contextlib import suppress
import csv
from fractions import Fraction
import json
from pathlib import Path
import random
import sys


def parse_ratio(text):
    try:
        ratio = Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError("Test oranı 0.25 veya 1/4 gibi bir sayı olmalı.") from error
    if not 0 < ratio < 1:
        raise argparse.ArgumentTypeError("Test oranı 0 ile 1 arasında olmalı.")
    return ratio


def parse_delimiter(text):
    value = "\t" if text == r"\t" else text
    if len(value) != 1 or value in '\r\n"\0':
        raise argparse.ArgumentTypeError("Ayraç tek bir karakter olmalı. Sekme için \\t kullan.")
    return value


def read_table(path, delimiter, encoding):
    with path.open(encoding=encoding, newline="") as source:
        reader = csv.reader(source, delimiter=delimiter, strict=True)
        try:
            header = next(reader, [])
            if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                raise ValueError("Sütun başlıkları dolu ve benzersiz olmalı.")
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Satır {reader.line_num}: {len(header)} alan bekleniyor, {len(row)} bulundu.")
                rows.append(row)
        except csv.Error as error:
            raise ValueError(f"CSV sözdizimi hatası (satır {reader.line_num}): {error}") from error
    return header, rows


def allocate_test_counts(groups, test_count, total):
    classes = len(groups)
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("Katmanlı bölme için her sınıfta en az iki kayıt olmalı.")
    if not classes <= test_count <= total - classes:
        raise ValueError("Bu test oranıyla her sınıftan iki bölüme de kayıt ayrılamıyor.")

    # Reserve one member of each class in each partition, then distribute the
    # remaining test places by largest remainders of proportional allocations.
    unassigned = test_count - classes
    capacity = max(1, total - 2 * classes)
    allocation = {}
    remainders = []
    for label, indices in groups.items():
        whole, remainder = divmod(unassigned * (len(indices) - 2), capacity)
        allocation[label] = 1 + whole
        remainders.append((-remainder, label))
    remaining = test_count - sum(allocation.values())
    for _, label in sorted(remainders)[:remaining]:
        allocation[label] += 1
    return allocation


def choose_test_rows(rows, test_count, seed, label_column):
    rng = random.Random(seed)
    if label_column is None:
        return set(rng.sample(range(len(rows)), test_count)), []

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        label = row[label_column]
        if not label.strip():
            raise ValueError(f"Veri kaydı {index + 1}: sınıf etiketi boş.")
        groups[label].append(index)
    allocation = allocate_test_counts(groups, test_count, len(rows))
    selected = set()
    class_counts = []
    for label in sorted(groups):
        indices = groups[label]
        count = allocation[label]
        selected.update(rng.sample(indices, count))
        class_counts.append({"label": label, "train": len(indices) - count, "test": count})
    return selected, class_counts


def write_outputs(output, header, rows, selected, metadata, delimiter):
    output.mkdir()
    created = []

    def new_file(name):
        path = output / name
        handle = path.open("x", encoding="utf-8", newline="")
        created.append(path)
        return handle

    try:
        with new_file("train.csv") as train, new_file("test.csv") as test:
            train_writer = csv.writer(train, delimiter=delimiter)
            test_writer = csv.writer(test, delimiter=delimiter)
            train_writer.writerow(header)
            test_writer.writerow(header)
            for index, row in enumerate(rows):
                writer = test_writer if index in selected else train_writer
                writer.writerow(row)
        with new_file("split.json") as manifest:
            json.dump(metadata, manifest, ensure_ascii=True, indent=2)
            manifest.write("\n")
    except BaseException:
        for path in created:
            with suppress(OSError):
                path.unlink()
        with suppress(OSError):
            output.rmdir()
        raise


def split_csv(source, output, ratio, seed, *, stratify=None, delimiter=",", encoding="utf-8-sig"):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Çıktı yolu zaten var: {output}")
    header, rows = read_table(source, delimiter, encoding)
    test_count = int(len(rows) * ratio + Fraction(1, 2))
    if not 1 <= test_count < len(rows):
        raise ValueError("Bu test oranıyla iki bölüme de en az bir kayıt ayrılamıyor.")
    if stratify is not None and stratify not in header:
        raise ValueError(f"Sınıf sütunu bulunamadı: {stratify}")
    label_column = header.index(stratify) if stratify is not None else None
    selected, class_counts = choose_test_rows(rows, test_count, seed, label_column)
    metadata = {
        "rows": len(rows),
        "train_rows": len(rows) - test_count,
        "test_rows": test_count,
        "requested_test_ratio": str(ratio),
        "seed": seed,
        "stratify": stratify,
        "classes": class_counts,
    }
    write_outputs(output, header, rows, selected, metadata, delimiter)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path, help="Oluşturulacak yeni klasör")
    parser.add_argument("--test-size", type=parse_ratio, default="0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", metavar="COLUMN", help="Her sınıfı iki bölüme dağıtmak için etiket sütunu")
    parser.add_argument("--delimiter", type=parse_delimiter, default=",")
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args(argv)
    try:
        result = split_csv(
            args.input, args.output_dir, args.test_size, args.seed,
            stratify=args.stratify, delimiter=args.delimiter, encoding=args.encoding,
        )
    except (OSError, LookupError, ValueError, csv.Error) as error:
        print(f"Bölme tamamlanamadı: {error}", file=sys.stderr)
        return 2
    print(f"Eğitim: {result['train_rows']} | Test: {result['test_rows']} | Tohum: {result['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
