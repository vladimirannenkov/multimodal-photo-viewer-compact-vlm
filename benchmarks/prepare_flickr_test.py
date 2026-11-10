"""Join Flickr30k mirror shard to the official split and captions by filename."""

import argparse
import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shard")
    parser.add_argument("annotations")
    parser.add_argument("out")
    args = parser.parse_args()
    with open(args.annotations, encoding="utf-8", newline="") as stream:
        annotations = {row["filename"]: row for row in csv.DictReader(stream) if row["split"] == "test"}
    rows = []
    for row in pq.read_table(args.shard).to_pylist():
        item = annotations.get(row["filename"])
        if item:
            row["caption"] = json.loads(item["raw"])
            row["split"] = item["split"]
            rows.append(row)
    if not rows:
        raise SystemExit("No matching test rows")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)
    print(f"{len(rows)} original Flickr30k test rows")  # noqa: T201


if __name__ == "__main__":
    main()
