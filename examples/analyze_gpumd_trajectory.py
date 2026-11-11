"""Analyze an existing GPUMD extended-XYZ trajectory without launching MD."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tdekit.occupancy import analyze_occupancy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="GPUMD extended-XYZ trajectory")
    parser.add_argument("reference", type=Path, help="Single-frame reference structure")
    parser.add_argument("--map-to-reference", action="store_true")
    parser.add_argument("--cluster-cutoff-angstrom", type=float)
    parser.add_argument("--describe-clusters", action="store_true")
    args = parser.parse_args()
    if args.describe_clusters and args.cluster_cutoff_angstrom is None:
        parser.error("--describe-clusters requires --cluster-cutoff-angstrom")
    frames = analyze_occupancy(
        args.trajectory, reference=args.reference, map_to_reference=args.map_to_reference,
        cluster_cutoff_angstrom=args.cluster_cutoff_angstrom,
        describe_clusters=args.describe_clusters,
    )
    for frame in frames:
        record = asdict(frame)
        record["cluster_size_histogram"] = frame.cluster_size_histogram()
        print(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
