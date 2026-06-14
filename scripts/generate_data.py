"""
Data Generation Script — run the SHUTEN-DŌJI data factory.

Usage:
    python scripts/generate_data.py --num 1000 --domain business
    python scripts/generate_data.py --config configs/factory/pipeline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.factory.state_generator import Domain
from src.training.data_pipeline import DataPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="SHUTEN-DŌJI Data Factory")
    parser.add_argument("--num", type=int, default=1000, help="Number of trajectories to generate")
    parser.add_argument("--domain", type=str, default=None, help="Specific domain (business/logistics/markets)")
    parser.add_argument("--output", type=str, default="data/trajectories", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export", action="store_true", help="Export for LLaMA Factory after generation")
    args = parser.parse_args()

    domains = [Domain(args.domain)] if args.domain else [Domain.BUSINESS, Domain.LOGISTICS, Domain.MARKETS]

    config = PipelineConfig(
        num_trajectories=args.num,
        domains=domains,
        output_dir=Path(args.output),
        seed=args.seed,
    )

    print(f"[SHUTEN-DŌJI] Starting data generation")
    print(f"  Trajectories: {config.num_trajectories}")
    print(f"  Domains: {[d.value for d in config.domains]}")
    print(f"  Output: {config.output_dir}")
    print()

    pipeline = DataPipeline(config)
    stats = pipeline.run()

    print(f"\n[SHUTEN-DŌJI] Generation complete:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    if args.export:
        print("\n[SHUTEN-DŌJI] Exporting for LLaMA Factory...")
        exports = pipeline.export_for_training()
        for name, path in exports.items():
            print(f"  {name}: {path}")

    print("\n[SHUTEN-DŌJI] Done.")


if __name__ == "__main__":
    main()
