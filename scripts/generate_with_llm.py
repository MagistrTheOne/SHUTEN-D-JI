"""
SHUTEN Synthetic Data Generator — generates training corpus using Qwen3-235B.

This script:
1. Connects to local vLLM endpoint serving Qwen3-235B-GPTQ
2. Generates world states (template-based, CPU)
3. Sends them to the LLM for rich content generation
4. Evaluates and scores outputs
5. Stores in LLaMA Factory sharegpt format

Usage:
    uv run python scripts/generate_with_llm.py --num 1000 --output data/trajectories
    uv run python scripts/generate_with_llm.py --num 50000 --batch-size 32 --output data/trajectories
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.factory.llm_client import LLMClient, LLMConfig
from src.factory.prompts import (
    SHUTEN_SYSTEM_PROMPT,
    STRATEGIC_ANALYSIS_PROMPT,
    SCENARIO_PLANNING_PROMPT,
    AGENT_DECISION_PROMPT,
    CRITIQUE_PROMPT,
    IDENTITY_REINFORCEMENT_PROMPTS,
    format_sharegpt_conversation,
    format_dpo_pair,
)
from src.factory.state_generator import StateGenerator, StateGeneratorConfig, Domain

console = Console()


async def generate_strategic_analysis(
    client: LLMClient,
    state_gen: StateGenerator,
    domain: Domain,
) -> dict | None:
    """Generate a single strategic analysis trajectory."""
    state = state_gen.generate(domain=domain)
    state_text = state.model_dump_json(indent=2)

    prompt = STRATEGIC_ANALYSIS_PROMPT.format(world_state=state_text)

    response = await client.generate(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=SHUTEN_SYSTEM_PROMPT,
        max_tokens=4096,
    )

    if "[ERROR" in response or len(response) < 200:
        return None

    return format_sharegpt_conversation(
        system=SHUTEN_SYSTEM_PROMPT,
        user=prompt,
        assistant=response,
    )


async def generate_scenario_planning(
    client: LLMClient,
    state_gen: StateGenerator,
    domain: Domain,
) -> dict | None:
    """Generate a scenario planning trajectory."""
    state = state_gen.generate(domain=domain)
    state_text = state.model_dump_json(indent=2)

    triggers = [
        "A major competitor unexpectedly exits the market",
        "Critical supply chain disruption in primary region",
        "Regulatory framework undergoes fundamental change",
        "Key technological breakthrough shifts competitive landscape",
        "Unexpected alliance forms between previously hostile actors",
        "Black swan financial event impacts core revenue stream",
        "Internal leadership crisis creates power vacuum",
        "New entrant with 10x resource advantage appears",
    ]

    import random
    trigger = random.choice(triggers)
    horizon = random.choice(["3 months", "6 months", "1 year", "2 years"])

    prompt = SCENARIO_PLANNING_PROMPT.format(
        world_state=state_text,
        trigger_event=trigger,
        horizon=horizon,
    )

    response = await client.generate(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=SHUTEN_SYSTEM_PROMPT,
        max_tokens=4096,
    )

    if "[ERROR" in response or len(response) < 200:
        return None

    return format_sharegpt_conversation(
        system=SHUTEN_SYSTEM_PROMPT,
        user=prompt,
        assistant=response,
    )


async def generate_critique_pair(
    client: LLMClient,
    state_gen: StateGenerator,
    domain: Domain,
) -> dict | None:
    """Generate analysis + critique for DPO training."""
    state = state_gen.generate(domain=domain)
    state_text = state.model_dump_json(indent=2)

    analysis_prompt = STRATEGIC_ANALYSIS_PROMPT.format(world_state=state_text)

    analysis = await client.generate(
        messages=[{"role": "user", "content": analysis_prompt}],
        system_prompt=SHUTEN_SYSTEM_PROMPT,
        max_tokens=3072,
        temperature=0.8,
    )

    if "[ERROR" in analysis or len(analysis) < 200:
        return None

    critique_prompt = CRITIQUE_PROMPT.format(
        analysis=analysis,
        context=state_text[:2000],
    )

    critique = await client.generate(
        messages=[{"role": "user", "content": critique_prompt}],
        system_prompt=SHUTEN_SYSTEM_PROMPT,
        max_tokens=2048,
    )

    if "[ERROR" in critique:
        return None

    improved_prompt = (
        f"{analysis_prompt}\n\n"
        f"IMPORTANT: Address the following critique in your analysis:\n{critique}"
    )
    improved_analysis = await client.generate(
        messages=[{"role": "user", "content": improved_prompt}],
        system_prompt=SHUTEN_SYSTEM_PROMPT,
        max_tokens=4096,
        temperature=0.6,
    )

    if "[ERROR" in improved_analysis or len(improved_analysis) < 200:
        return None

    return format_dpo_pair(
        system=SHUTEN_SYSTEM_PROMPT,
        user=analysis_prompt,
        chosen=improved_analysis,
        rejected=analysis,
    )


async def generate_identity_data() -> list[dict]:
    """Generate identity reinforcement data (no LLM needed)."""
    data = []
    for item in IDENTITY_REINFORCEMENT_PROMPTS:
        data.append(format_sharegpt_conversation(
            system=SHUTEN_SYSTEM_PROMPT,
            user=item["user"],
            assistant=item["assistant"],
        ))
    return data


async def main(args: argparse.Namespace):
    config = LLMConfig(
        base_url=f"http://localhost:{args.port}/v1",
        model=args.model,
        max_concurrent=args.concurrency,
    )
    client = LLMClient(config)

    console.print("[bold green]SHUTEN Synthetic Data Generator[/bold green]")
    console.print(f"Target: {args.num} trajectories")
    console.print(f"Output: {args.output}")
    console.print(f"Concurrency: {args.concurrency}")
    console.print()

    health = await client.health_check()
    if not health:
        console.print("[bold red]ERROR: vLLM server not reachable![/bold red]")
        console.print(f"Make sure server is running on port {args.port}")
        return

    console.print("[green]vLLM server connected.[/green]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_data: list[dict] = []
    dpo_data: list[dict] = []

    identity_data = await generate_identity_data()
    sft_data.extend(identity_data * 50)
    console.print(f"[dim]Added {len(identity_data) * 50} identity reinforcement samples[/dim]")

    state_gen = StateGenerator(StateGeneratorConfig(
        domains=[Domain.BUSINESS, Domain.LOGISTICS, Domain.MARKETS, Domain.GEOPOLITICS],
        seed=42,
    ))
    domains = [Domain.BUSINESS, Domain.LOGISTICS, Domain.MARKETS, Domain.GEOPOLITICS]

    num_analysis = int(args.num * 0.4)
    num_scenario = int(args.num * 0.3)
    num_critique = int(args.num * 0.3)

    import random
    random.seed(42)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_analysis = progress.add_task("Strategic Analysis", total=num_analysis)
        task_scenario = progress.add_task("Scenario Planning", total=num_scenario)
        task_critique = progress.add_task("Critique Pairs (DPO)", total=num_critique)

        for i in range(0, num_analysis, args.batch_size):
            batch_size = min(args.batch_size, num_analysis - i)
            tasks = [
                generate_strategic_analysis(client, state_gen, random.choice(domains))
                for _ in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is not None:
                    sft_data.append(r)
            progress.update(task_analysis, advance=batch_size)

            if (i + batch_size) % 100 == 0:
                _save_checkpoint(sft_data, dpo_data, output_dir)

        for i in range(0, num_scenario, args.batch_size):
            batch_size = min(args.batch_size, num_scenario - i)
            tasks = [
                generate_scenario_planning(client, state_gen, random.choice(domains))
                for _ in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is not None:
                    sft_data.append(r)
            progress.update(task_scenario, advance=batch_size)

            if (i + batch_size) % 100 == 0:
                _save_checkpoint(sft_data, dpo_data, output_dir)

        for i in range(0, num_critique, args.batch_size):
            batch_size = min(args.batch_size, num_critique - i)
            tasks = [
                generate_critique_pair(client, state_gen, random.choice(domains))
                for _ in range(batch_size)
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                if r is not None:
                    dpo_data.append(r)
            progress.update(task_critique, advance=batch_size)

            if (i + batch_size) % 50 == 0:
                _save_checkpoint(sft_data, dpo_data, output_dir)

    _save_final(sft_data, dpo_data, output_dir)

    console.print()
    console.print("[bold green]Generation Complete![/bold green]")
    console.print(f"  SFT samples: {len(sft_data)}")
    console.print(f"  DPO pairs: {len(dpo_data)}")
    console.print(f"  LLM stats: {client.stats}")


def _save_checkpoint(sft_data: list, dpo_data: list, output_dir: Path):
    """Save intermediate checkpoint."""
    if sft_data:
        with open(output_dir / "sft_train.json", "w") as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=None)
    if dpo_data:
        with open(output_dir / "dpo_train.json", "w") as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=None)


def _save_final(sft_data: list, dpo_data: list, output_dir: Path):
    """Save final datasets with proper splits."""
    import random
    random.shuffle(sft_data)
    random.shuffle(dpo_data)

    split_idx = int(len(sft_data) * 0.95)
    sft_train = sft_data[:split_idx]
    sft_eval = sft_data[split_idx:]

    with open(output_dir / "sft_train.json", "w") as f:
        json.dump(sft_train, f, ensure_ascii=False)
    with open(output_dir / "sft_eval.json", "w") as f:
        json.dump(sft_eval, f, ensure_ascii=False)

    if dpo_data:
        split_idx = int(len(dpo_data) * 0.9)
        with open(output_dir / "dpo_train.json", "w") as f:
            json.dump(dpo_data[:split_idx], f, ensure_ascii=False)
        with open(output_dir / "dpo_eval.json", "w") as f:
            json.dump(dpo_data[split_idx:], f, ensure_ascii=False)

    dataset_info = {
        "shuten_sft_train": {
            "file_name": "sft_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        },
        "shuten_sft_eval": {
            "file_name": "sft_eval.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        },
        "shuten_dpo_train": {
            "file_name": "dpo_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    console.print(f"[dim]Saved: {len(sft_train)} train, {len(sft_eval)} eval, {len(dpo_data)} DPO[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHUTEN Synthetic Data Generator")
    parser.add_argument("--num", type=int, default=1000, help="Number of trajectories to generate")
    parser.add_argument("--output", type=str, default="data/trajectories", help="Output directory")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B-GPTQ-Int4")
    parser.add_argument("--batch-size", type=int, default=16, help="Concurrent batch size")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(main(args))
