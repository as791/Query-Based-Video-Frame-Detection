#!/usr/bin/env python3
"""Run v1 domain-adaptation benchmark experiments against a live VideoVault API."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from run_benchmark import (
    Auth,
    compute_search_metrics,
    iter_dataset,
    normalize_label,
    replay_search,
    request_json,
    require_auth,
    upload_clips,
    wait_ready,
    write_json,
)


DOMAINS: dict[str, list[str]] = {
    "violence_threat": ["hit", "kick", "struggle", "throw", "gun"],
    "motion_posture": ["fall", "lyingdown", "sit", "stand", "walk", "run"],
    "intrusion_behavior": ["sneak", "grab", "walk", "run", "stand"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--data-dir", default="tests/bench/data")
    parser.add_argument("--runs-dir", default="tests/bench/runs")
    parser.add_argument("--cookie", default=os.environ.get("VIDEOVAULT_COOKIE", ""))
    parser.add_argument("--benchmark-sub", default=os.environ.get("VIDEOVAULT_BENCHMARK_SUB", ""))
    parser.add_argument("--train-per-label", type=int, default=5)
    parser.add_argument("--eval-per-label", type=int, default=20)
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--wait-timeout-sec", type=int, default=7200)
    parser.add_argument("--search-limit", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=2, help="Feedback rounds after round0 for experiment 2")
    parser.add_argument("--feedback-hits-per-query", type=int, default=5,
                        help="Top-N search hits to use when simulating user feedback")
    args = parser.parse_args()

    require_auth(args.cookie, args.benchmark_sub)
    auth = Auth(args.cookie, args.benchmark_sub)
    root = Path.cwd()
    data_dir = (root / args.data_dir).resolve()
    run_dir = (root / args.runs_dir / datetime.now().strftime("%Y-%m-%d-%H%M%S") / "domain_adaptation").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"domain-{int(time.time())}"

    clips = iter_dataset(data_dir)
    by_label = group_by_label(clips)
    summary: dict[str, object] = {
        "api": args.api,
        "data_dir": str(data_dir),
        "run_id": run_id,
        "domains": DOMAINS,
        "train_per_label": args.train_per_label,
        "eval_per_label": args.eval_per_label,
        "upload_workers": args.upload_workers,
        "search_limit": args.search_limit,
        "rounds": args.rounds,
        "feedback_hits_per_query": args.feedback_hits_per_query,
    }
    write_json(run_dir / "run.json", summary)

    experiment1: dict[str, object] = {}
    experiment2: dict[str, object] = {}
    per_domain: dict[str, object] = {}

    for domain_key, raw_labels in DOMAINS.items():
        labels = [normalize_label(label) for label in raw_labels]
        domain_notes: dict[str, object] = {"labels": labels}
        train, eval_clips, missing = select_clips(by_label, labels, args.train_per_label, args.eval_per_label)
        domain_notes["missing_labels"] = missing
        domain_notes["train_count"] = len(train)
        domain_notes["eval_count"] = len(eval_clips)
        if not eval_clips:
            domain_notes["skipped"] = "no eval clips found"
            per_domain[domain_key] = domain_notes
            continue

        seeded_domain = create_domain(args.api, auth, f"{domain_key} seeded {run_id}", "Few-shot seeded domain benchmark")
        for label in labels:
            request_json("POST", f"{args.api}/v1/domains/{seeded_domain['domainId']}/labels", auth=auth, body={
                "label": label,
                "description": f"Domain-specific {label} examples",
            })

        domain_dir = run_dir / domain_key
        domain_dir.mkdir(parents=True, exist_ok=True)
        queries = {label: query_for(label) for label in labels}

        if train:
            train_run_id = f"{run_id}-{domain_key}-train"
            train_labels = upload_clips(
                args.api,
                auth,
                train,
                domain_dir,
                train_run_id,
                offset=0,
                total=len(train),
                workers=args.upload_workers,
                use_few_shot_labels=True,
                domain_id=seeded_domain["domainId"],
            )
            write_json(domain_dir / "experiment1_train_labels.json", train_labels)
            wait_ready(args.api, auth, train_labels, domain_dir, args.wait_timeout_sec, label=f"{domain_key} train")

        eval_run_id = f"{run_id}-{domain_key}-eval"
        eval_labels = upload_clips(
            args.api,
            auth,
            eval_clips,
            domain_dir,
            eval_run_id,
            offset=0,
            total=len(eval_clips),
            workers=args.upload_workers,
            domain_id=seeded_domain["domainId"],
        )
        write_json(domain_dir / "eval_labels.json", eval_labels)
        wait_ready(args.api, auth, eval_labels, domain_dir, args.wait_timeout_sec, label=f"{domain_key} eval")

        generic_dir = domain_dir / "experiment1_generic"
        domain_search_dir = domain_dir / "experiment1_domain"
        generic_dir.mkdir(parents=True, exist_ok=True)
        domain_search_dir.mkdir(parents=True, exist_ok=True)
        generic_raw = replay_search(args.api, auth, eval_labels, queries, generic_dir, args.search_limit, eval_run_id)
        domain_raw = replay_search(args.api, auth, eval_labels, queries, domain_search_dir, args.search_limit, eval_run_id, seeded_domain["domainId"])
        generic_metrics, generic_per_class = compute_search_metrics(eval_labels, generic_raw)
        domain_metrics, domain_per_class = compute_search_metrics(eval_labels, domain_raw)
        write_json(domain_dir / "experiment1_generic_metrics.json", generic_metrics)
        write_json(domain_dir / "experiment1_domain_metrics.json", domain_metrics)
        write_json(domain_dir / "experiment1_generic_per_class.json", generic_per_class)
        write_json(domain_dir / "experiment1_domain_per_class.json", domain_per_class)
        experiment1[domain_key] = {"generic": generic_metrics, "domain": domain_metrics}

        feedback_domain = create_domain(args.api, auth, f"{domain_key} feedback {run_id}", "Feedback-only adaptation benchmark")
        feedback_eval_run_id = f"{run_id}-{domain_key}-feedback-eval"
        feedback_labels = upload_clips(
            args.api,
            auth,
            eval_clips,
            domain_dir,
            feedback_eval_run_id,
            offset=0,
            total=len(eval_clips),
            workers=args.upload_workers,
            domain_id=feedback_domain["domainId"],
        )
        write_json(domain_dir / "experiment2_eval_labels.json", feedback_labels)
        wait_ready(args.api, auth, feedback_labels, domain_dir, args.wait_timeout_sec, label=f"{domain_key} feedback eval")

        round_metrics: dict[str, object] = {}
        for round_no in range(0, args.rounds + 1):
            round_dir = domain_dir / f"experiment2_round{round_no}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_raw = replay_search(
                args.api,
                auth,
                feedback_labels,
                queries,
                round_dir,
                args.search_limit,
                feedback_eval_run_id,
                feedback_domain["domainId"],
            )
            metrics, per_class = compute_search_metrics(feedback_labels, round_raw)
            write_json(domain_dir / f"experiment2_round{round_no}_metrics.json", metrics)
            write_json(domain_dir / f"experiment2_round{round_no}_per_class.json", per_class)
            round_metrics[f"round{round_no}"] = metrics
            if round_no < args.rounds:
                simulate_feedback(args.api, auth, feedback_domain["domainId"], feedback_labels, round_raw, args.feedback_hits_per_query)
        experiment2[domain_key] = round_metrics
        per_domain[domain_key] = domain_notes

    write_json(run_dir / "experiment1_metrics.json", experiment1)
    for round_name, metrics in aggregate_round_metrics(experiment2).items():
        write_json(run_dir / f"experiment2_{round_name}_metrics.json", metrics)
    for key, value in experiment2.items():
        write_json(run_dir / f"experiment2_{key}_metrics.json", value)
    write_json(run_dir / "per_domain.json", per_domain)
    write_experiment_doc(run_dir, experiment1, experiment2, per_domain)
    print(f"Domain adaptation experiments written to {run_dir}")
    return 0


def group_by_label(clips: list[tuple[Path, str]]) -> dict[str, list[tuple[Path, str]]]:
    grouped: dict[str, list[tuple[Path, str]]] = {}
    for clip, label in clips:
        grouped.setdefault(normalize_label(label), []).append((clip, normalize_label(label)))
    for items in grouped.values():
        items.sort(key=lambda item: str(item[0]))
    return grouped


def select_clips(
        by_label: dict[str, list[tuple[Path, str]]],
        labels: list[str],
        train_per_label: int,
        eval_per_label: int) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[str]]:
    train: list[tuple[Path, str]] = []
    eval_clips: list[tuple[Path, str]] = []
    missing: list[str] = []
    for label in labels:
        clips = by_label.get(label, [])
        if not clips:
            missing.append(label)
            continue
        train.extend(clips[:train_per_label])
        eval_clips.extend(clips[train_per_label:train_per_label + eval_per_label])
        if len(clips) < train_per_label + 1:
            missing.append(label)
    return train, eval_clips, missing


def create_domain(api: str, auth: Auth, name: str, description: str) -> dict[str, object]:
    created = request_json("POST", f"{api}/v1/domains", auth=auth, body={"name": name, "description": description})
    if not isinstance(created, dict) or not created.get("domainId"):
        raise SystemExit(f"Domain creation failed for {name}: {created}")
    return created


def query_for(label: str) -> str:
    display = "lying down" if label == "lyingdown" else label
    return f"a person {display}"


def simulate_feedback(
        api: str,
        auth: Auth,
        domain_id: str,
        labels: dict[str, dict[str, str]],
        rows: list[dict[str, object]],
        hits_per_query: int) -> None:
    label_by_video = {video_id: item["label"] for video_id, item in labels.items()}
    limit = max(1, hits_per_query)
    for row in rows:
        query = str(row.get("query", ""))
        label = str(row.get("label", ""))
        hits = row.get("hits") if isinstance(row.get("hits"), list) else []
        feedback_events: list[dict[str, object]] = []
        for hit in hits[:limit]:
            if not isinstance(hit, dict):
                continue
            video_id = str(hit.get("video_id", ""))
            feedback_events.append({
                "domainId": domain_id,
                "query": query,
                "videoId": video_id,
                "frameId": str(hit.get("frame_id", "")),
                "chunkId": str(hit.get("chunk_id", "")),
                "relevant": label_by_video.get(video_id) == label,
            })
        if feedback_events:
            request_json("POST", f"{api}/v1/search/feedback/batch", auth=auth, body=feedback_events)


def aggregate_round_metrics(experiment2: dict[str, object]) -> dict[str, object]:
    by_round: dict[str, list[dict[str, object]]] = {}
    for rounds in experiment2.values():
        if not isinstance(rounds, dict):
            continue
        for round_name, metrics in rounds.items():
            if isinstance(metrics, dict):
                by_round.setdefault(str(round_name), []).append(metrics)
    return {round_name: average_metric_set(items) for round_name, items in sorted(by_round.items())}


def average_metric_set(items: list[dict[str, object]]) -> dict[str, object]:
    if not items:
        return {}
    search_items = [item.get("search", {}) for item in items if isinstance(item.get("search"), dict)]
    return {
        "domains": len(items),
        "search": {
            "recall@1": avg(search_items, "recall@1"),
            "recall@5": avg(search_items, "recall@5"),
            "recall@20": avg(search_items, "recall@20"),
            "precision@5": avg(search_items, "precision@5"),
            "map": avg(search_items, "map"),
            "mrr": avg(search_items, "mrr"),
            "latency_ms": {
                "p50": avg_nested(search_items, "latency_ms", "p50"),
                "p95": avg_nested(search_items, "latency_ms", "p95"),
            },
        },
    }


def avg(items: list[object], key: str) -> float:
    values = [float(item.get(key, 0.0)) for item in items if isinstance(item, dict)]
    return round(sum(values) / len(values), 4) if values else 0.0


def avg_nested(items: list[object], outer: str, inner: str) -> float:
    values = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(outer), dict):
            continue
        values.append(float(item[outer].get(inner, 0.0)))
    return round(sum(values) / len(values), 2) if values else 0.0


def write_experiment_doc(
        run_dir: Path,
        experiment1: dict[str, object],
        experiment2: dict[str, object],
        per_domain: dict[str, object]) -> None:
    lines = [
        "# Domain Adaptation Experiment",
        "",
        "V1 uses domain profiles, few-shot prototypes, and feedback centroids. It does not train LoRA, SSL adapters, or classifier checkpoints.",
        "",
        "## Experiment 1",
        "",
        "Each synthetic domain is seeded with up to 5 labeled videos per label, then generic search is compared with domain-aware search on disjoint eval clips.",
        "",
        "```json",
        json.dumps(experiment1, indent=2, sort_keys=True),
        "```",
        "",
        "## Experiment 2",
        "",
        "Each synthetic domain starts without labels. Search feedback is simulated from ground-truth labels and replayed for repeated same-domain queries.",
        "",
        "```json",
        json.dumps(experiment2, indent=2, sort_keys=True),
        "```",
        "",
        "## Per-Domain Setup",
        "",
        "```json",
        json.dumps(per_domain, indent=2, sort_keys=True),
        "```",
        "",
        "## Future Options",
        "",
        "- Classifier over frozen embeddings: next step when each domain has enough labeled examples.",
        "- Self-supervised adapter: useful when a tenant has many unlabeled domain videos.",
        "- LoRA: useful only after GPU training infra and enough labeled data exist.",
        "",
    ]
    (run_dir / "experiment.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
