# Domain Adaptation Experiment

V1 uses domain profiles, few-shot prototypes, and feedback centroids. It does not train LoRA, SSL adapters, or classifier checkpoints.

## Experiment 1

Each synthetic domain is seeded with up to 5 labeled videos per label, then generic search is compared with domain-aware search on disjoint eval clips.

```json
{
  "intrusion_behavior": {
    "domain": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 244.13,
          "p95": 318.31
        },
        "map": 0.6667,
        "mrr": 0.6667,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    },
    "generic": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 386.61,
          "p95": 5391.53
        },
        "map": 0.6,
        "mrr": 0.6,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    }
  },
  "motion_posture": {
    "domain": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 6
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 330.31,
          "p95": 416.18
        },
        "map": 0.5111,
        "mrr": 0.5111,
        "precision@5": 0.1667,
        "recall@1": 0.3333,
        "recall@20": 1.0,
        "recall@5": 0.8333
      }
    },
    "generic": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 6
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 203.39,
          "p95": 4902.51
        },
        "map": 0.4833,
        "mrr": 0.4833,
        "precision@5": 0.2,
        "recall@1": 0.3333,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    }
  },
  "violence_threat": {
    "domain": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 270.97,
          "p95": 318.01
        },
        "map": 0.6067,
        "mrr": 0.6067,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    },
    "generic": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 165.51,
          "p95": 5285.13
        },
        "map": 0.54,
        "mrr": 0.54,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    }
  }
}
```

## Experiment 2

Each synthetic domain starts without labels. Search feedback is simulated from ground-truth labels and replayed for repeated same-domain queries.

```json
{
  "intrusion_behavior": {
    "round0": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 146.23,
          "p95": 153.01
        },
        "map": 0.5667,
        "mrr": 0.5667,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    },
    "round1": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 230.46,
          "p95": 5435.36
        },
        "map": 0.6333,
        "mrr": 0.6333,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    }
  },
  "motion_posture": {
    "round0": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 6
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 138.09,
          "p95": 189.57
        },
        "map": 0.4139,
        "mrr": 0.4139,
        "precision@5": 0.2,
        "recall@1": 0.1667,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    },
    "round1": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 6
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 548.36,
          "p95": 5496.43
        },
        "map": 0.45,
        "mrr": 0.45,
        "precision@5": 0.2,
        "recall@1": 0.1667,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    }
  },
  "violence_threat": {
    "round0": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 306.05,
          "p95": 510.28
        },
        "map": 0.4567,
        "mrr": 0.4567,
        "precision@5": 0.2,
        "recall@1": 0.2,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    },
    "round1": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 5
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 259.61,
          "p95": 5406.26
        },
        "map": 0.5567,
        "mrr": 0.5567,
        "precision@5": 0.2,
        "recall@1": 0.4,
        "recall@20": 1.0,
        "recall@5": 1.0
      }
    }
  }
}
```

## Per-Domain Setup

```json
{
  "intrusion_behavior": {
    "eval_count": 5,
    "labels": [
      "sneak",
      "grab",
      "walk",
      "run",
      "stand"
    ],
    "missing_labels": [],
    "train_count": 5
  },
  "motion_posture": {
    "eval_count": 6,
    "labels": [
      "fall",
      "lyingdown",
      "sit",
      "stand",
      "walk",
      "run"
    ],
    "missing_labels": [],
    "train_count": 6
  },
  "violence_threat": {
    "eval_count": 5,
    "labels": [
      "hit",
      "kick",
      "struggle",
      "throw",
      "gun"
    ],
    "missing_labels": [],
    "train_count": 5
  }
}
```

## Future Options

- Classifier over frozen embeddings: next step when each domain has enough labeled examples.
- Self-supervised adapter: useful when a tenant has many unlabeled domain videos.
- LoRA: useful only after GPU training infra and enough labeled data exist.
