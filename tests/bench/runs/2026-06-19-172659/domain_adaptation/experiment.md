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
          "p50": 135.98,
          "p95": 160.29
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 345.05,
          "p95": 5395.21
        },
        "map": 0.7667,
        "mrr": 0.7667,
        "precision@5": 0.2,
        "recall@1": 0.6,
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
          "p50": 144.64,
          "p95": 165.18
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 212.22,
          "p95": 4888.56
        },
        "map": 0.875,
        "mrr": 0.875,
        "precision@5": 0.2,
        "recall@1": 0.8333,
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
          "p50": 116.84,
          "p95": 149.45
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 238.56,
          "p95": 5968.48
        },
        "map": 0.8,
        "mrr": 0.8,
        "precision@5": 0.2,
        "recall@1": 0.6,
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
          "p50": 107.44,
          "p95": 148.67
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 127.99,
          "p95": 135.03
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 120.78,
          "p95": 155.05
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 120.48,
          "p95": 150.64
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 191.23,
          "p95": 211.18
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 121.19,
          "p95": 136.57
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
