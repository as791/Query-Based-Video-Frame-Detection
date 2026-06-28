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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 176.09,
          "p95": 240.81
        },
        "map": 0.4293,
        "mrr": 0.85,
        "precision@5": 0.36,
        "recall@1": 0.16,
        "recall@20": 0.92,
        "recall@5": 0.36
      }
    },
    "generic": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 176.0,
          "p95": 238.95
        },
        "map": 0.414,
        "mrr": 0.85,
        "precision@5": 0.36,
        "recall@1": 0.16,
        "recall@20": 0.88,
        "recall@5": 0.36
      }
    }
  },
  "motion_posture": {
    "domain": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 234.35,
          "p95": 306.57
        },
        "map": 0.2971,
        "mrr": 0.625,
        "precision@5": 0.3333,
        "recall@1": 0.0667,
        "recall@20": 0.7333,
        "recall@5": 0.3333
      }
    },
    "generic": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 240.09,
          "p95": 425.83
        },
        "map": 0.303,
        "mrr": 0.6389,
        "precision@5": 0.3,
        "recall@1": 0.0667,
        "recall@20": 0.7333,
        "recall@5": 0.3
      }
    }
  },
  "violence_threat": {
    "domain": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 388.22,
          "p95": 547.25
        },
        "map": 0.31,
        "mrr": 0.42,
        "precision@5": 0.28,
        "recall@1": 0.04,
        "recall@20": 0.88,
        "recall@5": 0.28
      }
    },
    "generic": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 415.31,
          "p95": 795.31
        },
        "map": 0.3035,
        "mrr": 0.4515,
        "precision@5": 0.28,
        "recall@1": 0.04,
        "recall@20": 0.84,
        "recall@5": 0.28
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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 116.64,
          "p95": 242.66
        },
        "map": 0.2413,
        "mrr": 0.3533,
        "precision@5": 0.2,
        "recall@1": 0.04,
        "recall@20": 0.8,
        "recall@5": 0.2
      }
    },
    "round1": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 96.0,
          "p95": 140.54
        },
        "map": 0.4019,
        "mrr": 0.6567,
        "precision@5": 0.32,
        "recall@1": 0.12,
        "recall@20": 0.8,
        "recall@5": 0.32
      }
    },
    "round2": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 91.47,
          "p95": 124.12
        },
        "map": 0.4358,
        "mrr": 0.8167,
        "precision@5": 0.32,
        "recall@1": 0.16,
        "recall@20": 0.8,
        "recall@5": 0.32
      }
    }
  },
  "motion_posture": {
    "round0": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 185.16,
          "p95": 485.96
        },
        "map": 0.1809,
        "mrr": 0.2565,
        "precision@5": 0.2,
        "recall@1": 0.0,
        "recall@20": 0.7,
        "recall@5": 0.2
      }
    },
    "round1": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 217.84,
          "p95": 382.39
        },
        "map": 0.3324,
        "mrr": 0.8438,
        "precision@5": 0.2667,
        "recall@1": 0.1667,
        "recall@20": 0.7,
        "recall@5": 0.2667
      }
    },
    "round2": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 205.75,
          "p95": 325.63
        },
        "map": 0.3607,
        "mrr": 0.8438,
        "precision@5": 0.3,
        "recall@1": 0.1667,
        "recall@20": 0.6667,
        "recall@5": 0.3
      }
    }
  },
  "violence_threat": {
    "round0": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 190.76,
          "p95": 364.14
        },
        "map": 0.2791,
        "mrr": 0.3952,
        "precision@5": 0.16,
        "recall@1": 0.04,
        "recall@20": 0.88,
        "recall@5": 0.16
      }
    },
    "round1": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 205.03,
          "p95": 442.29
        },
        "map": 0.3581,
        "mrr": 0.6452,
        "precision@5": 0.24,
        "recall@1": 0.12,
        "recall@20": 0.88,
        "recall@5": 0.24
      }
    },
    "round2": {
      "dataset": "kaggle:jonathannield/cctv-action-recognition-dataset",
      "ingest": {
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 259.41,
          "p95": 274.21
        },
        "map": 0.3562,
        "mrr": 0.55,
        "precision@5": 0.24,
        "recall@1": 0.08,
        "recall@20": 0.88,
        "recall@5": 0.24
      }
    }
  }
}
```

## Per-Domain Setup

```json
{
  "intrusion_behavior": {
    "eval_count": 25,
    "labels": [
      "sneak",
      "grab",
      "walk",
      "run",
      "stand"
    ],
    "missing_labels": [],
    "train_count": 25
  },
  "motion_posture": {
    "eval_count": 30,
    "labels": [
      "fall",
      "lyingdown",
      "sit",
      "stand",
      "walk",
      "run"
    ],
    "missing_labels": [],
    "train_count": 30
  },
  "violence_threat": {
    "eval_count": 25,
    "labels": [
      "hit",
      "kick",
      "struggle",
      "throw",
      "gun"
    ],
    "missing_labels": [],
    "train_count": 25
  }
}
```

## Future Options

- Classifier over frozen embeddings: next step when each domain has enough labeled examples.
- Self-supervised adapter: useful when a tenant has many unlabeled domain videos.
- LoRA: useful only after GPU training infra and enough labeled data exist.
