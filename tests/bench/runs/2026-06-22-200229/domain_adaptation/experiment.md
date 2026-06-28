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
          "p50": 269.01,
          "p95": 380.18
        },
        "map": 0.3794,
        "mrr": 0.65,
        "precision@5": 0.36,
        "recall@1": 0.08,
        "recall@20": 0.84,
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
          "p50": 128.95,
          "p95": 349.24
        },
        "map": 0.4253,
        "mrr": 0.6667,
        "precision@5": 0.4,
        "recall@1": 0.08,
        "recall@20": 0.88,
        "recall@5": 0.4
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
          "p50": 333.77,
          "p95": 520.15
        },
        "map": 0.3058,
        "mrr": 0.4653,
        "precision@5": 0.3,
        "recall@1": 0.0333,
        "recall@20": 0.8333,
        "recall@5": 0.3
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
          "p50": 201.19,
          "p95": 398.78
        },
        "map": 0.2895,
        "mrr": 0.45,
        "precision@5": 0.2,
        "recall@1": 0.0333,
        "recall@20": 0.8333,
        "recall@5": 0.2
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
          "p50": 436.35,
          "p95": 446.53
        },
        "map": 0.2361,
        "mrr": 0.26,
        "precision@5": 0.2,
        "recall@1": 0.0,
        "recall@20": 0.88,
        "recall@5": 0.2
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
          "p50": 360.82,
          "p95": 1504.45
        },
        "map": 0.3041,
        "mrr": 0.4867,
        "precision@5": 0.24,
        "recall@1": 0.04,
        "recall@20": 0.92,
        "recall@5": 0.24
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
          "p50": 142.14,
          "p95": 374.25
        },
        "map": 0.2346,
        "mrr": 0.3486,
        "precision@5": 0.16,
        "recall@1": 0.04,
        "recall@20": 0.76,
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
          "p50": 131.83,
          "p95": 171.17
        },
        "map": 0.3584,
        "mrr": 0.6429,
        "precision@5": 0.24,
        "recall@1": 0.12,
        "recall@20": 0.76,
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
          "p50": 119.11,
          "p95": 123.03
        },
        "map": 0.3726,
        "mrr": 0.6476,
        "precision@5": 0.24,
        "recall@1": 0.12,
        "recall@20": 0.76,
        "recall@5": 0.24
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
          "p50": 119.3,
          "p95": 369.01
        },
        "map": 0.1969,
        "mrr": 0.3398,
        "precision@5": 0.2,
        "recall@1": 0.0333,
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
          "p50": 166.82,
          "p95": 194.55
        },
        "map": 0.3152,
        "mrr": 0.8426,
        "precision@5": 0.2,
        "recall@1": 0.1667,
        "recall@20": 0.7,
        "recall@5": 0.2
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
          "p50": 131.59,
          "p95": 265.25
        },
        "map": 0.3244,
        "mrr": 0.8426,
        "precision@5": 0.2,
        "recall@1": 0.1667,
        "recall@20": 0.7,
        "recall@5": 0.2
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
          "p50": 215.12,
          "p95": 413.63
        },
        "map": 0.2811,
        "mrr": 0.3952,
        "precision@5": 0.2,
        "recall@1": 0.04,
        "recall@20": 0.88,
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
          "p50": 166.71,
          "p95": 242.75
        },
        "map": 0.3679,
        "mrr": 0.6487,
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
          "p50": 191.75,
          "p95": 212.0
        },
        "map": 0.3462,
        "mrr": 0.5487,
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
