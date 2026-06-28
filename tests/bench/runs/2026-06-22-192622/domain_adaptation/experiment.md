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
          "p50": 391.13,
          "p95": 588.42
        },
        "map": 0.344,
        "mrr": 0.4186,
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
          "p50": 213.24,
          "p95": 392.42
        },
        "map": 0.4116,
        "mrr": 0.6167,
        "precision@5": 0.36,
        "recall@1": 0.08,
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
          "p50": 251.59,
          "p95": 458.52
        },
        "map": 0.3099,
        "mrr": 0.5208,
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
          "p50": 155.71,
          "p95": 421.49
        },
        "map": 0.3312,
        "mrr": 0.5571,
        "precision@5": 0.3,
        "recall@1": 0.0667,
        "recall@20": 0.8,
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
          "p50": 277.87,
          "p95": 510.44
        },
        "map": 0.2758,
        "mrr": 0.3908,
        "precision@5": 0.2,
        "recall@1": 0.04,
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
          "p50": 555.84,
          "p95": 1108.9
        },
        "map": 0.2813,
        "mrr": 0.2243,
        "precision@5": 0.2,
        "recall@1": 0.0,
        "recall@20": 0.92,
        "recall@5": 0.2
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
          "p50": 120.42,
          "p95": 380.45
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
          "p50": 172.5,
          "p95": 184.92
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
          "p50": 140.45,
          "p95": 188.56
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
          "p50": 156.49,
          "p95": 366.99
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
          "p50": 182.35,
          "p95": 290.62
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
          "p50": 127.94,
          "p95": 150.96
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
          "p50": 160.4,
          "p95": 431.8
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
          "p50": 160.97,
          "p95": 210.43
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
          "p50": 140.52,
          "p95": 161.38
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
