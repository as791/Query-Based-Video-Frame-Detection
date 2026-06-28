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
          "p50": 125.0,
          "p95": 142.17
        },
        "map": 0.4507,
        "mrr": 0.6333,
        "precision@5": 0.4,
        "recall@1": 0.08,
        "recall@20": 0.92,
        "recall@5": 0.4
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
          "p50": 73.85,
          "p95": 268.62
        },
        "map": 0.4388,
        "mrr": 0.6,
        "precision@5": 0.44,
        "recall@1": 0.08,
        "recall@20": 0.88,
        "recall@5": 0.44
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
          "p50": 136.64,
          "p95": 163.68
        },
        "map": 0.3332,
        "mrr": 0.6722,
        "precision@5": 0.3333,
        "recall@1": 0.1,
        "recall@20": 0.7667,
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
          "p50": 94.21,
          "p95": 239.26
        },
        "map": 0.3283,
        "mrr": 0.65,
        "precision@5": 0.3,
        "recall@1": 0.1,
        "recall@20": 0.7667,
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
          "p50": 235.58,
          "p95": 433.13
        },
        "map": 0.2853,
        "mrr": 0.5,
        "precision@5": 0.28,
        "recall@1": 0.04,
        "recall@20": 0.84,
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
          "p50": 265.62,
          "p95": 429.12
        },
        "map": 0.2753,
        "mrr": 0.4733,
        "precision@5": 0.24,
        "recall@1": 0.04,
        "recall@20": 0.84,
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
          "p50": 78.58,
          "p95": 218.19
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
          "p50": 83.51,
          "p95": 190.25
        },
        "map": 0.3283,
        "mrr": 0.644,
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
          "p50": 84.41,
          "p95": 277.73
        },
        "map": 0.3483,
        "mrr": 0.644,
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
          "p50": 104.9,
          "p95": 225.48
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
          "p50": 93.35,
          "p95": 296.74
        },
        "map": 0.311,
        "mrr": 0.8438,
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
          "p50": 118.44,
          "p95": 245.38
        },
        "map": 0.3122,
        "mrr": 0.8438,
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
          "p50": 121.43,
          "p95": 302.15
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
          "p50": 123.79,
          "p95": 358.97
        },
        "map": 0.3344,
        "mrr": 0.5667,
        "precision@5": 0.28,
        "recall@1": 0.08,
        "recall@20": 0.88,
        "recall@5": 0.28
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
          "p50": 88.3,
          "p95": 289.35
        },
        "map": 0.3691,
        "mrr": 0.7167,
        "precision@5": 0.28,
        "recall@1": 0.12,
        "recall@20": 0.88,
        "recall@5": 0.28
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
