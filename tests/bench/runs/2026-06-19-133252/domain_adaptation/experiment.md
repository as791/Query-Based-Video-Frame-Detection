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
          "p50": 101.94,
          "p95": 128.59
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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 481.14,
          "p95": 5323.92
        },
        "map": 0.4459,
        "mrr": 0.7167,
        "precision@5": 0.4,
        "recall@1": 0.12,
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
          "p50": 83.56,
          "p95": 315.23
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
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 371.96,
          "p95": 5330.18
        },
        "map": 0.3233,
        "mrr": 0.875,
        "precision@5": 0.2667,
        "recall@1": 0.1667,
        "recall@20": 0.7333,
        "recall@5": 0.2667
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
          "p50": 81.88,
          "p95": 124.31
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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 340.02,
          "p95": 5723.03
        },
        "map": 0.326,
        "mrr": 0.5333,
        "precision@5": 0.2,
        "recall@1": 0.08,
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
          "p50": 82.56,
          "p95": 121.84
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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 73.41,
          "p95": 132.71
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 84.35,
          "p95": 112.2
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
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 80.33,
          "p95": 598.74
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
        "videos_total": 30
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 88.21,
          "p95": 227.59
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 54.0,
          "p95": 99.46
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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 45.01,
          "p95": 110.29
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
        "videos_total": 25
      },
      "profile": "cctv",
      "search": {
        "latency_ms": {
          "p50": 62.18,
          "p95": 76.64
        },
        "map": 0.0,
        "mrr": 0.0,
        "precision@5": 0.0,
        "recall@1": 0.0,
        "recall@20": 0.0,
        "recall@5": 0.0
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
          "p50": 47.53,
          "p95": 64.58
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
