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
          "p50": 1144.95,
          "p95": 1243.52
        },
        "map": 0.4817,
        "mrr": 0.8,
        "precision@5": 0.44,
        "recall@1": 0.12,
        "recall@20": 0.84,
        "recall@5": 0.44
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
          "p50": 139.78,
          "p95": 314.24
        },
        "map": 0.4232,
        "mrr": 0.7667,
        "precision@5": 0.36,
        "recall@1": 0.12,
        "recall@20": 0.84,
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
          "p50": 253.78,
          "p95": 383.24
        },
        "map": 0.3188,
        "mrr": 0.7778,
        "precision@5": 0.2,
        "recall@1": 0.1333,
        "recall@20": 0.7667,
        "recall@5": 0.2
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
          "p50": 82.08,
          "p95": 260.85
        },
        "map": 0.3104,
        "mrr": 0.6722,
        "precision@5": 0.2333,
        "recall@1": 0.1,
        "recall@20": 0.8,
        "recall@5": 0.2333
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
          "p50": 285.47,
          "p95": 343.87
        },
        "map": 0.2806,
        "mrr": 0.2786,
        "precision@5": 0.16,
        "recall@1": 0.0,
        "recall@20": 0.96,
        "recall@5": 0.16
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
          "p50": 154.97,
          "p95": 770.09
        },
        "map": 0.2775,
        "mrr": 0.2917,
        "precision@5": 0.2,
        "recall@1": 0.0,
        "recall@20": 0.96,
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
          "p50": 1096.67,
          "p95": 61578.24
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
          "p50": 734.88,
          "p95": 816.4
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
          "p50": 274.7,
          "p95": 331.54
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
          "p50": 96.64,
          "p95": 253.21
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
          "p50": 126.61,
          "p95": 147.38
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
          "p50": 89.59,
          "p95": 150.54
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
          "p50": 133.41,
          "p95": 293.38
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
          "p50": 152.45,
          "p95": 244.51
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
          "p50": 114.39,
          "p95": 131.56
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
