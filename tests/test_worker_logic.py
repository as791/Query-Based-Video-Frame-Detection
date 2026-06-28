from pathlib import Path

import importlib.util
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chunker = load_module("chunker_app", ROOT / "MLService" / "chunker" / "app.py")
extractor = load_module("extractor_app", ROOT / "MLService" / "extractor" / "app.py")


def test_normalize_windows_merges_subsecond_scenes():
    windows = [(0.0, 6.4), (6.4, 7.0), (7.0, 8.5), (8.5, 20.0)]

    result = chunker.normalize_windows(windows, 20.0)

    assert result == [(0.0, 7.0), (7.0, 8.5), (8.5, 20.0)]
    assert all(end - start >= 1.0 for start, end in result)


def test_normalize_windows_caps_long_windows():
    result = chunker.normalize_windows([(0.0, 65.0)], 65.0)

    assert result == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]


def test_extractor_frame_time_is_absolute(monkeypatch, tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (1, 1), color=(index, index, index)).save(frame_dir / f"{index + 1:05d}.jpg")
    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(extractor.subprocess, "run", lambda *args, **kwargs: None)

    frames = extractor.extract_frames(str(video), fps=1.0, start_ms=7000)

    assert [frame[1] for frame in frames] == [7000, 8000]
    assert [frame[2] for frame in frames] == [0, 1000]


def test_extractor_scores_action_labels_from_frame_embeddings(monkeypatch):
    taxonomy = [
        {"label": "fall", "description": "a person falling down"},
        {"label": "gun", "description": "a person holding a gun"},
        {"label": "hit", "description": "a person hitting another person"},
    ]
    monkeypatch.setattr(extractor, "_ACTION_TAXONOMY", taxonomy)
    monkeypatch.setattr(extractor, "action_text_embeddings", lambda _, r=None: [
        {"label": "fall", "embedding": [1.0, 0.0]},
        {"label": "gun", "embedding": [0.0, 1.0]},
        {"label": "hit", "embedding": [0.5, 0.5]},
    ])

    result = extractor.score_actions_from_embeddings(
        frame_embeddings=[[0.9, 0.1], [1.0, 0.0]],
        action_taxonomy=taxonomy,
    )

    assert result["action_top"] == "fall"
    assert result["action_labels"][0] == "fall"
    assert result["action_scores"]["fall"] == 1.0
    assert result["action_confidence"] > 0
