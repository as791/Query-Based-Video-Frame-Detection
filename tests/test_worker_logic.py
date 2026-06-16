from pathlib import Path

import importlib.util


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
        (frame_dir / f"{index + 1:05d}.jpg").write_bytes(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00"
            b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08\x0a\x0c\x14\x0d\x0c\x0b"
            b"\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c"
            b"\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\"\x00"
            b"\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x07\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03"
            b"\x11\x00?\x00\xaa\xff\xd9"
        )
    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(extractor.subprocess, "run", lambda *args, **kwargs: None)

    frames = extractor.extract_frames(str(video), fps=1.0, start_ms=7000)

    assert [frame[1] for frame in frames] == [7000, 8000]
    assert [frame[2] for frame in frames] == [0, 1000]
