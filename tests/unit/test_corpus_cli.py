"""External corpus CLI 測試（Slice B）。

完全 offline：只走 ``snapshot --raw-zip``（由本機 ZIP 建立）與參數／config 驗證。
``fetch`` 是唯一會碰網路的子指令，本檔**不執行**它，只驗證其 parser 契約。
"""
from __future__ import annotations

import json
import os

import pytest

from evaluation.corpus import cli
from evaluation.corpus.snapshot import MANIFEST_FILENAME, NORMALIZED_DIRNAME
from evaluation.v11 import hashing
from tests.unit.test_corpus_moj_law import fixture_zip

PILOT_CONFIG_PATH = os.path.join(
    hashing.repo_root(), "evaluation", "configs", "corpus-moj-law-pilot.json"
)


def write_raw_zip(tmp_path) -> str:
    path = tmp_path / "ChLaw.zip"
    path.write_bytes(fixture_zip())
    return str(path)


def write_config(tmp_path, **overrides) -> str:
    with open(PILOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# parser 契約
# ---------------------------------------------------------------------------
def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "fetch" in out and "snapshot" in out


def test_stats_subcommand_is_not_implemented_in_slice_b():
    """``stats`` 屬 Slice D，此 slice 不得提前提供。"""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["stats", "--config", PILOT_CONFIG_PATH])
    assert excinfo.value.code != 0


def test_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# snapshot（offline，經 --raw-zip）
# ---------------------------------------------------------------------------
def test_snapshot_builds_from_local_raw_zip(tmp_path, capsys):
    root = tmp_path / "corpora"
    exit_code = cli.main(
        [
            "snapshot",
            "--config", write_config(tmp_path),
            "--raw-zip", write_raw_zip(tmp_path),
            "--output-root", str(root),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "snapshot_id:" in out and "corpus_hash:" in out

    snapshots = os.listdir(root)
    assert len(snapshots) == 1
    snapshot_dir = os.path.join(str(root), snapshots[0])
    assert os.path.isfile(os.path.join(snapshot_dir, MANIFEST_FILENAME))
    assert os.listdir(os.path.join(snapshot_dir, NORMALIZED_DIRNAME))


def test_snapshot_on_existing_snapshot_fails_without_partial_output(tmp_path, capsys):
    root = tmp_path / "corpora"
    argv = [
        "snapshot",
        "--config", write_config(tmp_path),
        "--raw-zip", write_raw_zip(tmp_path),
        "--output-root", str(root),
    ]
    assert cli.main(argv) == 0
    before = sorted(os.listdir(root))

    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "SnapshotExistsError" in err

    assert sorted(os.listdir(root)) == before, "失敗不得留下 tmp 或半成品 snapshot"


def test_snapshot_rejects_invalid_config(tmp_path, capsys):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"provider": "other"}), encoding="utf-8")

    exit_code = cli.main(["snapshot", "--config", str(config_path)])

    assert exit_code == 2
    assert "config 錯誤" in capsys.readouterr().err


def test_snapshot_reports_missing_config_file(tmp_path, capsys):
    exit_code = cli.main(["snapshot", "--config", str(tmp_path / "nope.json")])

    assert exit_code == 2
    assert "找不到 config" in capsys.readouterr().err


def test_snapshot_rejects_non_zip_raw(tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"<html>not a zip</html>")

    exit_code = cli.main(
        [
            "snapshot",
            "--config", write_config(tmp_path),
            "--raw-zip", str(bad),
            "--output-root", str(tmp_path / "corpora"),
        ]
    )

    assert exit_code == 1
    assert "不是 ZIP" in capsys.readouterr().err
    assert not os.path.exists(str(tmp_path / "corpora")) or os.listdir(tmp_path / "corpora") == []


def test_snapshot_reports_empty_selection(tmp_path, capsys):
    exit_code = cli.main(
        [
            "snapshot",
            "--config", write_config(tmp_path, category_prefix="不存在的類別前綴"),
            "--raw-zip", write_raw_zip(tmp_path),
            "--output-root", str(tmp_path / "corpora"),
        ]
    )

    assert exit_code == 1
    assert "挑選結果為空" in capsys.readouterr().err
