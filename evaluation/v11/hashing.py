"""集中式、deterministic hashing 與環境資訊（Slice 2 §八）。

本模組是 Optimizer 可重現性的單一真相來源：

- :func:`canonical_json`：穩定 JSON 序列化（key 排序、穩定 UTF-8、穩定 separators）。
- :func:`corpus_manifest` / :func:`corpus_hash`：受追蹤 demo corpus 的檔案清單與 hash。
- :func:`dataset_hash`：v1.1 dataset 的 canonical JSON hash。
- :func:`config_hash`：完整 Trial Config 的 canonical JSON hash。
- :func:`git_state`：``code_commit`` 與 ``working_tree_dirty``。

設計原則（§八）：
- **不得**把絕對路徑、檔案 mtime、OS 路徑分隔符納入 hash。
- 相同語意（key 順序不同、Windows/POSIX 相對路徑表示）必須產生相同 hash。
- 任一實質欄位（chunk config / threshold / top_k / candidate 參數 / 版本）改變 →
  必產生不同 config hash。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Optional

# corpus 收錄的文件副檔名（與 IndexingService 支援集合一致）。
SUPPORTED_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_root() -> str:
    """factory-rag-assistant 專案根目錄（此檔位於 evaluation/v11/ 之下兩層）。"""
    return _REPO_ROOT


# --------------------------------------------------------------------------
# canonical JSON
# --------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """穩定 JSON 序列化：key 排序、穩定 UTF-8、穩定 separators。

    相同語意物件（欄位順序不同）必得相同字串，故其 SHA-256 亦相同。
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# corpus hash（受追蹤 demo corpus：sample_docs/ + evaluation/fixtures/）
# --------------------------------------------------------------------------
def collect_corpus_files(
    corpus_paths: list[str] | tuple[str, ...],
    base_dir: Optional[str] = None,
) -> list[tuple[str, str]]:
    """收集 corpus 檔案，回傳 ``[(relative_posix_path, absolute_path)]``，依相對路徑排序。

    - ``corpus_paths`` 可為相對（相對 ``base_dir``）或絕對路徑之目錄。
    - 相對路徑一律以 POSIX ``/`` 表示（跨 OS 一致），故 Windows / POSIX 產生相同集合。
    - 只收錄 :data:`SUPPORTED_EXTENSIONS`；遞迴子目錄。
    - **不含**絕對路徑於回傳的第一元素（供 hash）；第二元素僅供讀取內容。
    """
    base = base_dir or _REPO_ROOT
    collected: dict[str, str] = {}
    for raw in corpus_paths:
        d = raw if os.path.isabs(raw) else os.path.join(base, raw)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"corpus 目錄不存在：{raw}")
        for dirpath, _dirnames, filenames in os.walk(d):
            for name in filenames:
                if not name.lower().endswith(SUPPORTED_EXTENSIONS):
                    continue
                abs_path = os.path.join(dirpath, name)
                rel = os.path.relpath(abs_path, base).replace(os.sep, "/")
                collected[rel] = abs_path
    if not collected:
        raise FileNotFoundError(f"corpus 目錄無任何支援檔案：{list(corpus_paths)}")
    return sorted(collected.items(), key=lambda kv: kv[0])


def corpus_manifest(
    corpus_paths: list[str] | tuple[str, ...],
    base_dir: Optional[str] = None,
) -> dict[str, Any]:
    """建立 corpus 檔案清單（相對路徑 + 每檔 SHA-256 + bytes）與整體 corpus hash。

    corpus hash 僅由「排序後相對路徑 + 檔案內容 SHA-256」組成，與列舉順序、
    絕對路徑、mtime、OS 分隔符皆無關。
    """
    files = collect_corpus_files(corpus_paths, base_dir)
    entries: list[dict[str, Any]] = []
    for rel, abs_path in files:
        with open(abs_path, "rb") as f:
            raw = f.read()
        entries.append(
            {"path": rel, "sha256": _sha256_bytes(raw), "bytes": len(raw)}
        )
    payload = [{"path": e["path"], "sha256": e["sha256"]} for e in entries]
    return {
        "documents": entries,
        "document_count": len(entries),
        "corpus_hash": _sha256_text(canonical_json(payload)),
    }


def corpus_hash(
    corpus_paths: list[str] | tuple[str, ...],
    base_dir: Optional[str] = None,
) -> str:
    return corpus_manifest(corpus_paths, base_dir)["corpus_hash"]


# --------------------------------------------------------------------------
# dataset hash / config hash
# --------------------------------------------------------------------------
def dataset_hash(dataset: dict[str, Any]) -> str:
    """對 v1.1 dataset dict 做 canonical JSON 序列化後 hash。"""
    return _sha256_text(canonical_json(dataset))


def config_hash(config: dict[str, Any]) -> str:
    """對完整 Trial Config dict 做 canonical JSON 序列化後 hash。

    相同語意 config（欄位順序不同）→ 相同 hash；任一實質欄位不同 → 不同 hash。
    """
    return _sha256_text(canonical_json(config))


# --------------------------------------------------------------------------
# git 狀態
# --------------------------------------------------------------------------
def git_state(cwd: Optional[str] = None) -> dict[str, Any]:
    """回傳 ``{"commit", "branch", "working_tree_dirty"}``；git 不可用時值為 None。"""
    root = cwd or _REPO_ROOT

    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:  # noqa: BLE001 - git 不可用時安全退化
            return None
        return None

    commit = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": bool(status) if status is not None else None,
    }
