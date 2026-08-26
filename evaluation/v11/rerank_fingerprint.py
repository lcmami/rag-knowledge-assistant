"""Rerank Fingerprint 與 Cache Key（v1.2.1 Slice R0 §0(3) / §0.9(4)）。

比照 :mod:`evaluation.v11.embedding_fingerprint`,但針對 Cross-Encoder reranker,並補上兩處
embedding fingerprint 沒有處理的問題:

1. **多 shard 權重**（§0.9(4)）：單一 ``model.safetensors`` 的假設對較大的 reranker
   （如 ``bge-reranker-v2-m3``）不成立。改以 deterministic **weights manifest** 辨識完整模型
   bytes —— 列舉所有權重檔與 index 檔,各記 ``{relative_posix_path, size_bytes, sha256}``,
   **依 relative posix path 排序**（不依賴檔案系統列舉順序,跨 OS deterministic）,
   再對 canonical JSON 取 SHA-256 得 :attr:`weights_manifest_sha256`。

2. **Activation semantics**（§0.9(1)）：``sentence_transformers.CrossEncoder.predict()`` 在
   ``num_labels == 1`` 時**預設套用 Sigmoid**。本專案的排序主鍵是 **raw logit**,因此
   fingerprint 明確記錄實際採用的 activation;若日後改為快取已激活值,fingerprint hash 會改變,
   舊 cache entry 自動失效,不會靜默混用兩種語意的分數。

Cache Key：``SHA256(key_algo_version + cache_schema_version + fingerprint_hash +
sha256(query) + sha256(chunk_text))``。**只**取決於模型 fingerprint 與實際送進 reranker 的
兩段 UTF-8 文字,**不含** threshold / top_k / rerank_top_n / batch size / trial_id / collection
（§0(2)：batch size 是純執行參數,經 R0 batch invariance gate 驗證後不進 key）。

安全:不含 API key;不含本機絕對路徑（model name 走
:func:`evaluation.v11.embedding_fingerprint.stable_model_identifier`,manifest 只存 relative 路徑）。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from evaluation.v11.embedding_fingerprint import (
    REVISION_UNKNOWN,
    stable_model_identifier,
)
from evaluation.v11.hashing import canonical_json

# Cache schema 版本（納入 fingerprint 與 cache key）。
RERANK_CACHE_SCHEMA_VERSION = "v1"
# Cache Key 推導演算法版本。
RERANK_CACHE_KEY_ALGO_VERSION = "v1"

# fingerprint 分隔用不可見字元（避免欄位串接歧義;與 embedding fingerprint 一致）。
_SEP = "\x1f"

#: 本專案排序主鍵所需的 activation —— raw logit,不套 sigmoid/softmax（§0.9(1)）。
ACTIVATION_IDENTITY = "identity"

MANIFEST_UNKNOWN = "unknown"

#: 納入 weights manifest 的檔名樣式（§0.9(4).1）。index 檔本身也是模型 bytes 的一部分。
_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    """reranker input text 的 exact UTF-8 SHA-256（不依賴 python ``hash()``）。"""
    return _sha256(text)


def _file_sha256(path: str, *, chunk_size: int = 1 << 20) -> str:
    """大檔案分塊 hash（權重檔可達數 GB,不可一次讀進記憶體）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _is_weight_file(name: str) -> bool:
    if name in _WEIGHT_INDEX_NAMES:
        return True
    if not name.endswith(_WEIGHT_SUFFIXES):
        return False
    # 只收模型權重,排除 tokenizer 等非權重的 .bin（保守:僅認可已知前綴）。
    return name.endswith(".safetensors") or name.startswith("pytorch_model")


# --------------------------------------------------------------------------
# Weights manifest（§0.9(4)）
# --------------------------------------------------------------------------
def build_weights_manifest(model_dir: Optional[str]) -> list[dict[str, Any]]:
    """列舉模型目錄的權重檔,回傳 deterministic manifest。

    - 遞迴列舉（分片模型可能置於子目錄）。
    - 每筆記 ``{relative_posix_path, size_bytes, sha256}``,**只存 relative 路徑**。
    - **依 ``relative_posix_path`` 排序**,不依賴 ``os.walk`` 的列舉順序 → 跨 OS deterministic。

    目錄不存在或無權重檔 → 回傳空 list（呼叫端據此記為 ``unknown``,不假裝已辨識）。
    """
    if not model_dir or not os.path.isdir(model_dir):
        return []

    entries: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(model_dir):
        for name in files:
            if not _is_weight_file(name):
                continue
            abs_path = os.path.join(root, name)
            rel = os.path.relpath(abs_path, model_dir).replace(os.sep, "/")
            try:
                size = os.path.getsize(abs_path)
                digest = _file_sha256(abs_path)
            except OSError:
                # 讀不到就跳過;寧可 manifest 不完整並如實反映,也不要編造。
                continue
            entries.append(
                {"relative_posix_path": rel, "size_bytes": size, "sha256": digest}
            )

    entries.sort(key=lambda e: e["relative_posix_path"])
    return entries


def weights_manifest_sha256(manifest: Iterable[dict[str, Any]]) -> str:
    """manifest 的 deterministic 摘要;空 manifest → ``unknown``（不編造 hash）。"""
    items = list(manifest)
    if not items:
        return MANIFEST_UNKNOWN
    return _sha256(canonical_json(items))


def _name_or_path_candidates(model: Any) -> list[str]:
    out: list[str] = []
    for obj in (
        getattr(model, "model_card_data", None),
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(model, "model", None),
    ):
        for attr in ("_name_or_path", "name_or_path"):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val and val not in out:
                out.append(val)
    return out


def resolve_model_dir(model: Any) -> Optional[str]:
    """best-effort 取得已載入模型的本機目錄（供 manifest 使用）。

    ``_name_or_path`` 常是 HF **repo id**（如 ``BAAI/bge-reranker-base``）而非本機路徑,
    此時改由 ``huggingface_hub`` 以 ``local_files_only=True`` 解析出 snapshot 目錄
    （純本機查表,不連網）。
    """
    candidates = _name_or_path_candidates(model)

    # 1. 已經是本機目錄。
    for path in candidates:
        if os.path.isdir(path):
            return str(path)

    # 2. 視為 repo id,查本機 HF cache 的 snapshot 目錄（純本機查表,不連網）。
    #    刻意不用 snapshot_download(local_files_only=True) —— 它對「部分下載」的
    #    snapshot 會丟 IncompleteSnapshotError,而我們只需要權重檔所在目錄,
    #    刻意略過 onnx/README 等無關檔案是正常且預期的狀態。
    for repo_id in candidates:
        if "/" not in repo_id or os.path.isabs(repo_id):
            continue
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:  # noqa: BLE001
            break
        for probe in ("config.json", "model.safetensors", "pytorch_model.bin"):
            try:
                hit = try_to_load_from_cache(repo_id, probe)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(hit, str) and os.path.isfile(hit):
                return os.path.dirname(hit)
    return None


def revision_from_snapshot_dir(model_dir: Optional[str]) -> Optional[str]:
    """HF cache 的 snapshot 目錄名即 commit sha（``.../snapshots/<sha>``),離線可讀。"""
    if not model_dir:
        return None
    parent = os.path.basename(os.path.dirname(os.path.normpath(model_dir)))
    if parent != "snapshots":
        return None
    sha = os.path.basename(os.path.normpath(model_dir))
    return sha or None


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RerankFingerprint:
    """Cross-Encoder reranker 的 deterministic 指紋。

    ``weights_manifest_sha256`` / ``weights_shard_count`` 為全域統一欄位名
    （§0.9(4).5：不再使用舊名 ``weights_sha256``）。
    """

    provider: str
    model_name: str
    model_revision: str
    num_labels: int
    max_length: Optional[int]
    #: 實際用於產生被快取分數的 activation。本專案一律 ``identity``（raw logit,§0.9(1)）。
    scored_activation: str
    weights_manifest_sha256: str
    weights_shard_count: int
    cache_schema_version: str = RERANK_CACHE_SCHEMA_VERSION

    def to_canonical_dict(self) -> dict[str, Any]:
        """deterministic、不含敏感資訊的 fingerprint（供 hash 與 Report）。"""
        return {
            "provider": self.provider,
            "model_name": stable_model_identifier(self.model_name),
            "model_revision": self.model_revision or REVISION_UNKNOWN,
            "num_labels": int(self.num_labels),
            "max_length": self.max_length,
            "scored_activation": self.scored_activation,
            "weights_manifest_sha256": self.weights_manifest_sha256,
            "weights_shard_count": int(self.weights_shard_count),
            "cache_schema_version": self.cache_schema_version,
        }

    def fingerprint_hash(self) -> str:
        return _sha256(canonical_json(self.to_canonical_dict()))

    def cache_key(self, query: str, chunk_text: str) -> str:
        """只依 fingerprint 與實際送進 reranker 的兩段文字（不含 batch size / top_k）。"""
        payload = _SEP.join(
            [
                RERANK_CACHE_KEY_ALGO_VERSION,
                self.cache_schema_version,
                self.fingerprint_hash(),
                text_sha256(query),
                text_sha256(chunk_text),
            ]
        )
        return _sha256(payload)

    # ------------------------------------------------------------------
    @classmethod
    def from_cross_encoder(
        cls,
        cross_encoder: Any,
        *,
        model_name: str,
        provider: str = "sentence_transformers",
        scored_activation: str = ACTIVATION_IDENTITY,
        pinned_revision: Optional[str] = None,
        cache_schema_version: str = RERANK_CACHE_SCHEMA_VERSION,
    ) -> tuple["RerankFingerprint", list[dict[str, Any]]]:
        """由已載入的 ``CrossEncoder`` 建立 fingerprint,並一併回傳完整 manifest。

        回傳 manifest 是為了讓 R0 baseline 能完整稽核（fingerprint 本身只存摘要）。
        Revision 解析順序見 §0(3)：pinned → snapshot sha → ``config._commit_hash``
        → weights manifest → ``unknown``。
        """
        model_dir = resolve_model_dir(cross_encoder)
        manifest = build_weights_manifest(model_dir)
        manifest_hash = weights_manifest_sha256(manifest)

        revision = (
            pinned_revision
            or revision_from_snapshot_dir(model_dir)
            or _read_commit_hash(cross_encoder)
            or (manifest_hash if manifest_hash != MANIFEST_UNKNOWN else None)
            or REVISION_UNKNOWN
        )

        fp = cls(
            provider=provider,
            model_name=model_name,
            model_revision=str(revision),
            num_labels=int(_read_num_labels(cross_encoder)),
            max_length=_read_max_length(cross_encoder),
            scored_activation=scored_activation,
            weights_manifest_sha256=manifest_hash,
            weights_shard_count=len(manifest),
            cache_schema_version=cache_schema_version,
        )
        return fp, manifest


def _read_commit_hash(cross_encoder: Any) -> Optional[str]:
    for attr in ("config", "model_card_data"):
        obj = getattr(cross_encoder, attr, None)
        rev = getattr(obj, "_commit_hash", None) or getattr(
            obj, "base_model_revision", None
        )
        if rev:
            return str(rev)
    inner = getattr(cross_encoder, "model", None)
    rev = getattr(getattr(inner, "config", None), "_commit_hash", None)
    return str(rev) if rev else None


def _read_num_labels(cross_encoder: Any) -> int:
    explicit = getattr(cross_encoder, "num_labels", None)
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    cfg = getattr(getattr(cross_encoder, "model", None), "config", None)
    val = getattr(cfg, "num_labels", None)
    return int(val) if isinstance(val, int) and val > 0 else 1


def _read_max_length(cross_encoder: Any) -> Optional[int]:
    for src in (cross_encoder, getattr(cross_encoder, "tokenizer", None)):
        val = getattr(src, "max_length", None) or getattr(
            src, "model_max_length", None
        )
        # tokenizer 未設上限時常見哨兵值(極大整數)，視為未知而非真實上限。
        if isinstance(val, int) and 0 < val < 1_000_000:
            return val
    return None


def make_fingerprint(
    *,
    provider: str,
    model_name: str,
    model_revision: Optional[str],
    num_labels: int,
    max_length: Optional[int],
    scored_activation: str = ACTIVATION_IDENTITY,
    weights_manifest_sha256_value: str = MANIFEST_UNKNOWN,
    weights_shard_count: int = 0,
    cache_schema_version: str = RERANK_CACHE_SCHEMA_VERSION,
) -> RerankFingerprint:
    """明確欄位建立 fingerprint（供單元測試,不載入模型）。"""
    return RerankFingerprint(
        provider=provider,
        model_name=model_name,
        model_revision=model_revision or REVISION_UNKNOWN,
        num_labels=num_labels,
        max_length=max_length,
        scored_activation=scored_activation,
        weights_manifest_sha256=weights_manifest_sha256_value,
        weights_shard_count=weights_shard_count,
        cache_schema_version=cache_schema_version,
    )
