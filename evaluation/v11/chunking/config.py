"""版本化 Chunking Config（v1.1 Slice 3 §六 / §十三）。

嚴格驗證、deterministic canonical 序列化的 Chunking 設定。Config Hash 由
:mod:`evaluation.v11.hashing` 對本模組輸出的 canonical dict 計算，故：

- 任一有效 chunk 欄位改變 → canonical dict 改變 → config hash 改變（§六.8）。
- 相同語意（JSON key 順序不同）→ 相同 canonical dict → 相同 hash（§六.9）。
- **不含**任何絕對路徑（§六.10）。

欄位效果矩陣（§六.6：不適用欄位不得被默默忽略——此處對每個 strategy 明確**定義或拒絕**）：

===========================  =======  =====  =========  =======
field                        current  fixed  paragraph  heading
===========================  =======  =====  =========  =======
target_chars                 用       用     用         用（每 section）
overlap_chars                用       用     用         用（section 內）
max_tokens                   用       用     用         用
prepend_section_title        拒絕¹    用     拒絕²      用
context_before_paragraphs    拒絕³    用     用         用
context_after_paragraphs     拒絕³    用     用         用
===========================  =======  =====  =========  =======

¹ ² ³ ``current`` 必須與 v1.0 TextSplitter byte-identical，故不接受 prefix / context；
``paragraph`` 未定義 section-title prefix 語意，故拒絕（改用 ``heading``）。被拒絕欄位若出現且
為非預設值即報錯，**不**默默忽略。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CHUNK_CONFIG_SCHEMA_VERSION = "1.0"

# Section title prefix 格式版本（§十三.4/5：deterministic、版本化，納入 config hash）。
# 格式：``[SECTION] <title>\n``；bump 此版本會改變 embedding text → 必 cache miss（§十三.6）。
PREFIX_FORMAT_VERSION = "v1"
_PREFIX_TEMPLATE = "[SECTION] {title}\n"

STRATEGY_CURRENT = "current"
STRATEGY_FIXED = "fixed"
STRATEGY_PARAGRAPH = "paragraph"
STRATEGY_HEADING = "heading"

SUPPORTED_STRATEGIES = (
    STRATEGY_CURRENT,
    STRATEGY_FIXED,
    STRATEGY_PARAGRAPH,
    STRATEGY_HEADING,
)

# 每個 field 的預設值。
_DEFAULTS: dict[str, Any] = {
    "target_chars": 600,
    "overlap_chars": 100,
    "max_tokens": 480,
    "prepend_section_title": False,
    "context_before_paragraphs": 0,
    "context_after_paragraphs": 0,
}

# 允許出現的 key（unknown key 一律拒絕，§六避免默默忽略）。
# ``prefix_format_version`` 為 to_canonical_dict 產生的**衍生**欄位：接受它以便 canonical dict
# 可 round-trip，但其值由程式常數決定（輸入值忽略、由 to_canonical_dict 重新寫入當前版本）。
_ALLOWED_KEYS = {"schema_version", "strategy", "prefix_format_version", *(_DEFAULTS.keys())}

# section-title prefix 適用的 strategy（其餘 strategy 若 prepend_section_title=True 即拒絕）。
_PREFIX_STRATEGIES = {STRATEGY_FIXED, STRATEGY_HEADING}
# 相鄰段落 context 適用的 strategy（current 須維持 v1.0 byte-identical，故拒絕）。
_CONTEXT_STRATEGIES = {STRATEGY_FIXED, STRATEGY_PARAGRAPH, STRATEGY_HEADING}


class ChunkConfigError(ValueError):
    """Chunking Config 驗證錯誤（訊息需可理解、指出違反的規則）。"""


def section_title_prefix(title: str) -> str:
    """回傳版本化、deterministic 的 section-title prefix（§十三.4）。

    title 為 None / 空白時回傳空字串（§十一.6：不得加入 "None" / 空白 / 虛構標題）。
    """
    if title is None:
        return ""
    t = title.strip()
    if not t:
        return ""
    return _PREFIX_TEMPLATE.format(title=t)


@dataclass(frozen=True)
class ChunkingConfig:
    """單一 Chunking Strategy 的嚴格驗證、版本化設定。"""

    strategy: str
    target_chars: int = _DEFAULTS["target_chars"]
    overlap_chars: int = _DEFAULTS["overlap_chars"]
    max_tokens: int = _DEFAULTS["max_tokens"]
    prepend_section_title: bool = _DEFAULTS["prepend_section_title"]
    context_before_paragraphs: int = _DEFAULTS["context_before_paragraphs"]
    context_after_paragraphs: int = _DEFAULTS["context_after_paragraphs"]
    schema_version: str = CHUNK_CONFIG_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # 建構 / 驗證
    # ------------------------------------------------------------------
    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "ChunkingConfig":
        """由 dict 建立並嚴格驗證。``None`` / 空 dict → 預設 ``current`` config。"""
        raw = dict(raw or {})
        raw.setdefault("strategy", STRATEGY_CURRENT)

        unknown = set(raw) - _ALLOWED_KEYS
        if unknown:
            raise ChunkConfigError(
                f"chunk_config 含未知欄位（不得默默忽略）：{sorted(unknown)}；"
                f"允許欄位：{sorted(_ALLOWED_KEYS)}。"
            )

        strategy = raw["strategy"]
        if strategy not in SUPPORTED_STRATEGIES:
            raise ChunkConfigError(
                f"不支援的 chunk strategy={strategy!r}；支援：{list(SUPPORTED_STRATEGIES)}。"
            )

        def _int(key: str) -> int:
            val = raw.get(key, _DEFAULTS[key])
            if isinstance(val, bool) or not isinstance(val, int):
                raise ChunkConfigError(f"{key} 必須為整數，得到 {val!r}。")
            return val

        def _bool(key: str) -> bool:
            val = raw.get(key, _DEFAULTS[key])
            if not isinstance(val, bool):
                raise ChunkConfigError(f"{key} 必須為布林值，得到 {val!r}。")
            return val

        cfg = cls(
            strategy=strategy,
            target_chars=_int("target_chars"),
            overlap_chars=_int("overlap_chars"),
            max_tokens=_int("max_tokens"),
            prepend_section_title=_bool("prepend_section_title"),
            context_before_paragraphs=_int("context_before_paragraphs"),
            context_after_paragraphs=_int("context_after_paragraphs"),
            schema_version=str(raw.get("schema_version", CHUNK_CONFIG_SCHEMA_VERSION)),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        # 數值約束（§六.2/3/4）：不允許負數；overlap < target；max_tokens > 0。
        if self.target_chars <= 0:
            raise ChunkConfigError("target_chars 必須 > 0。")
        if self.overlap_chars < 0:
            raise ChunkConfigError("overlap_chars 不得為負。")
        if self.overlap_chars >= self.target_chars:
            raise ChunkConfigError(
                f"overlap_chars ({self.overlap_chars}) 必須 < target_chars ({self.target_chars})。"
            )
        if self.max_tokens <= 0:
            raise ChunkConfigError("max_tokens 必須 > 0。")
        if self.context_before_paragraphs < 0 or self.context_after_paragraphs < 0:
            raise ChunkConfigError("context_before/after_paragraphs 不得為負。")

        # 適用性（§六.6）：不適用欄位若為非預設值即明確拒絕。
        if self.prepend_section_title and self.strategy not in _PREFIX_STRATEGIES:
            raise ChunkConfigError(
                f"strategy={self.strategy!r} 不支援 prepend_section_title=true；"
                f"僅 {sorted(_PREFIX_STRATEGIES)} 支援 section-title prefix。"
            )
        has_context = self.context_before_paragraphs or self.context_after_paragraphs
        if has_context and self.strategy not in _CONTEXT_STRATEGIES:
            raise ChunkConfigError(
                f"strategy={self.strategy!r} 不支援相鄰段落 context；"
                f"僅 {sorted(_CONTEXT_STRATEGIES)} 支援。"
            )

    # ------------------------------------------------------------------
    # canonical 序列化（供 config hash / 持久化 / 還原）
    # ------------------------------------------------------------------
    def to_canonical_dict(self) -> dict[str, Any]:
        """deterministic、可還原的 config dict（不含絕對路徑）。

        永遠包含所有欄位（即使為預設值），故任一欄位改變都會反映於 config hash；並包含
        ``prefix_format_version``（§十三.5：prefix 格式版本納入 config hash——啟用 prefix 時
        影響 embedding text，故其版本也必須進 hash）。
        """
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "target_chars": self.target_chars,
            "overlap_chars": self.overlap_chars,
            "max_tokens": self.max_tokens,
            "prepend_section_title": self.prepend_section_title,
            "context_before_paragraphs": self.context_before_paragraphs,
            "context_after_paragraphs": self.context_after_paragraphs,
            "prefix_format_version": PREFIX_FORMAT_VERSION,
        }
