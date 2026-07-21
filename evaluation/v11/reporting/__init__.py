"""Slice 8：Read-only Optimizer Reporting（供 UI / Read-only API 使用）。

本套件**只讀**：以 SQLite ``mode=ro`` 開啟既有 Search / Review / Deployment DB，
永不執行 migration、永不 ``CREATE TABLE``、DB 不存在時**不建立**檔案（§六 / §十四）。
"""
from evaluation.v11.reporting.readonly import (
    REASON_NOT_AVAILABLE,
    REASON_SCHEMA_INCOMPATIBLE,
    OptimizerDataUnavailable,
    OptimizerReportingStore,
    ReadOnlyDeploymentRepository,
    ReadOnlyReviewRepository,
    ReadOnlySearchRepository,
)

__all__ = [
    "REASON_NOT_AVAILABLE",
    "REASON_SCHEMA_INCOMPATIBLE",
    "OptimizerDataUnavailable",
    "OptimizerReportingStore",
    "ReadOnlyDeploymentRepository",
    "ReadOnlyReviewRepository",
    "ReadOnlySearchRepository",
]
