#!/usr/bin/env bash
# 容器內一鍵部署腳本
# 用法：把整個專案目錄複製到容器內，cd 進去，執行：
#   bash deploy.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "==> [1/6] 檢查 docker / docker compose"
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker 沒安裝" >&2; exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: docker compose plugin 沒裝" >&2; exit 1
fi

echo "==> [2/6] 準備 .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    已從 .env.example 產生 .env（請視需要編輯 LLM_* 設定）"
else
  echo "    .env 已存在，略過"
fi

echo "==> [3/6] 建立 image（首次約 3-8 分鐘）"
docker compose build

echo "==> [4/6] 啟動服務（背景）"
docker compose up -d

echo "==> [5/6] 等待服務就緒（最多 180 秒）"
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "    /health 已回應"
    break
  fi
  echo "    等待中... ($i/60)"
  sleep 3
done

echo "==> [6/6] 顯示狀態"
docker compose ps
echo
echo "--- /health ---"
curl -s http://localhost:8000/health || echo "(尚未就緒)"
echo
echo "--- /ready ---"
curl -s http://localhost:8000/ready || echo "(尚未就緒)"
echo
echo
echo "部署完成。"
echo "  - API:    http://localhost:8000"
echo "  - Qdrant: http://localhost:6333"
echo "  - Log:    docker compose logs -f api"
