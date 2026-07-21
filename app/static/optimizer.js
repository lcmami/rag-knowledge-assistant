"use strict";

// Retrieval Optimizer 前端（Slice 8）：純唯讀報表 + 安全的 typed-confirmation 管理流程。
// 前端不啟動長時間 Search、不批次接受、不自動 Apply/Rollback、不直接呼叫 Qdrant。
// Apply eligibility 一律以後端重新讀 DB 為準；前端不相信本地 apply_eligible boolean（§六.9）。
(function () {
  // ---- 小工具（與 app.js 解耦，方便 DOM 單元測試單獨載入本檔）----
  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }
  function shortHash(h, n) {
    if (!h) return "—";
    const short = String(h).slice(0, n || 12);
    return '<code class="hash" title="' + esc(h) + '">' + esc(short) + "</code>";
  }
  function fmt(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    return esc(v);
  }
  // 狀態徽章：文字為主，顏色為輔（§十七.4 不只靠顏色）。
  function badge(text, kind) {
    return '<span class="badge badge-' + (kind || "neutral") + '">' + esc(text) + "</span>";
  }
  function yesNo(v, yesText, noText) {
    return v ? (yesText || "是") : (noText || "否");
  }
  function unavailableBlock(block, label) {
    if (block && block.configured === false) {
      return '<div class="opt-empty" role="status">尚未設定 ' + esc(label) +
        ' 資料來源（optimizer_data_not_available）。</div>';
    }
    const reason = (block && block.reason) || "optimizer_data_not_available";
    return '<div class="opt-empty" role="status">' + esc(label) +
      " 資料無法讀取：" + esc(reason) + "。</div>";
  }
  function isAvailable(block) {
    return block && block.available === true;
  }

  async function fetchJson(url, opts) {
    const res = await fetch(url, opts);
    let body = null;
    try {
      body = await res.json();
    } catch (_) {
      /* 非 JSON */
    }
    return { ok: res.ok, status: res.status, body: body };
  }

  // ==================================================================
  // 純渲染函式（回傳 HTML 字串）——供 DOM 單元測試直接呼叫。
  // ==================================================================
  const UI = {};

  UI.demoBanner = function (isDemo) {
    if (!isDemo) return "";
    return '<div class="opt-demo-banner" role="note"><strong>Demo fixture</strong>：' +
      "以下為展示用資料，非真實 production run。</div>";
  };

  // Known limitations 區塊：通用地渲染 Search Detail API 的 outcome.known_limitations
  // （§一）。不針對任何特定 case 硬編碼；缺失／空陣列時顯示 Empty State，不 crash；
  // 文字完整可見（非 tooltip），並以 role="note" 標示語意。
  UI.renderKnownLimitations = function (limitations) {
    const items = (Array.isArray(limitations) ? limitations : [])
      .filter(function (x) { return x != null && String(x).trim() !== ""; })
      .map(function (x) { return String(x); });
    let body;
    if (!items.length) {
      body = '<p class="opt-muted" data-testid="known-limitations-empty">' +
        "此 search 未記錄任何 known limitation。</p>";
    } else {
      body = '<ul class="opt-limitations-list">' +
        items.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("") + "</ul>";
    }
    return '<div class="opt-card opt-limitations" role="note" aria-label="Known limitations"' +
      ' data-testid="known-limitations"><h3>Known limitations</h3>' + body + "</div>";
  };

  UI.renderOverview = function (data) {
    const s = data.search || {};
    const r = data.review || {};
    const d = data.deployment || {};
    const parts = [];
    parts.push(UI.demoBanner(data.is_demo_fixture));

    // -- Dataset / Corpus / Coverage --
    parts.push('<div class="opt-card"><h3>Dataset 與覆蓋</h3>');
    if (!isAvailable(s)) {
      parts.push(unavailableBlock(s, "Search"));
    } else {
      const sc = s.split_counts || {};
      parts.push('<dl class="opt-kv">');
      parts.push(kv("Dataset Version", fmt(s.dataset_version)));
      parts.push(kv("Corpus Hash", shortHash(s.corpus_hash_short, 12)));
      parts.push(kv("Case / Group 數", fmt(s.case_count) + " / " + fmt(s.group_count)));
      parts.push(kv("Tune / Validation / Test",
        fmt(sc.tune) + " / " + fmt(sc.validation) + " / " + fmt(sc.test)));
      parts.push("</dl>");
      const warns = s.coverage_warnings || [];
      if (warns.length) {
        parts.push('<div class="opt-warn" role="status"><strong>Coverage warnings：</strong><ul>');
        warns.forEach(function (w) { parts.push("<li>" + esc(w) + "</li>"); });
        parts.push("</ul></div>");
      } else {
        parts.push('<p class="opt-muted">無 coverage warning。</p>');
      }
    }
    parts.push("</div>");

    // -- Latest search / recommendation --
    parts.push('<div class="opt-card"><h3>最新 Search 與 Recommendation</h3>');
    const ls = s.latest_search;
    if (!isAvailable(s) || !ls) {
      parts.push('<div class="opt-empty" role="status">尚無 search 結果。</div>');
    } else {
      parts.push('<dl class="opt-kv">');
      parts.push(kv("Search Status", badge(ls.status, statusKind(ls.status))));
      parts.push(kv("Executable / Feasible / Pareto",
        fmt(ls.executable_candidate_count) + " / " + fmt(ls.feasible_candidate_count) +
        " / " + fmt(ls.pareto_count)));
      parts.push(kv("Recommended Config", shortHash(ls.recommended_config_hash_short, 12)));
      parts.push(kv("Recommendation Locked",
        ls.recommendation_locked ? badge("Recommendation locked", "ok")
          : badge("未鎖定", "warn")));
      parts.push(kv("Apply Eligible", applyBadge(ls.apply_eligible)));
      parts.push(kv("Confidence", fmt(ls.confidence)));
      parts.push(kv("Test Constraint",
        badge("Test constraints " + (ls.test_constraint_status || "n/a"),
          ls.test_constraint_status === "passed" ? "ok" : "danger")));
      parts.push(kv("Baseline Comparison", fmt(ls.baseline_comparison_status)));
      parts.push(kv("Selection Basis", fmt(ls.recommendation_selection_basis)));
      parts.push("</dl>");
      if (ls.apply_eligible === false && (ls.apply_ineligible_reasons || []).length) {
        parts.push(reasonsList("Apply 不符資格原因", ls.apply_ineligible_reasons));
      }
    }
    parts.push("</div>");

    // -- Alias / Admin --
    parts.push('<div class="opt-card"><h3>Alias 與 Admin</h3><dl class="opt-kv">');
    parts.push(kv("Active Alias", "<code>" + esc(d.active_alias || "—") + "</code>"));
    parts.push(kv("Alias Initialized",
      d.alias_initialized ? badge("已初始化", "ok") : badge("尚未初始化", "warn")));
    parts.push(kv("Admin Write",
      d.admin_write_enabled ? badge("Admin write enabled", "warn")
        : badge("Admin write disabled", "neutral")));
    parts.push("</dl></div>");
    return parts.join("");
  };

  UI.renderCandidateTable = function (candidates) {
    if (!candidates || !candidates.length) {
      return '<div class="opt-empty" role="status">無 candidate 資料。</div>';
    }
    const head = ["Candidate", "Chunk", "Retrieval", "Floor", "Threshold", "Boost",
      "Penalty", "Tune", "Validation", "Constraint", "Feasible", "Pareto", "Rec. Rank"];
    const rows = candidates.map(function (c) {
      const cells = [
        (c.is_baseline ? "★ " : "") + esc(c.candidate_id),
        fmt(c.chunk_strategy), fmt(c.retrieval_strategy), fmt(c.identifier_vector_floor),
        fmt(c.score_threshold), fmt(c.exact_match_boost), fmt(c.confusable_penalty),
        fmt(c.tune_status), fmt(c.validation_status),
        c.constraint_status ? badge(c.constraint_status,
          c.constraint_status === "pass" ? "ok" : "danger") : "—",
        c.feasible === true ? badge("feasible", "ok")
          : c.feasible === false ? badge("infeasible", "danger") : "—",
        c.pareto_member ? "✓" : "—",
        fmt(c.recommendation_rank),
      ];
      return "<tr>" + cells.map(function (x) { return "<td>" + x + "</td>"; }).join("") + "</tr>";
    });
    return '<div class="table-wrap"><table class="opt-table"><thead><tr>' +
      head.map(function (h) { return "<th>" + esc(h) + "</th>"; }).join("") +
      "</tr></thead><tbody>" + rows.join("") + "</tbody></table></div>";
  };

  UI.renderMetricsComparison = function (cmp) {
    if (!cmp || !cmp.metrics || !cmp.metrics.length) {
      return '<div class="opt-empty" role="status">無 metrics 可比較。</div>';
    }
    const head = ["Metric", "Scope/Type", "Baseline", "Candidate", "Delta", "Direction",
      "n", "Computable"];
    const rows = cmp.metrics.map(function (m) {
      const dir = m.direction === "lower_is_better" ? "越小越好" : "越大越好";
      let deltaText = m.delta === null || m.delta === undefined ? "—" : m.delta.toFixed(4);
      let deltaKind = "neutral";
      if (m.delta !== null && m.delta !== undefined && m.delta !== 0) {
        const better = (m.direction === "lower_is_better") ? m.delta < 0 : m.delta > 0;
        deltaKind = better ? "ok" : "danger";
      }
      const scope = m.scope + (m.query_type ? "@" + m.query_type : "");
      const computable = (m.baseline_computable && m.candidate_computable)
        ? "computable" : "not computable";
      return "<tr>" + [
        esc(m.metric_name), esc(scope), fmt(m.baseline_value), fmt(m.candidate_value),
        badge(deltaText, deltaKind), esc(dir), fmt(m.sample_count), esc(computable),
      ].map(function (x) { return "<td>" + x + "</td>"; }).join("") + "</tr>";
    });
    return '<div class="table-wrap"><table class="opt-table"><thead><tr>' +
      head.map(function (h) { return "<th>" + esc(h) + "</th>"; }).join("") +
      "</tr></thead><tbody>" + rows.join("") + "</tbody></table></div>";
  };

  UI.renderTop5 = function (rows) {
    if (!rows || !rows.length) return '<div class="opt-muted">（無結果）</div>';
    return rows.map(function (r) {
      const loc = r.page != null ? "第 " + esc(r.page) + " 頁"
        : r.section ? esc(r.section) : "—";
      const rel = r.relevance_level != null ? "rel " + esc(r.relevance_level) : "";
      const ex = (r.exact_identifier_matches && r.exact_identifier_matches.length)
        ? " · exact " + esc(r.exact_identifier_matches.join(",")) : "";
      const src = (r.candidate_sources && r.candidate_sources.length)
        ? " · " + esc(r.candidate_sources.join("+")) : "";
      return '<div class="opt-src-card">' +
        '<div class="opt-src-head"><span class="opt-rank">#' + esc(r.rank) + "</span>" +
        "<span>" + esc(r.source || "—") + "</span>" +
        '<span class="opt-src-scores">vec ' + fmt(r.vector_score != null ? r.vector_score : r.score) +
        (r.ranking_score != null ? " · rank " + fmt(r.ranking_score) : "") + "</span></div>" +
        '<div class="opt-src-meta">' + loc + " · " + rel + ex + src +
        (r.admission_reason ? " · " + esc(r.admission_reason) : "") + "</div></div>";
    }).join("");
  };

  UI.renderPerCaseDiff = function (perCase) {
    if (!perCase || !perCase.cases || !perCase.cases.length) {
      return '<div class="opt-empty" role="status">無 per-case diff。</div>';
    }
    const summary = '<p class="opt-muted">Improved ' + esc(perCase.improved) +
      " · Same " + esc(perCase.same) + " · Regressed " + esc(perCase.regressed) + "</p>";
    const cards = perCase.cases.map(function (c) {
      const verdictKind = c.verdict === "improved" ? "ok"
        : c.verdict === "regressed" ? "danger" : "neutral";
      return '<div class="opt-case">' +
        '<div class="opt-case-head"><strong>' + esc(c.question || c.case_id) + "</strong> " +
        badge(c.verdict, verdictKind) + "</div>" +
        '<div class="opt-case-meta">' + esc(c.case_id) + " · " + esc(c.query_type || "") +
        " · baseline FRR " + fmt(c.baseline_first_relevant_rank) +
        " → candidate FRR " + fmt(c.candidate_first_relevant_rank) + "</div>" +
        '<div class="opt-diff-cols"><div><h5>Baseline Top-5</h5>' +
        UI.renderTop5(c.baseline_top5) + "</div><div><h5>Recommendation Top-5</h5>" +
        UI.renderTop5(c.candidate_top5) + "</div></div></div>";
    });
    return summary + cards.join("");
  };

  UI.renderReviewList = function (sessions) {
    if (!sessions || !sessions.length) {
      return '<div class="opt-empty" role="status">無 review session。</div>';
    }
    const head = ["Session", "Status", "Base Dataset", "Proposed", "Accepted", "Rejected",
      "Published", "Revisions"];
    const rows = sessions.map(function (s) {
      const c = s.counts || {};
      return "<tr>" + [
        esc(s.session_name || s.session_id), badge(s.status, statusKind(s.status)),
        esc(s.base_dataset_version), fmt(c.proposed), fmt(c.accepted), fmt(c.rejected),
        fmt(c.published), fmt(s.revision_count),
      ].map(function (x) { return "<td>" + x + "</td>"; }).join("") + "</tr>";
    });
    return '<div class="table-wrap"><table class="opt-table"><thead><tr>' +
      head.map(function (h) { return "<th>" + esc(h) + "</th>"; }).join("") +
      "</tr></thead><tbody>" + rows.join("") + "</tbody></table></div>";
  };

  UI.renderReviewDetail = function (session, coverage) {
    const parts = [];
    parts.push('<div class="opt-card"><h4>' + esc(session.session_name || session.session_id) +
      "</h4><dl class=\"opt-kv\">");
    const c = session.counts || {};
    parts.push(kv("Status", badge(session.status, statusKind(session.status))));
    parts.push(kv("Base Dataset Version", fmt(session.base_dataset_version)));
    parts.push(kv("Proposed / Accepted / Rejected / Published",
      fmt(c.proposed) + " / " + fmt(c.accepted) + " / " + fmt(c.rejected) + " / " + fmt(c.published)));
    parts.push(kv("Revision Count", fmt(session.revision_count)));
    parts.push("</dl>");
    // 明確聲明：唯讀，無 auto-accept / batch / publish（§十.不得提供）。
    parts.push('<p class="opt-muted">此頁為唯讀顯示；接受／發布須透過 CLI 逐筆人工操作，' +
      "前端不提供 auto-accept／batch accept／自動 publish／LLM 生成按鈕。</p>");
    parts.push("</div>");

    if (coverage && coverage.coverage) {
      const warns = coverage.coverage.coverage_warnings || [];
      if (warns.length) {
        parts.push('<div class="opt-card"><h4>Coverage Gaps</h4>' +
          reasonsList("Coverage warnings", warns) + "</div>");
      }
    }

    const cands = session.candidates || [];
    if (cands.length) {
      const head = ["Question", "Type", "Answerable", "Split", "Status", "Evidence", "Revisions"];
      const rows = cands.map(function (q) {
        return "<tr>" + [
          esc(q.question), esc(q.query_type), yesNo(q.answerable),
          fmt(q.suggested_split), badge(q.status, statusKind(q.status)),
          fmt(q.evidence_count), fmt(q.revision_count),
        ].map(function (x) { return "<td>" + x + "</td>"; }).join("") + "</tr>";
      });
      parts.push('<div class="table-wrap"><table class="opt-table"><thead><tr>' +
        head.map(function (h) { return "<th>" + esc(h) + "</th>"; }).join("") +
        "</tr></thead><tbody>" + rows.join("") + "</tbody></table></div>");
    }
    return parts.join("");
  };

  // -- Deployment view --
  UI.renderDeployment = function (summary, latestSearch, aliasStatus) {
    const parts = [];
    const adminEnabled = summary.admin_write_enabled === true;
    const eligible = latestSearch && latestSearch.apply_eligible === true;
    const locked = latestSearch && latestSearch.recommendation_locked === true;
    const testPassed = latestSearch && latestSearch.test_constraint_status === "passed";
    const canApply = adminEnabled && eligible && locked && testPassed;

    // Admin 狀態
    parts.push('<div class="opt-card"><h3>Administrative Write</h3>');
    if (!adminEnabled) {
      parts.push('<div class="opt-warn" role="status">Administrative write operations are disabled.</div>');
    } else {
      parts.push('<div class="opt-warn" role="status">Admin write enabled（單機 local，無 RBAC）。</div>');
    }
    parts.push("</div>");

    // Alias 狀態
    parts.push('<div class="opt-card"><h3>Alias</h3><dl class="opt-kv">');
    parts.push(kv("Alias Name", "<code>" + esc(summary.active_alias || "—") + "</code>"));
    parts.push(kv("Alias Target (live)",
      aliasStatus && aliasStatus.alias_target ? "<code>" + esc(aliasStatus.alias_target) + "</code>"
        : badge("尚未初始化", "warn")));
    parts.push(kv("Current Collection",
      summary.current_collection ? "<code>" + esc(summary.current_collection) + "</code>" : "—"));
    parts.push(kv("Previous Collection",
      summary.previous_collection ? "<code>" + esc(summary.previous_collection) + "</code>" : "—"));
    parts.push("</dl></div>");

    // Eligibility
    parts.push('<div class="opt-card"><h3>Apply Eligibility</h3>');
    if (eligible) {
      parts.push('<div class="opt-ok" role="status">Recommendation apply eligible。</div>');
    } else {
      parts.push('<div class="opt-warn" role="status" data-testid="apply-blocked"><strong>Apply blocked。</strong></div>');
      const reasons = (latestSearch && latestSearch.apply_ineligible_reasons) || [];
      if (reasons.length) parts.push(reasonsList("Eligibility reasons", reasons));
    }
    parts.push("</div>");

    // 管理操作（Initialize / Apply / Rollback）——依 canApply gate 啟用/停用。
    parts.push('<div class="opt-card"><h3>管理操作</h3>');
    parts.push('<p class="opt-muted">須同時滿足 admin_enabled、apply_eligible、recommendation_locked、' +
      "test_constraint=passed 才可送出；即使前端啟用，後端仍會重新驗證。</p>");
    parts.push(UI.confirmationForm("apply", canApply, {
      alias: summary.active_alias,
      config_hash: latestSearch ? latestSearch.recommended_config_hash : "",
      current_collection: summary.current_collection || "",
    }));
    parts.push('<div class="opt-btn-row">');
    parts.push(actionButton("initialize", "Initialize Alias", adminEnabled && !summary.alias_initialized));
    parts.push(actionButton("rollback", "Rollback", adminEnabled && summary.history_count > 1));
    parts.push(actionButton("reconcile", "Reconcile", adminEnabled));
    parts.push("</div>");
    parts.push("</div>");

    // History
    parts.push('<div class="opt-card"><h3>Deployment History</h3>');
    const ops = summary.operations || [];
    if (!ops.length) {
      parts.push('<div class="opt-empty" role="status">尚無部署歷史。</div>');
    } else {
      const head = ["Operation", "Type", "Status", "Target", "Config", "Created"];
      const rows = ops.map(function (o) {
        return "<tr>" + [
          esc(o.operation_id), esc(o.operation_type), badge(o.status, statusKind(o.status)),
          o.target_collection ? "<code>" + esc(o.target_collection) + "</code>" : "—",
          shortHash(o.config_hash, 10), esc(o.created_at || "—"),
        ].map(function (x) { return "<td>" + x + "</td>"; }).join("") + "</tr>";
      });
      parts.push('<div class="table-wrap"><table class="opt-table"><thead><tr>' +
        head.map(function (h) { return "<th>" + esc(h) + "</th>"; }).join("") +
        "</tr></thead><tbody>" + rows.join("") + "</tbody></table></div>");
    }
    parts.push("</div>");
    return parts.join("");
  };

  // -- 精確 typed confirmation（§十二）--
  // 後端 deterministic 格式（confirmation.py）：
  //   apply：      APPLY <alias> CONFIG <config_hash> FROM <expected_current>
  //   initialize： INITIALIZE <alias> -> <collection>
  //   rollback：   ROLLBACK <alias> FROM <current> TO <target>
  UI.confirmationText = function (op, p) {
    if (op === "apply") return "APPLY " + p.alias + " CONFIG " + p.config_hash + " FROM " + p.current_collection;
    if (op === "initialize") return "INITIALIZE " + p.alias + " -> " + p.collection;
    if (op === "rollback") return "ROLLBACK " + p.alias + " FROM " + p.current_collection + " TO " + p.target;
    return "";
  };
  UI.confirmationMatches = function (input, expected) {
    return (input || "").trim() === (expected || "").trim();
  };

  UI.confirmationForm = function (op, enabled, params) {
    const expected = UI.confirmationText(op, params);
    const dis = enabled ? "" : " disabled";
    return '<form class="opt-confirm" data-op="' + esc(op) + '" data-expected="' + esc(expected) + '">' +
      '<label for="confirm-' + esc(op) + '">請完整輸入下列確認字串以送出 ' + esc(op) + '：</label>' +
      '<div class="opt-confirm-expected"><code>' + esc(expected) + "</code></div>" +
      '<input id="confirm-' + esc(op) + '" class="opt-confirm-input" type="text" autocomplete="off"' +
      ' aria-label="' + esc(op) + " confirmation text\"" + dis + " />" +
      '<button type="submit" class="opt-confirm-submit" data-op="' + esc(op) + '" disabled>送出 ' +
      esc(op) + "</button>" +
      '<div class="opt-confirm-msg msg" role="alert" aria-live="assertive"></div>' +
      "</form>";
  };

  // ---- 內部 helpers ----
  function kv(k, v) { return "<dt>" + esc(k) + "</dt><dd>" + v + "</dd>"; }
  function reasonsList(title, reasons) {
    return '<div class="opt-reasons" role="status"><strong>' + esc(title) + "：</strong><ul>" +
      reasons.map(function (r) { return '<li><code>' + esc(r) + "</code></li>"; }).join("") +
      "</ul></div>";
  }
  function applyBadge(v) {
    if (v === true) return badge("Apply eligible", "ok");
    if (v === false) return badge("Apply not eligible", "danger");
    return badge("n/a", "neutral");
  }
  function actionButton(op, label, enabled) {
    return '<button type="button" class="opt-action" data-op="' + esc(op) + '"' +
      (enabled ? "" : " disabled") + ">" + esc(label) + "</button>";
  }
  function statusKind(status) {
    const s = String(status || "").toLowerCase();
    if (/(passed|completed|published|ok|active|improved|accepted)/.test(s)) return "ok";
    if (/(failed|error|rejected|cancelled|regressed|blocked)/.test(s)) return "danger";
    if (/(reviewing|running|pending|draft|ready)/.test(s)) return "warn";
    return "neutral";
  }
  UI.statusKind = statusKind;
  UI.esc = esc;

  // ==================================================================
  // Controller：載入資料 + 綁定事件
  // ==================================================================
  const state = { loaded: false, overview: null, searchId: null };

  function setSub(name) {
    document.querySelectorAll(".subtab").forEach(function (t) {
      t.classList.toggle("active", t.dataset.sub === name);
    });
    document.querySelectorAll(".subpanel").forEach(function (p) {
      p.classList.toggle("active", p.id === "opt-subpanel-" + name);
    });
    if (name === "search") loadSearch();
    if (name === "review") loadReview();
    if (name === "deploy") loadDeploy();
  }

  async function loadOverview() {
    const el = document.getElementById("opt-subpanel-overview");
    el.innerHTML = '<div class="opt-loading" role="status">載入中…</div>';
    const r = await fetchJson("/api/optimizer/overview");
    if (!r.ok || !r.body) {
      el.innerHTML = '<div class="opt-error" role="alert">Overview 載入失敗（HTTP ' + r.status + "）。</div>";
      return;
    }
    state.overview = r.body;
    el.innerHTML = UI.renderOverview(r.body);
  }

  async function loadSearch() {
    const el = document.getElementById("opt-subpanel-search");
    el.innerHTML = '<div class="opt-loading" role="status">載入中…</div>';
    const list = await fetchJson("/api/optimizer/searches");
    if (!list.ok || !list.body) {
      el.innerHTML = '<div class="opt-error" role="alert">Search 載入失敗（HTTP ' + list.status + "）。</div>";
      return;
    }
    if (list.body.available === false) {
      el.innerHTML = unavailableBlock(list.body, "Search");
      return;
    }
    const searches = list.body.searches || [];
    if (!searches.length) {
      el.innerHTML = '<div class="opt-empty" role="status">尚無 search 結果。</div>';
      return;
    }
    const sid = searches[0].search_id;
    state.searchId = sid;
    const [detail, cands, cmp] = await Promise.all([
      fetchJson("/api/optimizer/searches/" + encodeURIComponent(sid)),
      fetchJson("/api/optimizer/searches/" + encodeURIComponent(sid) + "/candidates"),
      fetchJson("/api/optimizer/searches/" + encodeURIComponent(sid) + "/comparison"),
    ]);
    const cmpBody = (cmp.body && cmp.body.comparison) || null;
    // Known limitations 取自 Search Detail API 的 outcome（唯讀，不在前端重述實驗結論）。
    const searchDetail = (detail.body && detail.body.search) || null;
    const knownLimitations = (searchDetail && searchDetail.outcome &&
      searchDetail.outcome.known_limitations) || [];
    let html = "";
    html += UI.demoBanner(state.overview && state.overview.is_demo_fixture);
    if (cmpBody && cmpBody.baseline_comparison_status) {
      html += '<div class="opt-note opt-muted">Selection basis / latency 穩定性請見 Overview；' +
        "品質平手時單次 P95 winner 不代表品質最佳。</div>";
    }
    html += UI.renderKnownLimitations(knownLimitations);
    html += '<div class="opt-card"><h3>Candidate Table</h3>' +
      UI.renderCandidateTable((cands.body && cands.body.candidates) || []) + "</div>";
    html += '<div class="opt-card"><h3>Metrics Comparison（Baseline vs Recommendation）</h3>' +
      UI.renderMetricsComparison(cmpBody) + "</div>";
    html += '<div class="opt-card"><h3>Per-case Top-5 Diff</h3>' +
      UI.renderPerCaseDiff(cmpBody && cmpBody.per_case) + "</div>";
    el.innerHTML = html;
  }

  async function loadReview() {
    const el = document.getElementById("opt-subpanel-review");
    el.innerHTML = '<div class="opt-loading" role="status">載入中…</div>';
    const list = await fetchJson("/api/optimizer/reviews");
    if (!list.ok || !list.body) {
      el.innerHTML = '<div class="opt-error" role="alert">Review 載入失敗（HTTP ' + list.status + "）。</div>";
      return;
    }
    if (list.body.available === false) {
      el.innerHTML = unavailableBlock(list.body, "Review");
      return;
    }
    const sessions = list.body.sessions || [];
    let html = UI.demoBanner(state.overview && state.overview.is_demo_fixture);
    html += '<div class="opt-card"><h3>Review Sessions</h3>' + UI.renderReviewList(sessions) + "</div>";
    if (sessions.length) {
      const [detail, cov] = await Promise.all([
        fetchJson("/api/optimizer/reviews/" + encodeURIComponent(sessions[0].session_id)),
        fetchJson("/api/optimizer/coverage"),
      ]);
      if (detail.body && detail.body.session) {
        html += UI.renderReviewDetail(detail.body.session, cov.body);
      }
    }
    el.innerHTML = html;
  }

  async function loadDeploy() {
    const el = document.getElementById("opt-subpanel-deploy");
    el.innerHTML = '<div class="opt-loading" role="status">載入中…</div>';
    const [summary, alias] = await Promise.all([
      fetchJson("/api/optimizer/deployment/summary"),
      fetchJson("/api/optimizer/alias/status"),
    ]);
    const latest = (state.overview && state.overview.search && state.overview.search.latest_search) || null;
    let html = UI.demoBanner(state.overview && state.overview.is_demo_fixture);
    html += UI.renderDeployment(summary.body || {}, latest, alias.body || {});
    el.innerHTML = html;
    bindConfirmations(el);
  }

  // 綁定 confirmation 輸入即時驗證 + 送出（含 idempotency key、後端重新驗證、錯誤顯示）。
  function bindConfirmations(root) {
    root.querySelectorAll(".opt-confirm").forEach(function (form) {
      const input = form.querySelector(".opt-confirm-input");
      const submit = form.querySelector(".opt-confirm-submit");
      const expected = form.dataset.expected;
      if (!input) return;
      input.addEventListener("input", function () {
        submit.disabled = input.disabled || !UI.confirmationMatches(input.value, expected);
      });
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        if (!UI.confirmationMatches(input.value, expected)) return;
        await submitConfirmation(form);
      });
    });
  }

  // 每個 form 一次操作只用同一把 idempotency key（重複點擊沿用），除非重新開始操作。
  async function submitConfirmation(form) {
    const op = form.dataset.op;
    const msg = form.querySelector(".opt-confirm-msg");
    const submit = form.querySelector(".opt-confirm-submit");
    const input = form.querySelector(".opt-confirm-input");
    if (!form._idem) form._idem = "ui-" + op + "-" + Date.now();
    const payload = {
      confirmation: input.value.trim(),
      idempotency_key: form._idem,
    };
    submit.disabled = true;
    msg.className = "opt-confirm-msg msg loading";
    msg.textContent = "送出中…";
    // 端點對應（demo 情境下 admin 停用，後端會回 403；仍顯示明確錯誤）。
    const url = op === "apply" ? "/api/optimizer/candidates/_/apply"
      : op === "rollback" ? "/api/optimizer/rollback"
        : "/api/optimizer/alias/initialize";
    const r = await fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (r.ok) {
      msg.className = "opt-confirm-msg msg ok";
      msg.textContent = "操作成功，重新讀取狀態…";
      form._idem = null;
      await loadDeploy();
    } else {
      const errCode = (r.body && r.body.error && r.body.error.code) || ("HTTP_" + r.status);
      const errMsg = (r.body && r.body.error && r.body.error.message) || "操作失敗";
      msg.className = "opt-confirm-msg msg error";
      msg.textContent = errCode + "：" + errMsg;
      submit.disabled = false; // 允許修正後重試（沿用同一 idempotency key）
    }
  }

  function initOptimizer() {
    document.querySelectorAll(".subtab").forEach(function (t) {
      t.addEventListener("click", function () { setSub(t.dataset.sub); });
    });
    // 第一次點擊 optimizer 主 tab 才載入（lazy）。
    document.querySelectorAll('.tab[data-tab="optimizer"]').forEach(function (tab) {
      tab.addEventListener("click", function () {
        if (!state.loaded) {
          state.loaded = true;
          loadOverview();
        }
      });
    });
  }

  window.OptimizerUI = UI;
  window.OptimizerController = { init: initOptimizer, setSub: setSub, loadOverview: loadOverview };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initOptimizer);
  } else {
    initOptimizer();
  }
})();
