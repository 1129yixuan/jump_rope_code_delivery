const data = window.DASHBOARD_DATA;
const actualRows = data.actualRows;
const outputRows = data.outputRows;

const fmtPct = (value) => `${(value * 100).toFixed(1)}%`;
const fmtNum = (value, digits = 1) => Number(value).toFixed(digits);
const isNum = (value) => typeof value === "number" && Number.isFinite(value);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);

function metric(label, value, sub, tone = "") {
  return `<div class="metric ${tone}">
    <div class="label">${label}</div>
    <div class="value">${value}</div>
    <div class="sub">${sub}</div>
  </div>`;
}

function pill(value, ok) {
  if (!isNum(value)) return `<span class="pill neutral">-</span>`;
  const text = value > 0 ? `+${value}` : `${value}`;
  return `<span class="pill ${ok ? "ok" : "bad"}">${text}</span>`;
}

function shortFile(file) {
  return String(file || "").replace(/\.mkv$/i, "").replace(/^30s_?/i, "");
}

function renderMeta() {
  document.querySelector("#meta").textContent =
    `${data.meta.actualRows} Venue 2 ground-truth records · ${data.meta.tcnPeakRows} TCN outputs · ${data.meta.outputRows} total outputs · tolerance ±${data.meta.threshold}`;
}

function renderAccuracyKpis() {
  const tcnPaired = data.summaries.actual.tcnPeakPaired;
  const original = data.summaries.actual.original;
  const next = data.summaries.actual.new;
  const tcnPeak = tcnPaired.tcnPeak;
  const tcnDelta = tcnPeak.acc10 - tcnPaired.new.acc10;
  const tcnMaeDelta = tcnPaired.new.mae - tcnPeak.mae;
  const tcnVsOriginal = tcnPeak.acc10 - tcnPaired.original.acc10;
  document.querySelector("#accuracyKpis").innerHTML = [
    metric("Shoulder accuracy", fmtPct(original.acc10), `${original.n} matched records, MAE ${fmtNum(original.mae)}`),
    metric("Routing accuracy", fmtPct(next.acc10), `${next.n} matched records, MAE ${fmtNum(next.mae)}`),
    metric("TCN accuracy", fmtPct(tcnPeak.acc10), `${tcnPeak.n} matched records, MAE ${fmtNum(tcnPeak.mae)}`, "good"),
    metric("TCN vs routing", `${tcnDelta >= 0 ? "+" : ""}${fmtPct(tcnDelta)}`, `${tcnPeak.ok10 - tcnPaired.new.ok10} additional hits across ${tcnPeak.n} records, MAE reduced by ${fmtNum(tcnMaeDelta)}`, tcnDelta >= 0 ? "good" : "warn"),
    metric("TCN vs shoulder", `${tcnVsOriginal >= 0 ? "+" : ""}${fmtPct(tcnVsOriginal)}`, `${tcnPeak.ok10 - tcnPaired.original.ok10} additional hits across ${tcnPeak.n} records`, tcnVsOriginal >= 0 ? "good" : "warn"),
    metric("Evaluation scope", "Venue 2", data.meta.scope || "Venue 1 is excluded from evaluation"),
  ].join("");
}

function renderOutputKpis() {
  const diff = data.summaries.output.diff;
  document.querySelector("#outputKpis").innerHTML = [
    metric("Total outputs", `${diff.n}`, "Records produced by both algorithm versions"),
    metric("Mean absolute difference", fmtNum(diff.avgAbsDiff), `|routing - shoulder|, maximum ${diff.maxAbsDiff}`),
    metric("Routing higher / lower", `${diff.newHigher} / ${diff.newLower}`, `${diff.same} exact matches`),
    metric("Difference > 20", `${diff.gt20}`, `${diff.gt10} above 10 and ${diff.gt30} above 30`, diff.gt30 ? "warn" : ""),
  ].join("");
}

function renderGroupedAccuracy() {
  const groupKey = document.querySelector("#accuracyGroup").value;
  const groups = data.summaries.actual[groupKey];
  const max = 1;
  document.querySelector("#accuracyGroupChart").innerHTML = groups
    .map((group) => {
      const originalWidth = Math.max(2, group.original.acc10 / max * 100);
      const newWidth = Math.max(2, group.new.acc10 / max * 100);
      const tcnPeakWidth = Math.max(2, group.tcnPeak.acc10 / max * 100);
      return `<div class="bar-row">
        <div class="bar-label">${group.group}</div>
        <div class="bar-track">
          <div class="bar original" style="width:${originalWidth}%"></div>
          <div class="bar new" style="width:${newWidth}%"></div>
          <div class="bar tcn-peak" style="width:${tcnPeakWidth}%"></div>
        </div>
        <div class="bar-value">${fmtPct(group.tcnPeak.acc10)}</div>
      </div>`;
    })
    .join("");
}

function renderStatus() {
  const t = data.summaries.actual.tcnPeakStatus;
  document.querySelector("#statusChart").innerHTML = [
    ["Routing and TCN both accurate", t.bothCorrect, "ok"],
    ["Routing missed, TCN accurate", t.newWrongTcnPeakCorrect, "ok"],
    ["Routing accurate, TCN missed", t.newCorrectTcnPeakWrong, "bad"],
    ["Routing and TCN both missed", t.bothWrong, "bad"],
  ]
    .map(([label, value, tone]) => `<div class="status-item">
      <strong class="${tone}">${value}</strong><span>${label}</span>
    </div>`)
    .join("");
}

function renderVideoTable() {
  const query = document.querySelector("#videoSearch").value.trim().toLowerCase();
  const rows = data.summaries.actual.byVideo
    .filter((row) => `${row.area} ${row.file}`.toLowerCase().includes(query))
    .map((row) => `<tr>
      <td>${row.area}</td>
      <td>${shortFile(row.file)}</td>
      <td class="num">${row.n}</td>
      <td class="num">${row.original.ok10}/${row.original.n}</td>
      <td class="num">${row.new.ok10}/${row.new.n}</td>
      <td class="num">${row.tcnPeak.ok10}/${row.tcnPeak.n}</td>
      <td class="num">${fmtNum(row.original.mae)}</td>
      <td class="num">${fmtNum(row.new.mae)}</td>
      <td class="num">${fmtNum(row.tcnPeak.mae)}</td>
    </tr>`)
    .join("");
  document.querySelector("#videoTable").innerHTML = rows;
}

function renderHistogram(id, bins, colorClass = "") {
  const max = Math.max(...bins.map((bin) => bin.count), 1);
  document.querySelector(id).innerHTML = bins
    .map((bin) => {
      const height = Math.max(2, (bin.count / max) * 190);
      return `<div class="hist-bin">
        <div class="hist-bar ${colorClass}" style="height:${height}px"></div>
        <div class="hist-count">${bin.count}</div>
        <div class="hist-label">${bin.label}</div>
      </div>`;
    })
    .join("");
}

function renderOutputVideoTable() {
  const area = document.querySelector("#outputArea").value;
  const rows = data.summaries.output.byVideo
    .filter((row) => area === "all" || row.area === area)
    .map((row) => `<tr>
      <td>${row.area}</td>
      <td>${shortFile(row.file)}</td>
      <td class="num">${row.n}</td>
      <td class="num">${row.actualN}</td>
      <td class="num">${row.diff.newHigher}</td>
      <td class="num">${row.diff.newLower}</td>
      <td class="num">${fmtNum(row.diff.avgDiff)}</td>
      <td class="num">${fmtNum(row.diff.avgAbsDiff)}</td>
      <td class="num">${row.diff.gt10}</td>
    </tr>`)
    .join("");
  document.querySelector("#outputVideoTable").innerHTML = rows;
}

function recordFilters() {
  return {
    set: document.querySelector("#recordSet").value,
    area: document.querySelector("#areaFilter").value,
    zone: "all",
    status: "all",
    query: document.querySelector("#recordSearch").value.trim().toLowerCase(),
  };
}

function recordRowsForSet(set) {
  return set === "actual" ? actualRows : outputRows;
}

function videoKey(row) {
  return `${row.area}|${row.file}`;
}

function matchesRecordArea(row, filters) {
  return filters.area === "all" || row.area === filters.area;
}

function matchesRecordQuery(row, filters) {
  if (!filters.query) return true;
  return `${row.file} ${row.name} ${row.school}`.toLowerCase().includes(filters.query);
}

function matchesRecordZone(row, filters) {
  return filters.zone === "all" || String(row.zone) === filters.zone;
}

function matchesRecordStatus(row, filters) {
  if (filters.status === "all") return true;
  if (filters.status === "fixed") return row.newAcc10 === false && row.tcnPeakAcc10 === true;
  if (filters.status === "regressed") return row.newAcc10 === true && row.tcnPeakAcc10 === false;
  if (filters.status === "newWrong") return row.tcnPeakAcc10 === false;
  if (filters.status === "largeDiff") return Math.abs(row.diff || 0) > 20;
  return true;
}

function currentRecordRows() {
  const filters = recordFilters();
  let rows = recordRowsForSet(filters.set);

  rows = rows.filter((row) => matchesRecordArea(row, filters));
  rows = rows.filter((row) => matchesRecordZone(row, filters));
  rows = rows.filter((row) => matchesRecordQuery(row, filters));
  rows = rows.filter((row) => matchesRecordStatus(row, filters));

  return rows;
}

function currentChartRows() {
  const filters = recordFilters();
  const areaRows = recordRowsForSet(filters.set).filter((row) => matchesRecordArea(row, filters));
  const matchingVideoKeys = new Set(
    areaRows
      .filter((row) => matchesRecordQuery(row, filters))
      .filter((row) => matchesRecordZone(row, filters))
      .filter((row) => matchesRecordStatus(row, filters))
      .map(videoKey),
  );
  return areaRows.filter((row) => matchingVideoKeys.has(videoKey(row)));
}

function groupRowsByVideo(rows) {
  const groups = new Map();
  rows.forEach((row) => {
    const key = videoKey(row);
    if (!groups.has(key)) {
      groups.set(key, {
        key,
        area: row.area,
        file: row.file,
        video: row.video,
        lane: row.lane,
        rows: [],
      });
    }
    groups.get(key).rows.push(row);
  });
  return [...groups.values()].sort((a, b) => {
    const areaCompare = a.area.localeCompare(b.area, "en");
    if (areaCompare) return areaCompare;
    const aVideo = Number(a.video);
    const bVideo = Number(b.video);
    if (Number.isFinite(aVideo) && Number.isFinite(bVideo) && aVideo !== bVideo) return aVideo - bVideo;
    return a.file.localeCompare(b.file, "en");
  });
}

function lineSegments(points) {
  const segments = [];
  let current = [];
  points.forEach((point) => {
    if (!point) {
      if (current.length > 1) segments.push(current);
      current = [];
      return;
    }
    current.push(point);
  });
  if (current.length > 1) segments.push(current);
  return segments;
}

function renderSeriesLine(points, className, formatter = (value) => value) {
  const segments = lineSegments(points);
  const lines = segments
    .map((segment) => `<polyline class="trend-line ${className}" points="${segment.map((p) => `${p.x},${p.y}`).join(" ")}" />`)
    .join("");
  const dots = points
    .filter(Boolean)
    .map((point) => `<circle class="trend-dot ${className}" cx="${point.x}" cy="${point.y}" r="3.2"><title>${formatter(point.value)}</title></circle>`)
    .join("");
  return lines + dots;
}

function renderLegend(series) {
  return series
    .map((item) => `<button type="button" class="legend-toggle" data-series="${item.className}" aria-pressed="true">
      <i class="${item.className}"></i><span>${item.label}</span>
    </button>`)
    .join("");
}

function renderTrendChart(group, options = {}) {
  const width = options.wide ? 760 : 340;
  const height = options.wide ? 230 : 190;
  const chart = { left: options.wide ? 54 : 38, right: 14, top: 18, bottom: 30 };
  const plotWidth = width - chart.left - chart.right;
  const plotHeight = height - chart.top - chart.bottom;
  const zones = [1, 2, 3, 4, 5];
  const byZone = new Map(group.rows.map((row) => [Number(row.zone), row]));
  const series = options.series || [
    { key: "actual", label: "Actual", className: "actual" },
    { key: "original", label: "Shoulder", className: "shoulder" },
    { key: "new", label: "Routing", className: "routed" },
    { key: "tcnPeak", label: "TCN", className: "tcn-peak" },
  ];
  const values = group.rows.flatMap((row) => series.map((item) => row[item.key])).filter(isNum);
  const rawMin = values.length ? Math.min(...values) : 0;
  const rawMax = values.length ? Math.max(...values) : 10;
  const yMin = isNum(options.yMin) ? options.yMin : Math.max(0, Math.floor((rawMin - 6) / 10) * 10);
  const yMax = isNum(options.yMax) ? options.yMax : Math.ceil((rawMax + 6) / 10) * 10 || 10;
  const yRange = yMax - yMin || 1;
  const valueFormatter = options.valueFormatter || ((value) => value);
  const xForZone = (zone) => chart.left + ((zone - 1) / 4) * plotWidth;
  const yForValue = (value) => chart.top + (1 - (value - yMin) / yRange) * plotHeight;
  const seriesSvg = series
    .map((item) => {
      const points = zones.map((zone) => {
        const row = byZone.get(zone);
        if (!row || !isNum(row[item.key])) return null;
        return { x: xForZone(zone), y: yForValue(row[item.key]), value: row[item.key] };
      });
      return renderSeriesLine(points, item.className, valueFormatter);
    })
    .join("");

  return `<article class="trend-card ${options.wide ? "trend-card-wide" : ""}" data-chart-card>
    <div class="trend-card-head">
      <div>
        <strong>${esc(group.title || `${group.area} · ${shortFile(group.file)}`)}</strong>
        <span>${esc(group.subtitle || `${group.rows.length} zones`)}</span>
      </div>
      <div class="trend-legend">
        ${renderLegend(series)}
      </div>
    </div>
    <svg class="trend-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(group.title || shortFile(group.file))} five-zone trend comparison">
      <line class="trend-axis" x1="${chart.left}" y1="${chart.top}" x2="${chart.left}" y2="${height - chart.bottom}" />
      <line class="trend-axis" x1="${chart.left}" y1="${height - chart.bottom}" x2="${width - chart.right}" y2="${height - chart.bottom}" />
      <line class="trend-grid-line" x1="${chart.left}" y1="${chart.top}" x2="${width - chart.right}" y2="${chart.top}" />
      <line class="trend-grid-line" x1="${chart.left}" y1="${chart.top + plotHeight / 2}" x2="${width - chart.right}" y2="${chart.top + plotHeight / 2}" />
      <text class="trend-y-label" x="${chart.left - 8}" y="${chart.top + 4}">${valueFormatter(yMax)}</text>
      <text class="trend-y-label" x="${chart.left - 8}" y="${height - chart.bottom + 4}">${valueFormatter(yMin)}</text>
      ${zones.map((zone) => `<text class="trend-x-label" x="${xForZone(zone)}" y="${height - 8}">zone${zone}</text>`).join("")}
      ${seriesSvg}
    </svg>
  </article>`;
}

function renderPositionChart(zone, groups) {
  const rows = groups
    .map((group) => {
      const row = group.rows.find((item) => Number(item.zone) === zone);
      if (!row) return null;
      return {
        label: shortFile(group.file),
        actual: row.actual,
        original: row.original,
        new: row.new,
        tcnPeak: row.tcnPeak,
      };
    })
    .filter(Boolean);
  if (!rows.length) return "";
  const accuracyFor = (key) => {
    const valid = rows.filter((row) => isNum(row.actual) && isNum(row[key]));
    const ok = valid.filter((row) => Math.abs(row[key] - row.actual) <= data.meta.threshold).length;
    return {
      n: valid.length,
      ok,
      text: valid.length ? fmtPct(ok / valid.length) : "-",
    };
  };
  const originalAccuracy = accuracyFor("original");
  const newAccuracy = accuracyFor("new");
  const tcnPeakAccuracy = accuracyFor("tcnPeak");

  const series = [
    { key: "actual", label: "Actual", className: "actual" },
    { key: "original", label: "Shoulder", className: "shoulder" },
    { key: "new", label: "Routing", className: "routed" },
    { key: "tcnPeak", label: "TCN", className: "tcn-peak" },
  ];
  const width = Math.max(760, rows.length * 44 + 78);
  const height = 250;
  const chart = { left: 54, right: 18, top: 18, bottom: 58 };
  const plotWidth = width - chart.left - chart.right;
  const plotHeight = height - chart.top - chart.bottom;
  const values = rows.flatMap((row) => series.map((item) => row[item.key])).filter(isNum);
  const rawMin = values.length ? Math.min(...values) : 0;
  const rawMax = values.length ? Math.max(...values) : 10;
  const yMin = Math.max(0, Math.floor((rawMin - 6) / 10) * 10);
  const yMax = Math.ceil((rawMax + 6) / 10) * 10 || 10;
  const yRange = yMax - yMin || 1;
  const xForIndex = (index) => chart.left + (rows.length === 1 ? plotWidth / 2 : (index / (rows.length - 1)) * plotWidth);
  const yForValue = (value) => chart.top + (1 - (value - yMin) / yRange) * plotHeight;
  const seriesSvg = series
    .map((item) => {
      const points = rows.map((row, index) => {
        if (!isNum(row[item.key])) return null;
        return { x: xForIndex(index), y: yForValue(row[item.key]), value: row[item.key] };
      });
      return renderSeriesLine(points, item.className);
    })
    .join("");

  return `<article class="trend-card trend-card-wide position-trend-card" data-chart-card>
    <div class="trend-card-head">
      <div>
        <strong>Position ${zone} video trend</strong>
        <span>${rows.length} videos; the horizontal axis shows this position across videos</span>
      </div>
      <div class="trend-legend">
        ${renderLegend(series)}
      </div>
    </div>
    <div class="position-chart-scroll">
      <svg class="trend-svg position-trend-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Position ${zone} count trend across all videos">
        <line class="trend-axis" x1="${chart.left}" y1="${chart.top}" x2="${chart.left}" y2="${height - chart.bottom}" />
        <line class="trend-axis" x1="${chart.left}" y1="${height - chart.bottom}" x2="${width - chart.right}" y2="${height - chart.bottom}" />
        <line class="trend-grid-line" x1="${chart.left}" y1="${chart.top}" x2="${width - chart.right}" y2="${chart.top}" />
        <line class="trend-grid-line" x1="${chart.left}" y1="${chart.top + plotHeight / 2}" x2="${width - chart.right}" y2="${chart.top + plotHeight / 2}" />
        <text class="trend-y-label" x="${chart.left - 8}" y="${chart.top + 4}">${yMax}</text>
        <text class="trend-y-label" x="${chart.left - 8}" y="${height - chart.bottom + 4}">${yMin}</text>
        ${rows
          .map((row, index) => `<text class="trend-x-label position-x-label" x="${xForIndex(index)}" y="${height - 14}" transform="rotate(-35 ${xForIndex(index)} ${height - 14})">${esc(row.label)}</text>`)
          .join("")}
        ${seriesSvg}
      </svg>
    </div>
    <div class="position-accuracy">
      <div>
        <span>Shoulder accuracy</span>
        <strong>${originalAccuracy.text}</strong>
        <em>${originalAccuracy.ok}/${originalAccuracy.n} within tolerance</em>
      </div>
      <div>
        <span>Routing accuracy</span>
        <strong>${newAccuracy.text}</strong>
        <em>${newAccuracy.ok}/${newAccuracy.n} within tolerance</em>
      </div>
      <div>
        <span>TCN accuracy</span>
        <strong>${tcnPeakAccuracy.text}</strong>
        <em>${tcnPeakAccuracy.ok}/${tcnPeakAccuracy.n} within tolerance</em>
      </div>
    </div>
  </article>`;
}

function renderPositionCharts(groups) {
  const charts = [1, 2, 3, 4, 5].map((zone) => renderPositionChart(zone, groups)).filter(Boolean);
  return charts.length ? charts.join("") : `<div class="empty-state">No position trends match the filters</div>`;
}

function renderRecordCharts() {
  const rows = currentChartRows();
  const groups = groupRowsByVideo(rows);
  document.querySelector("#positionCharts").innerHTML = renderPositionCharts(groups);
  document.querySelector("#chartCount").textContent = `${groups.length} videos`;
  document.querySelector("#recordCharts").innerHTML = groups.length
    ? groups.map(renderTrendChart).join("")
    : `<div class="empty-state">No videos match the filters</div>`;
}

function renderRecordTable() {
  const rows = currentRecordRows();
  document.querySelector("#recordCount").textContent = `${rows.length} records`;
  document.querySelector("#recordTable").innerHTML = rows
    .slice(0, 400)
    .map((row) => `<tr>
      <td>${row.area}</td>
      <td>${shortFile(row.file)}</td>
      <td class="num">${row.zone}</td>
      <td>${row.name || "-"}</td>
      <td class="num">${isNum(row.actual) ? row.actual : "-"}</td>
      <td class="num">${isNum(row.original) ? row.original : "-"}</td>
      <td class="num">${pill(row.originalError, row.originalAcc10)}</td>
      <td class="num">${isNum(row.new) ? row.new : "-"}</td>
      <td class="num">${pill(row.newError, row.newAcc10)}</td>
      <td class="num">${isNum(row.tcnPeak) ? row.tcnPeak : "-"}</td>
      <td class="num">${pill(row.tcnPeakError, row.tcnPeakAcc10)}</td>
      <td class="num">${pill(row.tcnPeakDiff, Math.abs(row.tcnPeakDiff || 0) <= 10)}</td>
      <td class="num">${pill(row.diff, Math.abs(row.diff || 0) <= 10)}</td>
    </tr>`)
    .join("");
}

function renderRecordsView() {
  renderRecordCharts();
  renderRecordTable();
}

function toggleChartSeries(button) {
  const series = button.dataset.series;
  const card = button.closest("[data-chart-card]");
  if (!series || !card) return;
  const hiddenClass = `hide-${series}`;
  card.classList.toggle(hiddenClass);
  const isVisible = !card.classList.contains(hiddenClass);
  button.classList.toggle("muted", !isVisible);
  button.setAttribute("aria-pressed", String(isVisible));
}

function wireEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((node) => node.classList.remove("active"));
      document.querySelectorAll(".view").forEach((node) => node.classList.remove("active"));
      tab.classList.add("active");
      document.querySelector(`#${tab.dataset.view}`).classList.add("active");
    });
  });
  document.querySelector("#accuracyGroup").addEventListener("change", renderGroupedAccuracy);
  document.querySelector("#videoSearch").addEventListener("input", renderVideoTable);
  document.querySelector("#outputArea").addEventListener("change", renderOutputVideoTable);
  ["recordSet", "areaFilter", "recordSearch"].forEach((id) => {
    document.querySelector(`#${id}`).addEventListener("input", renderRecordsView);
  });
  document.querySelector("#records").addEventListener("click", (event) => {
    const button = event.target.closest(".legend-toggle");
    if (button) toggleChartSeries(button);
  });
}

function init() {
  renderMeta();
  renderAccuracyKpis();
  renderGroupedAccuracy();
  renderStatus();
  renderVideoTable();
  renderOutputKpis();
  renderHistogram("#diffHistogram", data.summaries.output.histogram);
  renderOutputVideoTable();
  renderRecordsView();
  wireEvents();
}

init();
