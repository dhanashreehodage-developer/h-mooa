Chart.defaults.color = "#9aa6c3";
Chart.defaults.borderColor = "#243052";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";

const C = { rf71: "#4f8cff", rf9: "#22d3ee", rf6: "#a78bfa", ok: "#34d399", warn: "#fbbf24" };
const f4 = (v) => v.toFixed(4);
const pct = (v) => (v * 100).toFixed(2) + "%";
const $ = (id) => document.getElementById(id);

function table(id, head, rows, hlRow) {
  const el = $(id);
  el.innerHTML =
    "<thead><tr>" + head.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r, i) => `<tr class="${i === hlRow ? "hl" : ""}">` + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") +
    "</tbody>";
}

new Chart($("classDist"), {
  type: "bar",
  data: { labels: DATA.classDist.labels, datasets: [{ data: DATA.classDist.pct, backgroundColor: DATA.classDist.labels.map((l) => (l === "benign" ? C.rf71 : C.rf9)), borderRadius: 6 }] },
  options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => c.raw + "%" } } }, scales: { y: { title: { display: true, text: "% of records" } } } },
});

table(
  "finalTable",
  ["Model", "Features", "Reduction", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Macro FPR", "Train (s)", "Predict (s)"],
  DATA.final.map((r) => [r.model, r.feats, r.red.toFixed(1) + "%", f4(r.acc), f4(r.prec), f4(r.rec), f4(r.f1), f4(r.wf1), f4(r.fpr), r.train.toFixed(1), r.pred.toFixed(3)]),
  1
);

new Chart($("tradeoff"), {
  type: "bar",
  data: {
    labels: DATA.final.map((r) => `${r.model} (${r.feats} feats)`),
    datasets: [
      { label: "Accuracy", data: DATA.final.map((r) => r.acc), backgroundColor: C.rf71, borderRadius: 6 },
      { label: "Macro F1", data: DATA.final.map((r) => r.f1), backgroundColor: C.rf9, borderRadius: 6 },
    ],
  },
  options: { scales: { y: { min: 0.94, max: 0.975 } }, plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f4(c.raw)}` } } } },
});

new Chart($("timing"), {
  type: "bar",
  data: {
    labels: DATA.final.map((r) => r.model),
    datasets: [
      { label: "Training time (s)", data: DATA.final.map((r) => r.train), backgroundColor: C.rf6, borderRadius: 6, yAxisID: "y" },
      { label: "Batch prediction, 109,771 rows (s)", data: DATA.final.map((r) => r.pred), backgroundColor: C.ok, borderRadius: 6, yAxisID: "y1" },
    ],
  },
  options: { scales: { y: { position: "left", title: { display: true, text: "training (s)" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "prediction (s)" } } } },
});

new Chart($("classwise"), {
  type: "bar",
  data: {
    labels: DATA.classes,
    datasets: [
      { label: "RF_71", data: DATA.classwise.RF_71, backgroundColor: C.rf71, borderRadius: 4 },
      { label: "RF_9", data: DATA.classwise.RF_9, backgroundColor: C.rf9, borderRadius: 4 },
      { label: "RF_6", data: DATA.classwise.RF_6, backgroundColor: C.rf6, borderRadius: 4 },
    ],
  },
  options: { scales: { y: { min: 0.9, max: 1 } }, plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${f4(c.raw)}` } } } },
});

(function confusion() {
  const m = DATA.confusionRF9, cls = DATA.classes, el = $("cm");
  let html = '<div class="h"></div>' + cls.map((c) => `<div class="h">${c.slice(0, 5)}</div>`).join("");
  m.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    html += `<div class="h rh">${cls[i]}</div>`;
    row.forEach((v, j) => {
      const share = v / total;
      const bg = i === j ? `rgba(52,211,153,${0.25 + share * 0.6})` : `rgba(248,113,113,${Math.min(share * 6, 0.8)})`;
      html += `<div style="background:${v ? bg : "transparent"}" title="true ${cls[i]} → predicted ${cls[j]}: ${v} (${(share * 100).toFixed(2)}% of row)">${v.toLocaleString()}</div>`;
    });
  });
  el.innerHTML = html;
})();

table("a2b", ["Model", "Missed as benign", "Rate"], DATA.attackToBenign.map((r) => [r[0], r[1].toLocaleString(), r[2].toFixed(2) + "%"]), 1);

new Chart($("importance"), {
  type: "bar",
  data: { labels: DATA.importance.labels, datasets: [{ data: DATA.importance.values, backgroundColor: C.rf6, borderRadius: 6 }] },
  options: { indexAxis: "y", plugins: { legend: { display: false } } },
});

$("rf9").innerHTML = DATA.rf9.map((f) => `<li>${f}</li>`).join("");
$("rf6").innerHTML = DATA.rf6.map((f) => `<li>${f}</li>`).join("");

table("bootstrap", ["Comparison", "Metric", "Mean diff", "95% CI"], DATA.bootstrap.map((r) => [r[0], r[1], r[2].toFixed(4), `[${r[3].toFixed(4)}, ${r[4].toFixed(4)}]`]));
table("mcnemar", ["Comparison", "A right, B wrong", "A wrong, B right", "Holm-adj. p"], DATA.mcnemar.map((r) => [r[0], r[1].toLocaleString(), r[2].toLocaleString(), `10<sup>${r[3].toFixed(1)}</sup>`]));

table("seeds", ["Seed", "Features", "Macro F1", "Macro FPR", "Jaccard vs RF_9"], DATA.seeds.map((r) => [r[0], r[1], f4(r[2]), f4(r[3]), r[4].toFixed(3)]));

new Chart($("freq"), {
  type: "bar",
  data: { labels: DATA.freq.labels, datasets: [{ data: DATA.freq.values, backgroundColor: DATA.freq.values.map((v) => (v === 5 ? C.ok : v >= 4 ? C.rf9 : "#64748b")), borderRadius: 4 }] },
  options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `selected in ${c.raw} of 5 seeds` } } }, scales: { x: { max: 5, ticks: { stepSize: 1 } } } },
});

new Chart($("fpso"), {
  type: "scatter",
  data: {
    datasets: [{
      label: "Archived solutions",
      data: DATA.fpsoArchive.feats.map((x, i) => ({ x, y: DATA.fpsoArchive.f1[i], fpr: DATA.fpsoArchive.fpr[i] })),
      backgroundColor: C.warn, pointRadius: 6,
    }],
  },
  options: {
    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.raw.x} features, F1 ${f4(c.raw.y)}, FPR ${f4(c.raw.fpr)}` } } },
    scales: { x: { title: { display: true, text: "feature count" } }, y: { title: { display: true, text: "validation macro F1" } } },
  },
});

new Chart($("cuckoo"), {
  type: "scatter",
  data: {
    datasets: [
      { label: "Evaluated settings", data: DATA.cuckoo.f1.map((y, i) => ({ x: DATA.cuckoo.fpr[i], y })).filter((_, i) => !DATA.cuckoo.archive.includes(i)), backgroundColor: "#64748b", pointRadius: 5 },
      { label: "Pareto archive", data: DATA.cuckoo.archive.map((i) => ({ x: DATA.cuckoo.fpr[i], y: DATA.cuckoo.f1[i] })), backgroundColor: C.warn, pointRadius: 7 },
    ],
  },
  options: {
    plugins: { tooltip: { callbacks: { label: (c) => `F1 ${f4(c.raw.y)}, FPR ${f4(c.raw.x)}` } } },
    scales: { x: { title: { display: true, text: "validation macro FPR" } }, y: { title: { display: true, text: "validation macro F1" } } },
  },
});

table("thresh", ["Rule (assessment split)", "Accuracy", "Macro F1", "Macro FPR"], DATA.thresholdAssessment.map((r) => [r[0], f4(r[1]), f4(r[2]), f4(r[3])]));
table("threshClasses", ["Class", "Optimized threshold"], DATA.classes.map((c, j) => [c, DATA.thresholds[j].toFixed(2)]));
