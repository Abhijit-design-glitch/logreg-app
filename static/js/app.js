// Tab nav
document.querySelectorAll(".tab").forEach(t => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById(t.dataset.tab).classList.add("active");
  };
});

const $ = id => document.getElementById(id);
async function api(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  return r.json();
}
function err(msg) { return `<div class="error">${msg}</div>`; }
function fmt(x) { return typeof x === "number" ? x.toFixed(4) : x; }

let FEATURE_NAMES = [];

// ---------- Upload ----------
$("uploadBtn").onclick = async () => {
  const f = $("fileInput").files[0];
  if (!f) return alert("Choose a CSV first.");
  const fd = new FormData(); fd.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd }).then(r => r.json());
  renderDataset(r);
};
$("sampleBtn").onclick = async () => {
  const r = await api("/api/load_sample");
  renderDataset(r);
};
function renderDataset(r) {
  if (r.error) return $("datasetInfo").innerHTML = err(r.error);
  const cols = r.columns;
  $("targetSel").innerHTML = cols.map(c => `<option>${c}</option>`).join("");
  const head = `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${r.head.map(row => `<tr>${cols.map(c => `<td>${row[c]}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("datasetInfo").innerHTML = `
    <div class="card"><h3>Dataset info</h3>
      <p><b>Shape:</b> ${r.shape[0]} rows × ${r.shape[1]} columns</p>
      <p><b>Missing:</b> ${Object.entries(r.missing).filter(([_,v])=>v>0).map(([k,v])=>`${k}: ${v}`).join(", ") || "none"}</p>
      <p><b>Dtypes:</b> ${Object.entries(r.dtypes).map(([k,v])=>`${k}: ${v}`).join(", ")}</p>
      <h3>First 5 rows</h3>${head}
    </div>`;
}

// ---------- Preprocess ----------
$("preprocBtn").onclick = async () => {
  const r = await api("/api/preprocess", {
    target: $("targetSel").value,
    missing: $("missingSel").value,
    encoding: $("encSel").value,
    scale: $("scaleChk").checked,
  });
  if (r.error) return $("preprocResult").innerHTML = err(r.error);
  FEATURE_NAMES = r.feature_names;
  buildPredictForm();
  const tbl = rows => `<table><thead><tr>${Object.keys(rows[0]).map(c=>`<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(row=>`<tr>${Object.values(row).map(v=>`<td>${typeof v==="number"?v.toFixed(3):v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("preprocResult").innerHTML = `
    <div class="grid-2">
      <div class="card"><h3>Before</h3>${tbl(r.before_head)}</div>
      <div class="card"><h3>After</h3>${tbl(r.after_head)}</div>
    </div>
    <div class="card"><b>Features:</b> ${r.feature_names.join(", ")} <br/>
    <b>Classes:</b> ${r.class_names.join(", ")} (${r.n_classes} classes)<br/>
    <b>Shape:</b> ${r.shape[0]} × ${r.shape[1]}</div>`;
};

// ---------- EDA ----------
$("edaBtn").onclick = async () => {
  const r = await api("/api/eda", { target: $("targetSel").value });
  if (r.error) return $("edaPlots").innerHTML = err(r.error);
  $("edaPlots").innerHTML = Object.entries(r).map(([k, v]) =>
    `<div class="card"><h3>${k.replace(/_/g," ")}</h3><img src="${v}" /></div>`
  ).join("");
};

// ---------- Train ----------
$("testSize").oninput = e => $("tsLabel").textContent = (+e.target.value).toFixed(2);
$("trainBtn").onclick = async () => {
  const r = await api("/api/train", {
    test_size: +$("testSize").value,
    k_folds: +$("kFolds").value,
    max_iter: +$("maxIter").value,
  });
  if (r.error) return $("trainOut").innerHTML = err(r.error);
  const wTbl = r.weights.map((row, i) =>
    `<h4>Class: ${r.class_names[i] || (i===0 ? r.class_names[1] || "1" : i)}</h4>
    <table><thead><tr><th>Feature</th><th>Weight</th></tr></thead>
    <tbody>${row.map((w, j) => `<tr><td>${r.feature_names[j]}</td><td>${fmt(w)}</td></tr>`).join("")}
    <tr><td><b>bias</b></td><td>${fmt(r.bias[i])}</td></tr></tbody></table>`).join("");
  $("trainOut").innerHTML = `
    <div class="card">
      <span class="metric">Train acc: ${fmt(r.train_score)}</span>
      <span class="metric">Test acc: ${fmt(r.test_score)}</span>
      <span class="metric">CV mean: ${fmt(r.cv_mean)} ± ${fmt(r.cv_std)}</span>
      <p>CV fold scores: ${r.cv_scores.map(fmt).join(", ")}</p>
      <h3>Learned parameters</h3>${wTbl}
    </div>`;
  const v = await api("/api/visualize");
  $("vizOut").innerHTML = Object.entries(v).filter(([k])=>k!=="error").map(([k, val]) =>
    `<div class="card"><h3>${k}</h3><img src="${val}"/></div>`).join("");
};

// ---------- Explain sample ----------
$("explainBtn").onclick = async () => {
  const r = await api("/api/explain_sample", { index: +$("sampleIdx").value });
  if (r.error) return $("explainOut").innerHTML = err(r.error);
  const t = r.terms[r.terms.length - 1]; // show last (positive) class
  const rows = t.contributions.map(c =>
    `<tr><td>${c.feature}</td><td>${fmt(c.x)}</td><td>${fmt(c.w)}</td><td>${fmt(c.wx)}</td></tr>`).join("");
  const probBars = r.probabilities.map((p, i) =>
    `<div class="bar-row"><span style="width:80px">${r.class_names[i]||i}</span>
     <div class="bar" style="width:${p*200}px"></div><span>${fmt(p)}</span></div>`).join("");
  $("explainOut").innerHTML = `
    <h4>Computation for class "${t.class}"</h4>
    <table><thead><tr><th>Feature</th><th>x</th><th>w</th><th>w·x</th></tr></thead>
      <tbody>${rows}<tr><td colspan="3"><b>bias</b></td><td>${fmt(t.bias)}</td></tr>
      <tr><td colspan="3"><b>z = Σ w·x + b</b></td><td><b>${fmt(t.z)}</b></td></tr></tbody></table>
    <p>σ(z) = 1 / (1 + e<sup>-${fmt(t.z)}</sup>) → probabilities below</p>
    ${probBars}
    <p><b>Predicted:</b> ${r.prediction} &nbsp; <b>Actual:</b> ${r.actual}</p>`;
};

// ---------- Predict ----------
function buildPredictForm() {
  $("predictForm").innerHTML = `<div class="card">${
    FEATURE_NAMES.map(f => `<label>${f} <input type="number" step="any" data-f="${f}" value="0"/></label>`).join("")
  }</div>`;
}
$("predictBtn").onclick = async () => {
  const values = {};
  document.querySelectorAll("#predictForm input").forEach(i => values[i.dataset.f] = i.value);
  const r = await api("/api/predict", { values });
  if (r.error) return $("predictOut").innerHTML = err(r.error);
  const probBars = r.probabilities.map((p, i) =>
    `<div class="bar-row"><span style="width:80px">${r.class_names[i]||i}</span>
     <div class="bar" style="width:${p*200}px"></div><span>${fmt(p)}</span></div>`).join("");
  $("predictOut").innerHTML = `<div class="card">
    <p><b>z =</b> [${r.z.map(fmt).join(", ")}]</p>
    ${probBars}
    <p><b>Prediction:</b> <span class="metric">${r.prediction}</span></p>
  </div>`;
};

// ---------- Evaluate ----------
$("evalBtn").onclick = async () => {
  const r = await api("/api/evaluate");
  if (r.error) return $("evalOut").innerHTML = err(r.error);
  $("evalOut").innerHTML = `
    <div class="card">
      <span class="metric">Accuracy: ${fmt(r.accuracy)}</span>
      <span class="metric">Precision: ${fmt(r.precision)}</span>
      <span class="metric">Recall: ${fmt(r.recall)}</span>
      <span class="metric">F1: ${fmt(r.f1)}</span>
      ${r.auc !== undefined ? `<span class="metric">AUC: ${fmt(r.auc)}</span>` : ""}
    </div>
    <div class="plots">
      <div class="card"><h3>Confusion matrix</h3><img src="${r.cm_plot}"/></div>
      ${r.roc_plot ? `<div class="card"><h3>ROC curve</h3><img src="${r.roc_plot}"/></div>` : ""}
    </div>
    <div class="card"><h3>Classification report</h3><pre>${r.report}</pre></div>`;
};

// ---------- Ask AI ----------
$("askBtn").onclick = async () => {
  const q = $("askInput").value.trim();
  if (!q) return;
  $("askOut").innerHTML = "Thinking…";
  const r = await api("/api/ask", { question: q });
  $("askOut").innerHTML = `<p>${r.answer}</p>`;
};
