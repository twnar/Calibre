(() => {
  const $ = (id) => document.getElementById(id);
  const state = { example: null, file: null };

  const chips = [...document.querySelectorAll(".chip")];
  const drop = $("drop"), fileInput = $("file");
  const settings = $("settings"), targetSel = $("target");
  const confidence = $("confidence"), confidenceOut = $("confidence-out");
  const runBtn = $("run"), errorBox = $("error"), results = $("results");

  const pct = (x, d = 1) => (x * 100).toFixed(d).replace(/\.0$/, "") + "%";
  const num = (x) => {
    const a = Math.abs(x);
    return x.toLocaleString("en-US", { maximumFractionDigits: a >= 100 ? 0 : a >= 10 ? 1 : 2 });
  };

  function showError(msg) { errorBox.textContent = msg; errorBox.hidden = !msg; }

  function source() {
    const fd = new FormData();
    if (state.example) fd.append("example", state.example);
    else if (state.file) fd.append("file", state.file);
    return fd;
  }

  async function post(url, fd) {
    const res = await fetch(url, { method: "POST", body: fd });
    let data = {};
    try { data = await res.json(); } catch (_) { /* non-JSON error */ }
    if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
    return data;
  }

  async function inspect() {
    showError("");
    results.hidden = true;
    try {
      const info = await post("/api/inspect", source());
      targetSel.innerHTML = "";
      info.columns.forEach((c) => {
        const o = document.createElement("option");
        o.value = c; o.textContent = c;
        if (c === info.default_target) o.selected = true;
        targetSel.appendChild(o);
      });
      settings.hidden = false;
    } catch (e) {
      settings.hidden = true;
      showError(e.message);
    }
  }

  chips.forEach((chip) => chip.addEventListener("click", () => {
    chips.forEach((c) => c.setAttribute("aria-checked", "false"));
    chip.setAttribute("aria-checked", "true");
    state.example = chip.dataset.example;
    state.file = null;
    fileInput.value = "";
    drop.classList.remove("has-file");
    $("drop-main").textContent = "Or drop your own CSV here";
    inspect();
  }));

  function takeFile(file) {
    if (!file) return;
    state.file = file;
    state.example = null;
    chips.forEach((c) => c.setAttribute("aria-checked", "false"));
    drop.classList.add("has-file");
    $("drop-main").textContent = file.name;
    inspect();
  }

  fileInput.addEventListener("change", () => takeFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => takeFile(e.dataTransfer.files[0]));

  confidence.addEventListener("input", () => { confidenceOut.textContent = confidence.value + "%"; });

  runBtn.addEventListener("click", async () => {
    showError("");
    runBtn.disabled = true;
    const label = runBtn.textContent;
    runBtn.textContent = "Running…";
    try {
      const fd = source();
      fd.append("target", targetSel.value);
      fd.append("confidence", confidence.value);
      const r = await post("/api/run", fd);
      render(r);
    } catch (e) {
      showError(e.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = label;
    }
  });

  function pill(value, label, kind = "") {
    const d = document.createElement("div");
    d.className = "pill " + kind;
    const b = document.createElement("b"); b.textContent = value;
    const s = document.createElement("span"); s.textContent = label;
    d.append(b, s);
    return d;
  }

  function render(r) {
    const m = r.metrics, conf = 1 - r.alpha, reg = r.task === "regression";
    $("headline").textContent =
      `You asked for ${pct(conf, 0)} confidence. On ${r.split.test} rows the model never saw, ` +
      `the truth was inside its prediction ${pct(m.coverage)} of the time.`;

    $("subline").textContent = reg
      ? `A typical "average error" range would have caught only ${pct(m.naive_coverage)}. ` +
        `Calibre's ranges are wider, but they are the width you actually need.`
      : `Predictions are short lists of possible classes. ` +
        `${pct(m.single_answer_share, 0)} of the time it commits to a single answer; ` +
        `the rest it honestly says "one of these".`;

    const pills = $("pills");
    pills.replaceChildren();
    const ok = Math.abs(m.coverage - conf) <= 0.06 || m.coverage >= conf;
    pills.append(pill(pct(m.coverage), "coverage on held-out rows", ok ? "good" : "warn"));
    if (reg) {
      pills.append(
        pill("±" + num(m.avg_width / 2), `typical range (model error ${num(m.mae)})`),
        pill(pct(m.naive_coverage), "coverage of a naive range", "warn"),
        pill(num(m.target_spread), "spread of the target (std)")
      );
    } else {
      pills.append(
        pill(pct(m.accuracy), "top pick alone is right"),
        pill(m.avg_set_size.toFixed(2), "answers per prediction"),
        pill(pct(m.multi_answer_share, 0), "predictions that hedge")
      );
    }

    $("chart-hint").textContent =
      "Each point is a confidence level you could have asked for. The closer the line hugs the dashed diagonal, the more honest the promise.";
    drawChart(r);
    drawSamples(r);

    const notes = $("notes");
    notes.replaceChildren();
    const lines = [...r.warnings];
    lines.push(`${r.n_rows.toLocaleString()} rows and ${r.n_features} input columns: ` +
      `${r.split.train} to train, ${r.split.calibration} to calibrate, ${r.split.test} to test.`);
    lines.forEach((w) => { const li = document.createElement("li"); li.textContent = w; notes.appendChild(li); });

    results.hidden = false;
    results.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  function drawChart(r) {
    const W = 640, H = 340, L = 58, R = 20, T = 16, B = 48;
    const xs = r.sweep.map((p) => p.confidence);
    const lo = 0.4, hi = 1.0, xmin = Math.min(...xs), xmax = Math.max(...xs);
    const X = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R);
    const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const ns = "http://www.w3.org/2000/svg";
    const el = (name, attrs, text) => {
      const e = document.createElementNS(ns, name);
      Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
      if (text !== undefined) e.textContent = text;
      return e;
    };
    const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Chart of promised confidence versus measured coverage" });

    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0].forEach((v) => {
      svg.append(el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: css("--line"), "stroke-width": 1 }));
      svg.append(el("text", { x: L - 12, y: Y(v) + 5, "text-anchor": "end", fill: css("--muted"), "font-size": 13 }, pct(v, 0)));
    });
    xs.filter((_, i) => i % 2 === 1 || i === xs.length - 1).forEach((v) => {
      svg.append(el("text", { x: X(v), y: H - 22, "text-anchor": "middle", fill: css("--muted"), "font-size": 13 }, pct(v, 0)));
    });
    svg.append(el("text", { x: (L + W - R) / 2, y: H - 2, "text-anchor": "middle", fill: css("--muted"), "font-size": 13 }, "confidence you asked for"));

    svg.append(el("line", { x1: X(xmin), y1: Y(xmin), x2: X(xmax), y2: Y(xmax),
      stroke: css("--muted"), "stroke-width": 2, "stroke-dasharray": "2 8", "stroke-linecap": "round" }));

    const path = (key, color) => {
      const pts = r.sweep.map((p) => `${X(p.confidence)},${Y(Math.max(lo, p[key]))}`).join(" ");
      svg.append(el("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": 5,
        "stroke-linecap": "round", "stroke-linejoin": "round" }));
      r.sweep.forEach((p) => svg.append(el("circle", { cx: X(p.confidence), cy: Y(Math.max(lo, p[key])), r: 5.5, fill: color })));
    };
    const legend = $("legend");
    legend.replaceChildren();
    const addLegend = (color, text, dashed) => {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.style.background = dashed ? "transparent" : color;
      if (dashed) i.style.borderTop = `3px dotted ${color}`;
      s.append(i, text);
      legend.appendChild(s);
    };

    if (r.task === "regression") path("naive_coverage", css("--warn"));
    path("coverage", css("--accent"));
    addLegend(css("--accent"), "Calibre");
    if (r.task === "regression") addLegend(css("--warn"), "Naive range from training error");
    addLegend(css("--muted"), "Perfect promise", true);

    $("chart").replaceChildren(svg);
  }

  function drawSamples(r) {
    const t = $("samples");
    t.replaceChildren();
    const head = r.task === "regression"
      ? ["Row", "Actual", "Prediction", "Range", "Result"]
      : ["Row", "Actual", "Calibre says", "Result"];
    const thead = t.createTHead().insertRow();
    head.forEach((h) => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
    const body = t.createTBody();
    r.samples.forEach((s) => {
      const tr = body.insertRow();
      const cells = r.task === "regression"
        ? [s.row, num(s.truth), num(s.prediction), `${num(s.low)} to ${num(s.high)}`]
        : [s.row, s.truth, s.prediction.length === 1 ? s.prediction[0] : "one of: " + s.prediction.join(", ")];
      cells.forEach((c) => { tr.insertCell().textContent = c; });
      const tag = document.createElement("span");
      tag.className = "tag " + (s.hit ? "hit" : "miss");
      tag.textContent = s.hit ? "Covered" : "Missed";
      tr.insertCell().appendChild(tag);
    });
  }
})();
