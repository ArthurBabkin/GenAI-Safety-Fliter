---
marp: true
theme: poster-theme
paginate: false
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

  section {
    padding: 0;
    margin: 0;
    background: #f0ede6;
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
    color: #1a1a2e;
    overflow: visible;
    display: flex;
    flex-direction: column;
    height: 1700px;
  }

  /* ── TITLE BAND ── */
  .poster-title {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d5a40 60%, #1e4060 100%);
    padding: 16px 40px 12px;
    text-align: center;
    flex-shrink: 0;
    border-bottom: 3px solid #4a9060;
  }
  .poster-title h1 {
    color: #ffffff;
    font-size: 30px;
    font-weight: 900;
    margin: 0 0 3px;
    letter-spacing: -0.5px;
    line-height: 1.1;
  }
  .poster-title h2 {
    color: #a8d8b8;
    font-size: 15px;
    font-weight: 400;
    margin: 0 0 4px;
  }
  .poster-title .authors {
    color: #c8e8d8;
    font-size: 12px;
    font-weight: 600;
    margin: 0;
  }

  /* ── GRID ── */
  .poster-grid {
    display: grid;
    grid-template-columns: 1fr 3fr;
    gap: 7px;
    padding: 7px;
    flex: 1;
    min-height: 0;
    align-items: stretch;
  }
  .col1 { display: flex; flex-direction: column; gap: 7px; }
  .col1 .panel { margin-bottom: 0; }
  .col-rest {
    columns: 3;
    column-fill: auto;
    column-gap: 7px;
    height: 1550px;
    overflow: hidden;
  }

  /* ── PANELS ── */
  .panel {
    background: #ffffff;
    border-radius: 7px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    break-inside: avoid;
    margin-bottom: 7px;
  }
  .story-flow { break-inside: avoid; margin-bottom: 7px; }
  .panel-header {
    padding: 5px 10px;
    font-size: 10.5px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .panel-body {
    padding: 7px 10px;
    flex: 1;
    overflow: hidden;
  }
  .ph-green, .ph-blue, .ph-purple, .ph-maroon, .ph-neutral, .ph-teal { background: #3a3a4a; }

  /* ── STAT CARDS ── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    margin-bottom: 5px;
  }
  .stat-card {
    background: #f5f5f5;
    border-radius: 5px;
    padding: 5px 3px;
    text-align: center;
    border: 1px solid #e0e0e0;
  }
  .stat-num { font-size: 17px; font-weight: 900; color: #1a1a2e; display: block; line-height: 1.1; }
  .stat-lbl { font-size: 8.5px; color: #667788; margin-top: 2px; line-height: 1.3; }

  /* ── SEGMENTED BARS ── */
  .seg-bar {
    display: flex; height: 20px; border-radius: 4px; overflow: hidden;
    margin: 3px 0; font-size: 9px; font-weight: 700; color: white;
  }
  .seg-bar div { display: flex; align-items: center; justify-content: center; }
  .ru   { background: #7744bb; flex: 55.7; }
  .en   { background: #3366aa; flex: 44.3; }
  .safe { background: #3d7a52; flex: 80.1; }
  .tox  { background: #b03838; flex: 19.9; font-size: 8px; }

  /* ── AXIS CARDS ── */
  .axes-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; margin-bottom: 5px; }
  .axis-card {
    border-radius: 5px; padding: 6px 7px; border-left: 3px solid #c0c0c0; font-size: 9.5px; line-height: 1.4;
    background: #f5f5f5; border: 1px solid #e0e0e0;
  }
  .axis-card b { display: block; font-size: 10.5px; margin-bottom: 2px; color: #1a1a2e; }
  .ax-blue  { }
  .ax-blue b { }
  .ax-teal  { }
  .ax-teal b { }
  .ax-red   { }
  .ax-red b { }

  /* ── TABLES ── */
  .pt { width: 100%; border-collapse: collapse; font-size: 10px; margin: 4px 0; }
  .pt th { color: #1a1a2e; padding: 4px 5px; text-align: center; font-weight: 700; font-size: 9.5px; border-bottom: 2px solid #c0c0c0; background: none; }
  .pt td { padding: 3px 5px; text-align: center; border-bottom: 1px solid #e8e8e8; }
  .pt td:first-child { text-align: left; }
  .pt tr.hl  td { font-weight: 700; }
  .pt tr.warn td { }
  .th-green, .th-blue, .th-purple, .th-neutral { background: none; }

  /* ── IMAGES ── */
  .panel-img { width: 100%; height: 210px; object-fit: contain; display: block; border-radius: 3px; margin: 3px 0; background: #fafafa; }

  /* ── TEXT ── */
  .caption { font-size: 9px; color: #555; line-height: 1.4; margin: 3px 0; }
  .body-sm { font-size: 10px; line-height: 1.5; margin: 3px 0; }
  .body-sm b { color: #1e4060; }

  /* ── SECTION BANNER ── */
  .sec-banner {
    border-radius: 6px; padding: 5px 10px 6px; margin-bottom: 6px; flex-shrink: 0;
  }
  .sec-banner .sec-num {
    font-size: 8.5px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; opacity: 0.8; display: block; margin-bottom: 1px;
  }
  .sec-banner .sec-title { font-size: 11.5px; font-weight: 800; display: block; line-height: 1.2; }
  .sec-banner .sec-desc  { font-size: 8.5px; opacity: 0.8; margin-top: 2px; display: block; line-height: 1.4; }
  .sb-blue  { background: #1e4060; color: white; border-left: 4px solid #5a9fd4; }
  .sb-green { background: #2d5a40; color: white; border-left: 4px solid #6abf84; }
  .sb-neutral { background: #3a3a4a; color: white; border-left: 4px solid #aaa; }

  /* ── BADGE ── */
  .badge-row { display: flex; flex-direction: column; gap: 4px; margin: 4px 0; }
  .badge {
    border-radius: 4px; padding: 5px 8px; font-size: 9.5px; font-weight: 600; line-height: 1.4;
  }
  .badge b { font-size: 10px; }
  .badge-green, .badge-blue, .badge-red, .badge-gold { background: #f5f5f5; color: #1a1a2e; border-left: 3px solid #c0c0c0; }

  /* ── DELTA CARD ── */
  .delta-row { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; margin: 5px 0; }
  .delta-card { border-radius: 4px; padding: 5px 8px; font-size: 9px; line-height: 1.4; background: #f7f7f7; border: 1px solid #e0e0e0; }
  .dc-blue  { }
  .dc-blue b { font-size: 10px; display: block; }

  /* ── CONCLUSIONS BAND ── */
  .conclusion-item {
    display: flex; align-items: flex-start; gap: 6px; margin: 4px 0;
    font-size: 9.5px; line-height: 1.45;
  }
  .c-dot {
    min-width: 7px; height: 7px; border-radius: 50%; margin-top: 3px;
  }
  .cd-green  { background: #2d5a40; }
  .cd-blue   { background: #1e4060; }
  .cd-red    { background: #aa3030; }
  .cd-gold   { background: #c8a020; }

  /* ── STORY ARROW ── */
  .story-flow {
    background: #2d2d3a;
    color: #ccc;
    font-size: 8.5px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    text-align: center;
    padding: 4px 0;
    flex-shrink: 0;
    border-radius: 5px;
    margin-bottom: 6px;
  }
  .story-flow span { color: #6abf84; }

  p, ul, li, h1, h2, h3 { margin: 0; padding: 0; }
  .sp { height: 4px; }
---

<div class="poster-title">
<h1>Parameter-Efficient Transformer Fine-tuning for Bilingual Toxic Text Classification & Regression </h1>
<h2>LoRA Adapters vs Full Fine-tuning — Quality · Speed · Robustness on a Noisy RU/EN Corpus</h2>
<div class="authors">Arthur Babkin &nbsp;&nbsp;|&nbsp;&nbsp; Innopolis University &nbsp;&nbsp;|&nbsp;&nbsp; NLP 2026 &nbsp;·&nbsp; Case Study 2.6</div>
</div>

<div class="poster-grid">

<div class="col1">
<div class="story-flow">① <span>Setup</span> — question, data, models</div>

<div class="panel">
<div class="panel-header ph-neutral">The Question</div>
<div class="panel-body">
<p class="caption" style="margin-bottom:5px;"><b>Case Study 2.6 — Parameter-Efficient Transformer Fine-tuning.</b> Take a small pretrained transformer LM, apply LoRA layers to specialize it for a niche domain corpus, and document the trade-offs between adaptation speed, GPU memory footprint, and downstream perplexity. NLP course case study at Innopolis University. Topic overlaps with the <b>Generative AI</b> course — Alexander Malyy, a fellow Generative AI student, assisted with select tasks.</p>
<div class="axes-row">
<div class="axis-card ax-blue"><b>Quality</b>F1 on subtle, context-dependent toxic content</div>
<div class="axis-card ax-teal"><b>Speed</b>Training cost and inference latency</div>
<div class="axis-card ax-red"><b>Robustness</b>l33tspeak, char spacing, homoglyphs</div>
</div>
<p class="body-sm"><b>Can LoRA — training only 1% of parameters — match full fine-tuning on all three axes?</b> We answer this with two independent experiments:</p>
<div class="badge-row" style="margin-top:5px;">
<div class="badge badge-green" style="font-size:9px;"><b>Exp 1 — Classification (sections ②④):</b> DistilBERT full FT vs LoRA vs TF-IDF on toxic-text detection. Metrics: F1, EN/RU gap, obfuscation robustness, class-imbalance ablations.</div>
<div class="badge badge-blue" style="font-size:9px;"><b>Exp 2 — Autoregressive LM (section ③):</b> DistilGPT-2 full FT vs LoRA rank sweep (r=4/8/16). Metrics: perplexity, training time, memory, checkpoint size.</div>
</div>
</div>
</div>

<div class="panel">
<div class="panel-header ph-neutral">Dataset</div>
<div class="panel-body">
<div class="stats-row">
<div class="stat-card"><span class="stat-num">460K</span><span class="stat-lbl">samples<br>after LSH</span></div>
<div class="stat-card"><span class="stat-num">4</span><span class="stat-lbl">Kaggle<br>datasets</span></div>
<div class="stat-card"><span class="stat-num">92K</span><span class="stat-lbl">test set<br>seed=42</span></div>
<div class="stat-card"><span class="stat-num">2</span><span class="stat-lbl">languages<br>RU + EN</span></div>
</div>
<div class="seg-bar"><div class="ru">Russian 56%</div><div class="en">English 44%</div></div>
<div class="seg-bar"><div class="safe">Safe 80%</div><div class="tox">Toxic 20%</div></div>
<p class="caption" style="margin-top:4px;">HTML/URLs stripped. MinHash-LSH t=0.80 removed near-duplicates. One train/val/test split (seed=42) shared by all models.</p>
</div>
</div>

<div class="panel" style="flex:1;">
<div class="panel-header ph-neutral">Models &amp; Setup</div>
<div class="panel-body">

<p class="caption" style="font-weight:700;font-size:10px;margin:0 0 1px;">Exp 1 — Classification</p>
<table class="pt" style="margin:0 0 1px;">
<thead><tr><th>Model</th><th>Params</th><th>Data</th></tr></thead>
<tbody>
<tr><td>TF-IDF + LogReg</td><td>10K feat</td><td>133K bal.</td></tr>
<tr><td>DistilBERT full FT</td><td>67.6M</td><td>331K</td></tr>
<tr><td>DistilBERT + LoRA r=8</td><td>739K (1.09%)</td><td>90K bal.</td></tr>
</tbody>
</table>
<p class="caption" style="margin:0 0 6px;">LoRA: α=16, q/v projection. Merged before inference → serving cost = full FT.</p>

<p class="caption" style="font-weight:700;font-size:10px;margin:0 0 1px;">Ablations (Exp 1)</p>
<table class="pt" style="margin:0 0 6px;">
<thead><tr><th>#</th><th>Question</th><th>Change vs baseline</th></tr></thead>
<tbody>
<tr><td>1</td><td>Does class balancing help?</td><td>All 3 models, undersample majority</td></tr>
<tr><td>2</td><td>Does weighted loss help?</td><td>DistilBERT, 4× toxic weight</td></tr>
<tr><td>3</td><td>Data size vs architecture?</td><td>LoRA vs full FT, both at 90K</td></tr>
<tr><td>4</td><td>Robust to obfuscation?</td><td>All 3 + rule-based deobfuscation</td></tr>
</tbody>
</table>



<p class="caption" style="font-weight:700;font-size:10px;margin:10px 0 1px;">Exp 2 — Autoregressive LM</p>
<table class="pt" style="margin:0 0 1px;">
<thead><tr><th>Model</th><th>Params</th><th>Data</th></tr></thead>
<tbody>
<tr><td>DistilGPT-2 full FT</td><td>82M (100%)</td><td>90K</td></tr>
<tr><td>DistilGPT-2 + LoRA r=4/8/16</td><td>73–295K</td><td>90K</td></tr>
</tbody>
</table>
<p class="caption" style="margin:0;">Rank sweep: same data, 1 epoch. Metrics: PPL, time, memory, checkpoint size.</p>

</div>
</div>

<div class="panel">
<div class="panel-header ph-neutral">Code &amp; Reproduce</div>
<div class="panel-body">
<p class="caption">Full implementation, training scripts, and experiment logs:</p>
<p class="caption" style="margin-top:4px;font-family:monospace;font-size:8.5px;word-break:break-all;"><a href="https://github.com/ArthurBabkin/GenAI-Safety-Fliter/tree/nlp-case-study" style="color:#1e4060;">github.com/ArthurBabkin/GenAI-Safety-Filter/tree/nlp-case-study</a></p>
</div>
</div>
</div>

<div class="col-rest">
<div class="story-flow">② <span>Exp 1: Classification</span> — does LoRA match full FT on F1, speed, robustness?</div>

<div class="panel" style="flex:1.2;">
<div class="panel-header ph-green">Headline Results (92K test set)</div>
<div class="panel-body">
<img class="panel-img" src="images/quality_comparison.png">
<table class="pt">
<thead><tr class="th-green"><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>Latency</th><th>Throughput</th></tr></thead>
<tbody>
<tr><td>TF-IDF + LogReg</td><td>0.805</td><td>0.835</td><td>0.778</td><td><b>0.009 ms</b></td><td><b>112K/s</b></td></tr>
<tr class="hl"><td>DistilBERT full FT</td><td><b>0.932</b></td><td><b>0.943</b></td><td><b>0.921</b></td><td>3.58 ms</td><td>287/s</td></tr>
<tr class="warn"><td>DistilBERT + LoRA</td><td>0.804</td><td>0.836</td><td>0.773</td><td>4.51 ms</td><td>226/s</td></tr>
</tbody>
</table>
<p class="caption" style="margin-top:3px;"><b>Answer: No.</b> Full FT leads by <b>+12.8pp F1</b>. LoRA has transformer-class latency (4.5 ms) at LogReg-class quality (F1 0.80) — worst-of-both-worlds for serving. Its wins are in training only.</p>
</div>
</div>

<div class="panel">
<div class="panel-header ph-green">Speed vs Quality Trade-off</div>
<div class="panel-body">
<img class="panel-img" src="images/performance_comparison.png">
<p class="caption" style="margin-top:3px;">LoRA occupies the worst quadrant: transformer-class latency (4.5 ms) at LogReg-class quality (F1 0.80). Training benefits don't survive into serving.</p>
</div>
</div>

<div class="panel">
<div class="panel-header ph-green">Per-language: EN vs RU</div>
<div class="panel-body">
<table class="pt">
<thead><tr class="th-green"><th>Model</th><th>EN F1</th><th>RU F1</th><th>Gap</th></tr></thead>
<tbody>
<tr><td>TF-IDF + LogReg</td><td>0.828</td><td>0.785</td><td>4.3pp</td></tr>
<tr class="hl"><td>DistilBERT</td><td><b>0.936</b></td><td><b>0.928</b></td><td><b>0.9pp</b></td></tr>
<tr class="warn"><td>LoRA</td><td>0.849</td><td>0.761</td><td><b>8.8pp</b></td></tr>
</tbody>
</table>
<p class="caption" style="margin-top:4px;">Full FT (67.6M trainable params) realigns the English-only backbone for Cyrillic. LoRA's 1% parameter budget is insufficient — the English bias leaks through, yielding the largest EN/RU gap of all three models.</p>
</div>
</div>


<div class="story-flow">③ <span>Exp 2: Autoregressive LM</span> — DistilGPT-2 rank sweep: speed · memory · perplexity</div>

<div class="panel">
<div class="panel-header ph-blue">Perplexity Results by Rank</div>
<div class="panel-body">
<p class="caption" style="margin-bottom:5px;">DistilGPT-2 (82M) · 90K texts · 1 epoch · AdamW + linear warmup · batch 16 · Apple MPS</p>
<table class="pt" style="margin-bottom:6px;">
<thead><tr class="th-blue"><th>Config</th><th>Trainable</th><th>PPL ↓</th><th>ΔPPL</th><th>Time</th><th>Mem MB</th><th>Ckpt MB</th></tr></thead>
<tbody>
<tr><td>Base (no FT)</td><td>—</td><td>27.5</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>LoRA r=4</td><td>73K</td><td>21.4</td><td>+6.1</td><td>30 min</td><td>999</td><td>0.3</td></tr>
<tr><td>LoRA r=8</td><td>147K</td><td>20.9</td><td>+6.7</td><td>33 min</td><td>988</td><td>0.6</td></tr>
<tr><td>LoRA r=16</td><td>295K</td><td>20.1</td><td>+7.4</td><td>35 min</td><td>985</td><td>1.1</td></tr>
<tr class="hl"><td>Full FT</td><td>82M (100%)</td><td><b>17.1</b></td><td><b>+10.4</b></td><td>43 min</td><td>2160</td><td>312</td></tr>
</tbody>
</table>
<img class="panel-img" src="images/lm_perplexity_tradeoff.png">
</div>
</div>

<div class="panel" style="flex:1;">
<div class="panel-header ph-blue">Three Trade-offs Explained</div>
<div class="panel-body">
<div class="delta-row">
<div class="delta-card dc-blue">
<b>Speed</b>
All LoRA ranks finish in ~30 min. Full FT: 43 min. Rank has <i>no effect</i> on speed — r=4 ≈ r=16.
</div>
<div class="delta-card dc-blue">
<b>Memory</b>
LoRA ~990 MB (any rank). Full FT: 2160 MB. <b>2.2× savings</b> — memory is rank-independent.
</div>
<div class="delta-card dc-blue">
<b>Perplexity</b>
Rank <i>does</i> improve PPL: r=4→21.4, r=16→20.1. Full FT: 17.1. Rank is the quality knob.
</div>
<div class="delta-card dc-blue">
<b>Checkpoint</b>
LoRA r=16: 1.1 MB vs 312 MB full FT. <b>285× smaller</b> — key for multi-variant deployment.
</div>
</div>
<p class="caption"><b>Insight:</b> LoRA decouples quality from cost. Raise rank to lower perplexity with no speed or memory penalty. Full FT wins on quality but requires 2× memory and a 285× larger checkpoint.</p>
</div>
</div>


<div class="story-flow">④ <span>Exp 1 Deep Dive</span> — classification ablations → joint conclusions</div>

<div class="panel">
<div class="panel-header ph-green">Abl. 1: Class Imbalance</div>
<div class="panel-body">
<table class="pt">
<thead><tr class="th-green"><th>Model</th><th>Baseline</th><th>Balanced</th><th>Δ</th></tr></thead>
<tbody>
<tr class="hl"><td>LogReg</td><td>0.765</td><td><b>0.805</b></td><td><b>+4.1pp</b></td></tr>
<tr class="warn"><td>DistilBERT</td><td><b>0.932</b></td><td>0.884</td><td>−4.7pp</td></tr>
<tr class="hl"><td>LoRA (same size)</td><td>0.793</td><td><b>0.804</b></td><td><b>+1.1pp</b></td></tr>
</tbody>
</table>
<p class="caption" style="margin-top:3px;">Effect is model-capacity dependent: linear models gain from balanced priors (+4.1pp), while pretrained transformers on 331K show no majority-class bias — undersampling only removes useful training signal (−4.7pp).</p>
</div>
</div>

<div class="panel">
<div class="panel-header ph-green">Abl. 2: Class Weighting</div>
<div class="panel-body">
<img class="panel-img" src="images/class_weighting_bar.png">
<img class="panel-img" src="images/class_weighting_confusion.png">
<p class="caption" style="margin-top:3px;">4× toxic weight: F1 <b>−3.8pp</b>, PR-AUC −2.1pp, threshold shifts 0.53→0.83. DistilBERT already achieves 92% recall on unweighted data — loss reweighting assumes a bias that doesn't exist, inflating toxic-class probabilities and breaking calibration.</p>
</div>
</div>

<div class="panel">
<div class="panel-header ph-green">Abl. 3: Architecture vs Data Size</div>
<div class="panel-body">
<img class="panel-img" src="images/arch_ablation_bar.png">
<img class="panel-img" src="images/arch_ablation_confusion.png">
<table class="pt">
<thead><tr class="th-green"><th>Model</th><th>Recall</th><th>F1</th><th>PR-AUC</th></tr></thead>
<tbody>
<tr><td>LoRA 90K</td><td>0.732</td><td>0.793</td><td>0.885</td></tr>
<tr class="hl"><td>Full FT 90K</td><td><b>0.824</b></td><td><b>0.864</b></td><td><b>0.935</b></td></tr>
<tr><td>Δ (arch.)</td><td>+0.092</td><td><b>+7.1pp</b></td><td>+0.050</td></tr>
</tbody>
</table>
<p class="caption" style="margin-top:3px;"><b>13pp total gap decomposes evenly: ~7pp architecture + ~7pp data volume.</b> Recall is hit hardest (+9.2pp gap), showing LoRA's constrained parameter budget limits learning of minority-class discriminative features.</p>
</div>
</div>

<div class="panel">
<div class="panel-header ph-neutral">Abl. 4: Robustness Under Obfuscation</div>
<div class="panel-body">
<img class="panel-img" src="images/robustness_f1_degradation.png">
<table class="pt">
<thead><tr><th>Model</th><th>Clean</th><th>Obf.</th><th>+Deobf.</th><th>Recov.</th></tr></thead>
<tbody>
<tr><td>LogReg</td><td>0.805</td><td>0.737</td><td>0.756</td><td>26.8%</td></tr>
<tr class="hl"><td>DistilBERT</td><td><b>0.932</b></td><td>0.840</td><td><b>0.910</b></td><td><b>75.8%</b></td></tr>
<tr><td>LoRA</td><td>0.804</td><td>0.728</td><td>0.781</td><td>69.5%</td></tr>
</tbody>
</table>
<p class="caption" style="margin-top:3px;"><b>Zero retraining needed.</b> A rule-based deobfuscation preprocessor recovers 76% of DistilBERT's obfuscation drop — high-value defense against evasion. TF-IDF has no char-level representation: only 27% recovery regardless.</p>
</div>
</div>

<div class="panel" style="flex:1;">
<div class="panel-header ph-maroon">Conclusions</div>
<div class="panel-body">
<p class="caption" style="font-weight:700;margin-bottom:3px;color:#2d5a40;">Exp 1 — Classification</p>
<div class="badge-row" style="margin-bottom:5px;">
<div class="badge badge-green" style="font-size:9px;"><b>Full FT wins on quality:</b> F1 0.932, EN/RU gap 0.9pp, 75.8% robustness recovery. Best when compute allows.</div>
<div class="badge badge-red" style="font-size:9px;"><b>LoRA = worst-of-both for serving:</b> transformer latency (4.5 ms) at LogReg-class F1 (0.80). Benefits only during training.</div>
<div class="badge badge-gold" style="font-size:9px;"><b>Free robustness:</b> rule-based deobfuscation recovers 76% of transformer drop — zero retraining.</div>
</div>
<p class="caption" style="font-weight:700;margin-bottom:3px;color:#1e4060;">Exp 2 — Autoregressive LM</p>
<div class="badge-row" style="margin-bottom:5px;">
<div class="badge badge-blue" style="font-size:9px;"><b>LoRA rank = quality dial:</b> r=4→21.4 PPL, r=16→20.1 PPL — no speed or memory penalty. Rank is the only quality knob.</div>
<div class="badge badge-blue" style="font-size:9px;"><b>Resource savings are rank-independent:</b> 2.2× less memory, 285× smaller checkpoint vs full FT. Decisive for multi-variant deployment.</div>
</div>
<p class="caption" style="font-weight:700;margin-bottom:3px;color:#555;">Joint Takeaway</p>
<div class="badge-row">
<div class="badge badge-gold" style="font-size:9px;"><b>LoRA's 1% parameter budget has a hidden multilingual cost:</b> EN/RU gap 8.8pp vs 0.9pp for full FT — frozen English-biased weights can't be realigned without full gradient flow. Critical for any non-English deployment.</div>
<div class="badge badge-green" style="font-size:9px;"><b>Use LoRA when:</b> rapid prototyping, compute-constrained training, or serving many domain variants from one base. <b>Use full FT when:</b> production quality, multilingual corpus, or maximum recall is required.</div>
</div>
</div>
</div>

<div class="panel">
<div class="panel-header ph-neutral">Limitations &amp; Future Work</div>
<div class="panel-body">
<div class="badge-row">
<div class="badge badge-red" style="font-size:9px;"><b>Domain mismatch.</b> Social-media comments only. LLM-output moderation unverified — different register and length distribution.</div>
<div class="badge badge-red" style="font-size:9px;"><b>English-centric backbone.</b> XLM-R or mDeBERTa would close the EN/RU gap by design; distilbert-base-uncased was chosen for speed.</div>
<div class="badge badge-red" style="font-size:9px;"><b>Single-seed training.</b> No variance across random seeds; small numeric deltas are indicative only.</div>
<div class="badge badge-red" style="font-size:9px;"><b>Adversarial scope.</b> 4 rule-based obfuscation families (leetspeak, spacing, repetition, mixed case). No paraphrase or semantic attacks tested.</div>
<div class="badge badge-red" style="font-size:9px;"><b>Binary label collapse.</b> Severity level, target group, and context discarded; all 4 source datasets reduced to safe/toxic.</div>
<div class="badge badge-red" style="font-size:9px;"><b>Regression not evaluated.</b> Toxicity severity as a continuous score (ordinal regression / MSE objective) was not explored — binary framing may discard signal useful for ranked moderation queues.</div>
<div class="badge badge-red" style="font-size:9px;"><b>Rule-based deobfuscation.</b> Brittle to novel evasion patterns; a learned character-level normalizer is natural future work.</div>
</div>
</div>
</div>
</div>
</div>
