# 3D Workflow Diagrams — Prompt Crash Test Lab

> **Professional 3D isometric visualizations** of the LLM Robustness Evaluation Pipeline.
> All diagrams are high-resolution (≥ 2100 × 2100 px), no-clutter, and research-paper ready.

---

## 1. Project Overview

### What this workflow represents

**Prompt Crash Test Lab** is an automated framework that evaluates how robust Large Language Models (LLMs) are when their input prompts are slightly changed. The pipeline:

1. Takes **100 base prompts** across two task types (JSON Extraction and Grounded Q&A)
2. Generates **20 semantically equivalent variants** per prompt using 5 mutation strategies
3. Executes all variants across **4 LLM providers** (GPT-4, Claude, Gemini, Llama)
4. Caches responses to control API costs via **SQLite**
5. Scores every response using **6 evaluation metrics**
6. Produces **research-grade visualizations** — diagrams, charts, and an interactive dashboard

### Purpose of the diagrams

These 3D isometric diagrams serve as:

- **System documentation** — understand the architecture at a glance
- **Research paper figures** — publication-quality visuals
- **Presentation slides** — walk stakeholders through the pipeline step-by-step
- **Onboarding material** — help new contributors understand the codebase quickly

---

## 2. Folder Structure

```
3D_Workflow_Output/
│
├── Overview/
│     └── full_workflow_3D.png          ← Complete system view
│
├── Pipeline/
│     └── workflow_pipeline_3D.png      ← Flat data-flow view with volumes
│
├── Steps/
│     ├── step_01.png                   ← Input Layer
│     ├── step_02.png                   ← Variant Generator
│     ├── step_03.png                   ← Model Execution Layer
│     ├── step_04.png                   ← Storage & Cache Layer
│     ├── step_05.png                   ← Evaluation Engine
│     └── step_06.png                   ← Visualization Layer
│
├── Components/
│     ├── component_01.png              ← Model Clients (4 LLM providers)
│     ├── component_02.png              ← Scoring System (6 metrics + formulas)
│     ├── component_03.png              ← Variant Generator (5 strategies)
│     └── component_04.png             ← Dashboard & Exports
│
├── Assets/
│     ├── Icons/                        ← Icon assets (future use)
│     ├── Textures/                     ← Texture assets (future use)
│     └── Labels/                       ← Label assets (future use)
│
└── README.md                           ← This file
```

| Folder | Contents | Best used for |
|--------|----------|---------------|
| `Overview/` | Single panoramic system view | Title slides, README header |
| `Pipeline/` | Horizontal data-flow with volume labels | Technical documentation |
| `Steps/` | One zoomed diagram per pipeline stage | Step-by-step explanations |
| `Components/` | Internal structure of key modules | Deep-dive technical docs |
| `Assets/` | Raw assets for customization | Custom branding / extensions |

---

## 3. Workflow Explanation

### End-to-End Pipeline

```
[Input Layer]
     │  100 prompts
     ▼
[Variant Generator]
     │  2,000 variants (20 × 100)
     ▼
[Model Execution Layer]
     │  ~8,000 LLM responses (2,000 × 4 models)
     ▼
[Storage & Cache]
     │  Deduplicated, persisted responses
     ▼
[Evaluation Engine]
     │  Scored results + statistical tests
     ▼
[Visualization Layer]
     └── Diagrams · Dashboard · Research exports
```

### Step-by-step explanation

#### Step 01 — Input Layer
Loads **100 base prompts** split across two task types:

| Task Type | Count | Description |
|-----------|-------|-------------|
| JSON Extraction | 50 | Extract structured fields from unstructured text |
| Grounded Q&A | 50 | Answer questions using provided context with citations |

Each prompt is paired with a **JSON schema** (for extraction tasks) or **ground truth answer** (for Q&A tasks).

---

#### Step 02 — Variant Generator
Generates **20 semantically equivalent variants** per base prompt using 5 strategies:

| Strategy | Count | Technique |
|----------|-------|-----------|
| Paraphrase | 5 | Rule-based rewording ("Please…", "Your task is to…") |
| Format | 4 | Output structure change (Markdown, Plaintext, XML, List) |
| Role | 3 | System persona change (Expert, Assistant, Teacher) |
| Constraint | 3 | Output style constraint (Concise, Detailed, Simplified) |
| Template | 5 | Prompt template (Zero-shot, Few-shot, CoT, Step, Structured) |

Each variant gets a unique 8-character ID and links back to its parent prompt via `parent_id`.

**Total:** 100 prompts × 20 variants = **2,000 variants** saved as JSONL files.

---

#### Step 03 — Model Execution Layer
Sends every variant to each of the 4 LLM providers in parallel:

| Provider | Model | Input Cost | Output Cost |
|----------|-------|-----------|------------|
| OpenAI | `gpt-4-turbo` | $0.010/1K tokens | $0.030/1K tokens |
| Anthropic | `claude-3-5-sonnet-20241022` | $0.003/1K tokens | $0.015/1K tokens |
| Google | `gemini-1.5-pro` | $0.0035/1K tokens | $0.0105/1K tokens |
| Together AI | `meta-llama/Llama-3.1-70B-Instruct-Turbo` | $0.0009/1K tokens | $0.0009/1K tokens |

Features: retry logic (3×), 1-second rate limiting, cost tracking, latency measurement.

**Total:** 2,000 × 4 = **~8,000 API responses**.

---

#### Step 04 — Storage & Cache Layer
Persists all responses to avoid redundant API calls:

- **SQLite cache** keyed by `SHA-256(prompt + model + parameters)`
- **Cache hit** → skip API call, return stored response (~30% savings on re-runs)
- **JSONL result files** written per model with token counts and latency
- **Metadata linking** preserves `variant_id → parent_id → base_prompt` chain

---

#### Step 05 — Evaluation Engine
Scores every response using 6 metrics:

| Metric | Formula | Weight (Q&A) |
|--------|---------|-------------|
| **Robustness** | `1 − (σ / μ)` across variants | Primary KPI |
| **Semantic Similarity** | Cosine similarity of MiniLM embeddings | 40% |
| **Format Compliance** | JSON schema validation + field accuracy | N/A for Q&A |
| **Answer Correctness** | Semantic + keyword + citation composite | 100% |
| **Citation Accuracy** | Regex quote matching vs. source context | 30% |
| **Cost Efficiency** | Accuracy / token usage ratio | Benchmark |

Statistical significance: **Mann-Whitney U test** (non-parametric) between model pairs.

---

#### Step 06 — Visualization Layer
Produces all output artefacts:

- `pipeline.png` — Graphviz end-to-end flow diagram
- `architecture.png` — 3D-style layered architecture
- `metrics.png` — Grouped bar chart of robustness scores
- **11 × 3D isometric diagrams** (this package)
- **Streamlit dashboard** — interactive leaderboard, heatmaps, cost analysis

---

## 4. How to Use

### Navigating the diagrams

**Start here → Overview**
Open `Overview/full_workflow_3D.png` to see the complete system at a glance. All 6 stages appear as cascading 3D blocks connected by arrows.

**Understand the flow → Pipeline**
Open `Pipeline/workflow_pipeline_3D.png` to see the flat data-flow view with exact data volumes on every connector (100 prompts → 2,000 variants → 8,000 responses → …).

**Deep-dive a stage → Steps**
Open the relevant `Steps/step_0N.png` to see the internal sub-components of any pipeline stage, with arrows showing how data flows inside that stage.

**Understand a module → Components**
Open `Components/component_0N.png` for technical detail on specific modules (model clients, scoring formulas, variant strategies, dashboard views).

### Suggested usage

| Purpose | Recommended diagrams |
|---------|---------------------|
| Conference talk title slide | `Overview/full_workflow_3D.png` |
| Research paper system figure | `Pipeline/workflow_pipeline_3D.png` |
| Technical documentation | `Steps/step_01.png` … `step_06.png` |
| Code review / PR description | `Components/component_01.png` … `component_04.png` |
| README header | `Overview/full_workflow_3D.png` |

---

## 5. Visual Guide

### Color coding

Each pipeline stage has a consistent color used across all diagrams:

| Color | Stage | Hex |
|-------|-------|-----|
| 🟦 Teal | Input Layer | `#4ECDC4` |
| 🔵 Sky Blue | Variant Generator | `#45B7D1` |
| 💠 Cornflower | Model Execution | `#7EC8E3` |
| 🟡 Gold | Storage & Cache | `#FFD93D` |
| 🟣 Lavender | Evaluation Engine | `#C9A7EB` |
| 🟢 Mint Green | Visualization Layer | `#6BCB77` |

**Model colors** (used in Component 01):

| Color | Provider |
|-------|----------|
| 🟢 Green | OpenAI / GPT-4 |
| 🟠 Orange | Anthropic / Claude |
| 🔵 Blue | Google / Gemini |
| 🟣 Purple | Together AI / Llama |

**Metric colors** (used in Component 02):

| Color | Metric |
|-------|--------|
| 🔴 Coral | Robustness Score |
| 🟦 Teal | Semantic Similarity |
| 🔵 Sky | Format Compliance |
| 🟡 Gold | Answer Correctness |
| 🟢 Green | Citation Accuracy |
| 🟣 Lavender | Cost Efficiency |

### Diagram logic

- **Top face (brightest)** → represents the active "output" surface of each block
- **Front face (medium)** → label area — read from here
- **Right face (darkest)** → depth/shadow, no labels
- **Glow halo** → indicates an "active" or "running" state
- **Grid platform** → spatial grounding to avoid floating appearance
- **Data-pipe connectors** (Pipeline view) → rectangular 3D tunnels showing data volumes
- **Arrow labels** → verb describing the transformation (`generate`, `execute`, `cache`, `score`, `render`)

### Icons

| Icon | Meaning |
|------|---------|
| 📄 | Data / file input |
| ⚙ | Processing / generation |
| 🤖 | AI model execution |
| 🗄 | Storage / database |
| 📊 | Evaluation / metrics |
| 🖼 | Visualization output |
| 💪 | Robustness metric |
| 🔗 | Semantic similarity |
| ✅ | Format validation |
| 🎯 | Correctness scoring |
| 📝 | Citation matching |
| 💰 | Cost efficiency |

---

## 6. Output Details

### File format & resolution

| Property | Value |
|----------|-------|
| Format | PNG (lossless) |
| Resolution | ≥ 2100 × 2100 px (Overview, Steps, Components) |
| Resolution | ≥ 3000 × 1650 px (Pipeline — panoramic) |
| DPI | 150 |
| Background | Dark navy `#0D1B2A` |
| Colour depth | 24-bit RGB |
| Transparency | None (opaque background) |

### 3D model files (GLB / OBJ)

3D model exports are not included in this release. The diagrams are rendered directly from Python (matplotlib mplot3d with orthographic projection) as pixel-perfect 2D images that simulate 3D via isometric perspective.

To request 3D model output, the `backend/visualization/diagram_3d/` module can be extended with Blender's Python API (`bpy`) or Open3D to export `.glb` files from the same scene geometry definitions in `utils.py`.

### Reproducing the diagrams

```bash
# Clone the repository
git clone https://github.com/dp2426-NAU/prompt-crash-test-lab.git
cd prompt-crash-test-lab

# Install only the visualization dependencies (no LLM API keys needed)
pip install matplotlib numpy

# Generate all 11 diagrams
python generate_3d_workflow.py
```

All PNGs are written to `3D_Workflow_Output/` in the exact folder structure above.

---

## 7. Quick Reference

```
Start         Overview/full_workflow_3D.png
  │
  ├─ Flow     Pipeline/workflow_pipeline_3D.png
  │
  ├─ Detail   Steps/step_01.png   Input Layer
  │            Steps/step_02.png   Variant Generator
  │            Steps/step_03.png   Model Execution
  │            Steps/step_04.png   Storage & Cache
  │            Steps/step_05.png   Evaluation Engine
  │            Steps/step_06.png   Visualization Layer
  │
  └─ Module   Components/component_01.png   Model Clients
               Components/component_02.png   Scoring System
               Components/component_03.png   Variant Generator
               Components/component_04.png   Dashboard & Exports
```
