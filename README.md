# MedCred-Bench: Evaluation Benchmark and Replication Package

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-blue.svg)]()

Evaluation benchmark, Description Logic ontology schemas, and replication scripts for the research paper:

> **Mitigating Generative Hallucinations in Compliance-Critical Medical Manufacturing Recruitment: A Tri-Agent Verification Architecture with Semantic Entropy Calibration**  
> *Status: Under Peer Review at ICDTM 2026 (ACM ICPS)*  
> *Contact: junjun ([3588@duck.com](mailto:3588@duck.com))*

---

> [!NOTE]
> **Progressive Release Notice / 开源发布说明**  
> Benchmark data, evaluation harnesses, and baseline reasoning traces are currently being **progressively curated and open-sourced**. If you have any questions, reproduction inquiries, or suggestions, please feel free to **[open an Issue](https://github.com/jhuang-icdtm2026/medcred-bench/issues)** directly or contact the authors.  
> （本项目相关的数据集、评测脚本与实验日志目前正在持续整理并逐步公开中。如有任何问题或复现疑问，欢迎直接提交 GitHub Issue。）

---

## Statutory Data Governance and Privacy Statement

Pursuant to the **Personal Information Protection Law of the People's Republic of China (PIPL, Articles 13 & 28)**, the **European Union General Data Protection Regulation (EU GDPR, Article 9)**, and corporate Non-Disclosure Agreements (NDAs):

* **Raw Proprietary Resumes**: Historical job applicant resumes collected from medical device manufacturers contain Personally Identifiable Information (PII) and are legally confidential. They cannot be released publicly on open web repositories.
* **Standardized Synthetic Benchmark**: This repository provides **`MedCred-Bench`**, an open-access, fully de-identified synthetic benchmark comprising **1,200 fine-grained credential verification tasks across 250 candidate dossiers**. It precisely models the statistical distributions, statutory job requirements, and adversarial failure modes observed in real-world regulatory recruitment pipelines.
* **Benchmark Generator**: The benchmark is deterministically reproducible via `simulate_tri_agent_uq.py::generate_medcred_bench(seed=42)`.
* **De-identified Enterprise Access**: Academic researchers requiring access to raw de-identified enterprise records may submit a formal Data Use Agreement (DUA) request to the corresponding author.

---

## Quickstart & Replication Options

This repository provides three complementary verification paths depending on your hardware environment:

### Path A: 10-Second Minimal Architecture Demo (Zero Dependencies)
To inspect the internal algorithmic mechanics of the Tri-Agent 4-branch decision logic (DL axioms, conflict graph checking, semantic entropy, and HITL routing) on concrete test cases:
```bash
python demo_minimal_pipeline.py
```

### Path B: Table 2 Reproducibility Audit (Dynamic Trace Parser)
To independently verify the confusion matrices, accuracy rates, and 95% Wilson Score / Bootstrap confidence intervals reported in Table 2:
```bash
python replicate_table2_metrics.py
```
> **Audit Note**: This script dynamically loads and parses the 960 candidate prediction traces stored in `data/eval_predictions_960.json` (or falls back to the closed-form contingency counts), recomputing every confidence interval from raw task-level decisions.

### Path C: Production Tri-Agent Evaluation Pipeline
To run the full evaluation engine across all 960 held-out evaluation tasks:
```bash
# 1. Fast deterministic audit mode (Zero GPU required):
python run_real_evaluation.py --mode offline

# 2. Live vLLM / OpenAI-compatible API mode (e.g. Qwen2.5-32B server):
python run_real_evaluation.py --mode api --api-url http://localhost:8000/v1

# 3. Quick subsample test (e.g., 10 tasks):
python run_real_evaluation.py --mode offline --sample-size 10
```

---

### Expected Table 2 Metric Summary:
* **Autonomous Processing Subset ($N = 783$, 81.6% Coverage)**:
  * Contingency Counts: $\text{TP} = 547, \; \text{FP} = 3, \; \text{TN} = 231, \; \text{FN} = 2$ ($\sum = 783$)
  * Precision: $99.45\%$ (95% Wilson CI: $[98.4\%, 99.8\%]$)
  * Recall: $99.64\%$ (95% Wilson CI: $[98.7\%, 99.9\%]$)
  * $\text{F}_1$ (positive class): $99.55\%$ (95% Bootstrap CI: $[99.1\%, 99.9\%]$)
  * Residual Hallucination Rate: $0.38\% \approx 0.4\%$ (95% Wilson CI: $[0.1\%, 1.1\%]$)
* **Full Pipeline with HITL Adjudication ($N = 960$, 100% Coverage)**:
  * Contingency Counts: $\text{TP} = 484, \; \text{FP} = 13, \; \text{TN} = 436, \; \text{FN} = 27$ ($\sum = 960$)
  * Precision: $97.4\%$ (95% Wilson CI: $[95.6\%, 98.5\%]$)
  * Recall: $94.7\% \approx 94.8\%$ (95% Wilson CI: $[92.4\%, 96.3\%]$)
  * $\text{F}_1$ (positive class): $96.0\% \approx 96.1\%$ (95% Bootstrap CI: $[94.7\%, 97.2\%]$)
  * Residual Hallucination Rate: $0.4\%$ (95% Wilson CI: $[0.2\%, 1.1\%]$)
  * Human Escalation Rate (HITL): $18.4\%$ ($177/960$, 95% Wilson CI: $[16.1\%, 21.0\%]$)

---

## Repository Structure

```text
medcred-bench/
├── README.md                      # Comprehensive documentation and reproducibility guide
├── LICENSE                        # MIT Open-Source License
├── requirements.txt               # Optional environment dependencies
├── demo_minimal_pipeline.py       # Minimal reproducible demo of 4-branch architecture
├── replicate_table2_metrics.py    # Dynamic metric auditor parsing predictions JSON
├── run_real_evaluation.py         # Full Tri-Agent evaluation pipeline (offline/API/HF)
├── simulate_tri_agent_uq.py       # Monte Carlo statistical simulation environment
├── data/
│   ├── medcred_bench_1200_tasks.json     # All 1,200 fine-grained credential tasks
│   ├── benchmark_split_ids.json          # Strict 20%/80% dossier-level partition IDs
│   ├── eval_predictions_960.json         # Complete 960 held-out prediction & trace records
│   └── sample_qualitative_traces.json    # 3 representative adversarial case traces
└── ontology/
    └── statutory_dl_schemas.json         # Description Logic (ALCQIO) axioms & registrars
```

---

## Benchmark Specification (`MedCred-Bench`)

`MedCred-Bench` models multi-agent verification under high-stakes medical device compliance requirements (e.g., ISO 13485:2016, ISO 14644 cleanroom, China NMPA Decree No. 739, U.S. FDA 21 CFR Part 820 QMSR).

### 1. Benchmark Composition (1,200 Tasks across 250 Dossiers)
* **200 Authentic-Profile Dossiers (1,000 Tasks)**:
  * **Track 1: Cleanroom GMP & Sterilization** (50 dossiers, 250 tasks)
  * **Track 2: ISO 13485 Quality Assurance & V&V** (50 dossiers, 250 tasks)
  * **Track 3: Class II / Class III Regulatory Affairs** (50 dossiers, 250 tasks)
  * **Track 4: Medical Electronics & Biocompatibility** (50 dossiers, 250 tasks)
* **50 Adversarial Stress Dossiers (200 Tasks)**:
  * **Type I: Fabricated Certification Injection** (15 dossiers, 60 tasks)
  * **Type II: Spatiotemporal Collision & Overlap** (15 dossiers, 60 tasks)
  * **Type III: Scope-of-Practice Distortion** (10 dossiers, 40 tasks)
  * **Type IV: Adversarial Injection & Subtle Boundary Tampering** (10 dossiers, 40 tasks)

### 2. Zero-Leakage 20%/80% Dossier-Level Split
Partitioning is enforced strictly at the **dossier level** to avoid cross-task contamination:
* **Calibration Split (20%)**: 50 dossiers (240 tasks: 40 authentic + 10 adversarial) dedicated to calibrating the epistemic safety gating threshold $\tau_{\text{safe}} = 0.35$.
* **Held-Out Evaluation Split (80%)**: 200 dossiers (960 tasks: 160 authentic + 40 adversarial) held out exclusively for unbiased evaluation.

---

## Tri-Agent Architecture Overview

```text
Raw Candidate Dossier (D)
        │
        ▼
[Stage 1: Extractor Agent (A_ext)]  ──► Grammar-constrained context-free decoding (JSON)
        │
        ▼
[Stage 2: Ontological Verifier (A_ver)] ──► Description Logic ALCQIO axioms (Phi(c, K))
        │                              │ (Hard Reject if accredited registrar missing)
        ▼                              ▼
[Stage 3: Conflict Auditor (A_aud)] ──► Spatiotemporal conflict graph (|E_conflict| == 0)
        │                              │ (Hard Reject if geodesic overlap > 150km)
        ▼                              ▼
Deterministic Pass? ──────────────────► NO ──► Branch 1: Reject (Deterministic Halt)
        │ YES
        ▼
[Stage 4: Semantic Entropy Calibrator]
        │ M=5 reasoning paths sampled (T=0.7)
        │ Bidirectional NLI clustering (DeBERTa-v3-large-MNLI)
        ▼
SE(D, q) > tau_safe (0.35)?
        ├── YES ───────────────────────► Branch 2: Route to HITL (Human Escalation)
        └── NO
             ├── Consensus == 1 ───────► Branch 3: Approve (Autonomous Clearance)
             └── Consensus == 0 ───────► Branch 4: Reject (Autonomous Rejection)
```
