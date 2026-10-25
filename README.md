# A Self-Evolving Explainable Rational Agent for Adaptive Industrial Decision Intelligence Using Statistical Trend Learning

## 1. Project Overview

This project investigates a self-evolving, explainable rational agent for adaptive industrial decision intelligence using statistical trend learning.

The proposed framework combines predictive maintenance, remaining useful life (RUL) estimation, temporal feature engineering, statistical trend analysis, explainability, uncertainty assessment, risk-aware decision-making, and safety-gated adaptation.

The objective is to investigate how an industrial decision-support agent can use predictive evidence and observed changes in operating conditions to evaluate maintenance decisions and adapt its policies under controlled experimental conditions.

The project evaluates two distinct industrial maintenance datasets:

- **AI4I 2020 Predictive Maintenance Dataset:** Used for predictive-maintenance classification and associated evaluation.
- **NASA C-MAPSS FD001 Dataset:** Used for remaining useful life prediction, temporal analysis, adaptive maintenance policies, and controlled adaptation experiments.

These datasets serve different evaluation purposes. Their results are therefore analyzed separately rather than treated as directly interchangeable measurements.

## 2. Research Objectives

The main objectives are to:

1. Develop a modular framework for industrial predictive maintenance and decision support.
2. Investigate statistical trend learning for identifying changes in temporal operating patterns.
3. Incorporate predictive outputs into risk-aware maintenance decisions.
4. Examine explainability and error analysis to support the interpretation of model behavior.
5. Investigate uncertainty-aware decision-making and safety-constrained adaptation.
6. Evaluate policy robustness through controlled experiments and repeated random seeds.
7. Maintain traceable experimental outputs to support reproducibility and research analysis.

## 3. Main Components

The project is organized around the following functional components.

### 3.1 Data preprocessing

Prepares dataset inputs for the corresponding prediction and evaluation tasks. Preprocessing is performed separately for the AI4I and C-MAPSS workflows.

### 3.2 Predictive modeling

Supports predictive-maintenance classification on AI4I and RUL prediction on C-MAPSS.

### 3.3 Statistical trend learning

Investigates temporal patterns and changes in operating signals to support the analysis of degradation, changing conditions, and adaptation.

### 3.4 Temporal feature engineering

Extracts temporal information for use in the relevant prediction and decision-making experiments.

### 3.5 Explainability and error analysis

Examines model errors and explanatory evidence to help interpret predictions and identify potential weaknesses.

### 3.6 Uncertainty and risk-aware decisions

Investigates how uncertainty and estimated risk can inform maintenance policies and decision thresholds.

### 3.7 Adaptive maintenance policies

Evaluates maintenance decision policies and investigates how policy behavior changes under specified experimental conditions.

### 3.8 Safety-gated adaptation

Evaluates controlled adaptation mechanisms that accept or reject candidate policy changes according to defined criteria. A safety gate is an experimental safeguard, not a guarantee of safety under all conditions.

### 3.9 Robustness and evidence audits

Uses repeated-seed experiments, statistical comparisons, validation checks, and evidence-synchronization procedures to assess the consistency and traceability of experimental results.

## 4. Datasets and Evaluation Tracks

### Track A: AI4I 2020 Predictive Maintenance

**Purpose:** Predictive-maintenance classification.

This track investigates the use of machine and process variables to predict maintenance-related outcomes.

The workflow includes relevant preprocessing, predictive modeling, evaluation metrics, and associated outputs.

Reported classifier metrics from the project include:

- ROC-AUC: 0.9709
- Average precision (AP): 0.8147
- Brier score after isotonic calibration: 0.01173
- Brier score after sigmoid calibration: 0.01246

These are previously reported project results and should be interpreted in the context of the exact evaluation split, preprocessing procedure, model configuration, and calibration protocol used to produce them.

### Track B: NASA C-MAPSS FD001

**Purpose:** Remaining useful life prediction and adaptive maintenance-policy evaluation.

This track investigates temporal degradation modeling, RUL-related decisions, statistical trend analysis, policy adaptation, safety gates, and robustness under controlled experimental conditions.

Reported experimental components include:

- RUL prediction and temporal feature engineering.
- Explainability and error analysis.
- Uncertainty-aware risk decisions.
- Adaptive and safety-constrained maintenance policies.
- Repeated-seed robustness evaluation.
- Controlled sensor-shift experiments.
- Comparative policy evaluation and evidence audits.

The reported repeated-seed evaluation used ten seeds:

`101, 202, 303, 404, 505, 606, 707, 808, 909, 1010`

The reported safety-gated experiment accepted adaptation in 4 of 10 seeds and rejected it in 6 of 10.

No paired statistical comparison in that reported repeated-seed evaluation established a statistically significant improvement at the 0.05 significance level. Accordingly, these experiments should be presented as an evaluation of controlled adaptation and safety gating, not as proof that adaptation always improves performance.

## 5. Repository Organization

The intended logical organization is:

```text
self-evolving-industrial-agent/
│
├── README.md
├── PACKAGE_INVENTORY.txt
│
├── AI4I/
│   ├── src/
│   ├── data/                 # Only data permitted for redistribution
│   ├── outputs/
│   └── documentation/
│
└── CMAPSS/
    ├── src/
    ├── data/                 # Only data permitted for redistribution
    ├── outputs/
    └── documentation/
```

This is the recommended logical structure. Actual filenames and subdirectories may differ in the current uploaded project.

Keep the two evaluation tracks clearly separated. Preserve the original relative paths of scripts and their associated configuration files, inputs, and outputs when organizing the repository.

The inventory file records package contents; it should not be interpreted as independent verification of every experimental result.

## 6. Software Requirements

The project uses Python scripts and data-processing and machine-learning workflows.

The precise software requirements should be taken from the project files. Before running the project on a new machine, identify the dependencies from any existing `requirements.txt`, `environment.yml`, or documented environment configuration.

Typical setup steps for a Python project are:

1. Install a compatible Python version.
2. Clone or download this repository.
3. Create and activate a virtual environment.
4. Install the dependencies specified by the project.
5. Obtain the required datasets from their authorized sources.
6. Review the instructions and configuration for the relevant evaluation track.
7. Run the corresponding preprocessing and experiment scripts in the documented order.

Example virtual-environment commands on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Dependency installation should follow the actual project configuration. Do not assume that a generic dependency list is sufficient to reproduce the experiments.

## 7. Running the Experiments

Run the AI4I and C-MAPSS workflows independently.

### AI4I workflow

1. Locate the AI4I project directory.
2. Identify its preprocessing script and required input data.
3. Check the configuration and expected input paths.
4. Run the preprocessing stage.
5. Run the classification and evaluation stages in the documented order.
6. Verify the generated metrics, calibration results, and plots.
7. Record the software environment and configuration used.

### C-MAPSS workflow

1. Locate the C-MAPSS project directory.
2. Identify the FD001 input data and preprocessing requirements.
3. Review the numbered scripts and their dependencies.
4. Execute the pipeline in its intended order, using the existing project documentation and configuration.
5. Verify the RUL, trend-learning, explainability, uncertainty, and policy-evaluation outputs.
6. Review safety-gate and repeated-seed results.
7. Confirm that final summaries agree with the authoritative experiment outputs.

**Important:** The commands above describe the execution workflow, not a verified, copy-and-run command sequence. Exact script commands and prerequisites must be documented from the actual files before claiming that a clean installation can reproduce all results.

## 8. Evaluation Methodology

Evaluation should use metrics appropriate to each task.

### AI4I classification

Relevant evaluation measures include ROC-AUC, average precision, and probability calibration metrics such as the Brier score.

The evaluation protocol should specify data splits, class balance, preprocessing, calibration, and any threshold-selection procedure.

### C-MAPSS RUL and policy evaluation

Report the RUL prediction metrics actually produced by the implementation, along with the policy metrics used in the experiments.

For adaptation and robustness experiments, document:

- The baseline and candidate policies.
- The experimental conditions and any imposed shifts.
- The repeated random seeds.
- The acceptance and rejection criteria for adaptation.
- The number of accepted and rejected adaptations.
- Paired comparisons, confidence intervals, and significance tests where applicable.
- Any safety constraints and their limitations.

The classification metrics from AI4I and the RUL or policy metrics from C-MAPSS should not be combined into a single performance score unless a justified common evaluation framework is explicitly defined.

## 9. Explainability, Rationality, and Adaptation

In this project, explainability concerns the evidence used to interpret model predictions and maintenance decisions.

Rational decision-making concerns how predictive information, uncertainty, risk, and policy criteria are used to evaluate maintenance actions.

Self-evolution refers to the controlled adaptation mechanisms implemented and evaluated in the project. It should not automatically be interpreted as unrestricted online learning, autonomous deployment, or continuous learning from a live industrial system.

The experimental results must determine which claims are justified. A policy change that passes an implemented safety gate is not necessarily safe under untested conditions.

## 10. Reproducibility and Research Integrity

For reproducible experiments:

- Preserve source code and relevant configuration files.
- Record dataset provenance and permitted access methods.
- Record random seeds and experimental settings.
- Keep raw or authoritative outputs distinguishable from derived summaries.
- Preserve the correspondence between result tables and their source files.
- Document preprocessing and evaluation splits.
- Report unsuccessful experiments and statistically inconclusive results where relevant.
- Avoid presenting simulated sensor shifts as real-world deployment evidence.
- Avoid claiming causal explanations unless the experimental design supports them.

Existing audit outputs support traceability but do not replace independent reproduction of the experiments.

## 11. Limitations

The current experimental evidence has several important limitations:

1. The AI4I and C-MAPSS datasets represent different tasks and require separate interpretation.
2. Performance on benchmark datasets does not establish readiness for deployment in a live industrial environment.
3. Repeated-seed experiments may have limited statistical power.
4. Safety-gated adaptation cannot guarantee that every accepted change will be safe in all operating conditions.
5. Controlled or synthetic sensor shifts do not represent every type of real-world distribution change.
6. Explainability outputs should be interpreted as supporting evidence rather than automatic proof of a model's reasoning or correctness.
7. Reproducibility depends on the exact software environment, data, configuration, and evaluation protocol.

These limitations should be considered when interpreting the results and preparing publications.

## 12. Responsible Data and Repository Management

Before publishing the repository publicly:

- Check dataset licenses and redistribution conditions.
- Do not commit credentials, API keys, personal information, or private configuration.
- Avoid uploading unnecessary caches, temporary files, and duplicate artifacts.
- Check large files against GitHub's current limits.
- Document external dataset sources rather than redistributing data when permission is unclear.
- Keep meaningful graphs, evaluation outputs, and scripts needed to understand the results.

## 13. Current Project Status

The project has an implemented experimental workflow covering two evaluation tracks, with reported predictive, adaptation, robustness, and audit outputs. The project files have also been assembled into a final package.

The next research tasks are to verify the final evidence against the source outputs, document the exact reproducibility procedure, and prepare the research manuscript and presentation.

The current repository should not be considered independently validated solely because its files have been uploaded successfully.

## 14. Research Contributions

The project investigates the combination of:

- Statistical trend learning for industrial decision support.
- Predictive-maintenance classification and RUL estimation.
- Explainability and uncertainty-aware decisions.
- Controlled adaptation of maintenance policies.
- Safety-gated policy evaluation.
- Repeated-seed robustness analysis and experimental evidence audits.

The novelty and strength of these contributions must ultimately be established through comparison with relevant prior research and careful analysis of the experimental evidence.

## 15. Citation and Publication

A formal citation should be added once a stable project version, authorship details, and publication or archival information are established.

When preparing a research paper, cite the original sources of the AI4I and NASA C-MAPSS datasets, relevant baseline methods, and the methods used for model evaluation and calibration.

Do not describe this repository as peer-reviewed or published unless that has actually occurred.

## 16. License

A license has not been specified in this README. Before adding one, confirm the authorship and redistribution rights for the code and all included materials.

## 17. Contact and Contributions

Questions, reproducibility issues, and suggestions may be submitted through the repository's GitHub Issues page.

Contributions should include a clear description of the change and, where appropriate, evidence that the change preserves the intended evaluation protocol.

---

**Project:** A Self-Evolving Explainable Rational Agent for Adaptive Industrial Decision Intelligence Using Statistical Trend Learning

**Evaluation tracks:** AI4I 2020 Predictive Maintenance and NASA C-MAPSS FD001

**Repository purpose:** Research implementation, experimental evidence, and reproducibility documentation.
