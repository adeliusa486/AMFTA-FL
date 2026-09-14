# Sustainable Byzantine-Robust Federated Learning for Smart-City IoT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 67 passed](https://img.shields.io/badge/Tests-67%20passed-brightgreen.svg)](tests/)

Reference implementation for the research paper:
**"Sustainable Byzantine-Robust Federated Learning for Resilient Smart-City IoT Systems"**

Federated intrusion detection allows smart cities to train collaborative defense models directly on local traffic without centralizing sensitive data. Real-world IoT deployments face two challenges: detecting poisoned model updates without trusted server data, and managing battery drain on resource-constrained devices. This repository provides the complete framework to evaluate robustness and energy sustainability together.

---

## Table of Contents

- [Key Results](#key-results)
- [Study Parameters](#study-parameters)
- [Core Findings](#core-findings)
- [Methodology](#methodology)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Linux and macOS](#linux-and-macos)
  - [Windows PowerShell](#windows-powershell)
- [Dataset Preparation](#dataset-preparation)
  - [NF-TON-IoT Dataset](#nf-ton-iot-dataset)
  - [Subsampling for Rapid Testing](#subsampling-for-rapid-testing)
  - [CI Smoke Testing](#ci-smoke-testing)
- [Quick Start](#quick-start)
  - [Federated Training](#federated-training)
  - [Inference CLI](#inference-cli)
- [Programmatic API](#programmatic-api)
- [Reproducing Paper Experiments](#reproducing-paper-experiments)
- [Sustainability and Resource Accounting](#sustainability-and-resource-accounting)
- [Testing and Verification](#testing-and-verification)
- [Scope and Limitations](#scope-and-limitations)
- [Citation](#citation)
- [License and Governance](#license-and-governance)

---

## Key Results

When you enforce an operational accuracy floor of 80%, robustness and energy efficiency do not trade off. The data-free AMFTA-ND aggregator occupies the Pareto frontier at 30% Byzantine client compromise.

<p align="center">
  <img src="figures/fig_pareto.png" width="620" alt="Robustness and Sustainability Pareto Frontier at rho = 0.30">
</p>

---

## Study Parameters

| Parameter | Configuration |
|---|---|
| Dataset | NF-TON-IoT (9,195,116 train / 2,627,177 test flows, 41 NetFlow features) |
| Client Fleet | 100 clients, Dirichlet non-IID partition (alpha = 0.5), full participation |
| Attacker Fractions | rho in {0.10, 0.20, 0.30} |
| Attack Families | Label flipping, Gaussian noise model poisoning, Sign flipping |
| Aggregation Rules | FedAvg, Trimmed Mean, Krum, Multi-Krum, Median, FLTrust, FedDBC, FoolsGold, NormClip-Only, AMFTA, AMFTA-ND, AMFTA-S |
| Model Architecture | Logistic regression (41 features) and LocalMLP |
| Random Seeds | 42, 123, 456 (reported as final 5-round mean with sample standard deviation, ddof = 1) |

---

## Core Findings

| Finding | Concrete Evidence |
|---|---|
| **Certified Stability** | AMFTA-ND is the only rule whose accuracy drop between 10% and 30% attacker compromise stays within a pre-declared 5 percentage point margin in both attack families under two one-sided equivalence tests (TOST). |
| **Gated Pareto Frontier** | AMFTA-ND reaches 90.6% worst-case accuracy versus Krum's 87.3% at 9.8% lower total system energy. It delivers 163.2 robust accuracy points per kilojoule versus 142.0 for Krum. |
| **Linear Aggregation Scaling** | Benchmarked aggregation runtimes show Krum grows 100.4x from 50 to 500 clients, whereas AMFTA-ND grows 5.1x. FedDBC grows 11.9x and Trimmed Mean grows 6.1x. |
| **Data Independence** | Removing the clean server validation buffer (AMFTA-ND) matches or exceeds buffered AMFTA accuracy, gaining up to 11.4 percentage points at rho = 0.30 under label flipping. |
| **Constrained Tier Relief** | AMFTA-S scheduling cuts total client energy by 24.3% and battery sensor energy by 66.7%, leaving mains gateway participation intact while mathematically bounding Byzantine influence amplification to 4.0x. |

---

## Methodology

The framework separates defense trust from sustainability scheduling:

```
[IoT Sensor Fleet: Battery / Intermediate / Mains]
                 │
                 ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Sustainability Layer                                    │
  │ Tracks per-device energy budgets and compute burdens    │
  │ Resource-aware scheduling (AMFTA-S, Proposition 1)      │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Security Layer                                          │
  │ Centroid cosine similarity + EMA reputation tracker     │
  │ Data-free median-norm clipping (AMFTA-ND, Proposition 2)│
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
                 [Robust Global Model Update]
```

1. **Security Layer**: Evaluates update credibility without server data. It measures update direction against the fleet centroid, tracks client reliability across rounds via exponential moving average (EMA) reputation, and applies median-norm rescaling.
2. **Sustainability Layer**: Tracks energy consumption, communication payload size, and local compute operations across heterogeneous hardware tiers (Mains Gateways, Intermediate Nodes, Battery Sensors).
3. **Decoupled Controller**: Adjusts participation rates and local epochs rather than altering trust weights. This design prevents the algorithm from silencing battery nodes that hold critical edge traffic.

---

## Repository Layout

```
Github/
├── .gitignore              # Ignores raw parquet datasets, caches, and logs
├── CODE_OF_CONDUCT.md      # Contributor Covenant standard
├── CONTRIBUTING.md         # Contribution guidelines and workflow
├── LICENSE                 # MIT License
├── Makefile                # Test, clean, and workflow automation
├── pyproject.toml          # PEP 517 build and dependency metadata
├── pytest.ini              # Pytest configuration and test path definitions
├── README.md               # Repository documentation and reproduction guide
├── REPRODUCTION.md         # Detailed paper-to-code mapping
├── requirements.txt        # Production and testing Python dependencies
├── amfta/                  # Core library package
│   ├── aggregation/        # AMFTA, AMFTA-ND, AMFTA-S, and 8 baseline aggregators
│   ├── attacks/            # Label flipping, Gaussian noise, sign flipping, adaptive
│   ├── data/               # NetFlow preprocessing and Dirichlet partitioner
│   ├── models/             # 41-feature logistic regression and LocalMLP
│   └── utils/              # Metrics calculation with sample standard deviation
├── configs/                # Experiment configuration YAML and loader
├── data/                   # Dataset documentation and access instructions
├── docs/                   # System architecture and setup documentation
├── evaluation/             # Visualisation tools and plotting helpers
├── experiments/            # Paper experiment runners and statistical tests
├── figures/                # Architectural diagrams and publication vector plots
├── results/                # 396 empirical JSON/CSV execution logs and paper tables
├── scripts/                # Data setup, inference CLI, and subsampling tools
├── sustainability/         # Resource models, equivalence tests, and figure scripts
├── tables_from_logs/       # Generated LaTeX tables from raw logs
├── tests/                  # 67 unit and integration tests
└── training/               # Federated training runner and client coordinator
```

---

## Installation

### Prerequisites

- Python 3.11 or later
- PyTorch 2.1 or later
- Git

### Linux and macOS

```bash
# Clone repository
git clone https://github.com/adeliusa486/AMFTA-FL.git
cd AMFTA-FL

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
# Clone repository
git clone https://github.com/adeliusa486/AMFTA-FL.git
cd AMFTA-FL

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch GPU Acceleration (Optional):**
> If you have an NVIDIA GPU with CUDA 12.1, install the dedicated PyTorch build:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Dataset Preparation

### NF-TON-IoT Dataset

The paper evaluates models on real network traffic from the NF-TON-IoT NetFlow dataset (9,195,116 training flows and 2,627,177 test flows across 41 features).

1. Download `NF-TON-IoT.parquet` from the University of Queensland repository:
   [https://staff.itee.uq.edu.au/marius/NIDS_datasets/](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
2. Place the parquet file into `data/raw/NF-TON-IoT.parquet`.
3. Process data into partitions and evaluation splits:
   ```bash
   python scripts/setup_data.py --dataset nf-ton-iot
   ```

This generates `data/processed/train.npz`, `data/processed/test.npz`, and the 100-client Dirichlet partitions.

### Subsampling for Rapid Testing

To test full federated runs without processing all 9.2 million records, generate a class-stratified subsample:

```bash
python scripts/make_subsample.py --n 500000 --out data/processed_500k
```

### CI Smoke Testing

For continuous integration testing without downloading large datasets, generate synthetic verification data:

```bash
python scripts/setup_data.py --dataset synthetic
```

*Note: Synthetic data verifies execution flow and shapes. It does not reproduce paper results.*

---

## Quick Start

### Federated Training

Run a single federated learning simulation using AMFTA-ND with 30% Byzantine attackers:

```bash
python experiments/run_main.py \
  --method amfta_noq \
  --byzantine_fraction 0.30 \
  --attack_type label_flipping \
  --num_rounds 25 \
  --num_clients 100 \
  --seed 42
```

Supported methods: `amfta_noq`, `amfta`, `amfta_s`, `krum`, `multi_krum`, `trimmed_mean`, `median`, `fltrust`, `feddbc`, `foolsgold`, `normclip`, `fedavg`.

### Inference CLI

Run inference on pre-extracted feature vectors using trained model weights:

```bash
python scripts/inference.py \
  --weights results/model_weights.pt \
  --model logistic \
  --features 41
```

---

## Programmatic API

You can import and use the AMFTA aggregator directly in custom federated learning pipelines:

```python
import torch
from amfta.aggregation import AMFTAAggregator

# Initialize data-free aggregator (AMFTA-ND)
aggregator = AMFTAAggregator(
    use_val_buffer=False,
    beta=0.8,
    tau_sim=0.1,
    tau_rep=0.2,
)

# Mock client model parameter updates for 10 clients
client_updates = [
    {f"layer_{i}": torch.randn(41, 1) for i in range(2)}
    for _ in range(10)
]

# Aggregate updates
aggregated_update = aggregator.aggregate(client_updates)
diagnostics = aggregator.get_diagnostics()

print(f"Aggregated parameters: {list(aggregated_update.keys())}")
print(f"Fleet trust scores: {diagnostics['trust_scores']}")
```

---

## Reproducing Paper Experiments

Run the study blocks to regenerate the tables and statistical tests reported in the manuscript:

```bash
# Core Attack Evaluations (Tables 6, 7, 8, 9)
python experiments/run_paper_study.py --block labelflip --model logistic
python experiments/run_paper_study.py --block gaussian  --model logistic

# Scalability Timings (Table 18)
python experiments/run_paper_study.py --block scalability --model logistic

# Resource-Aware Scheduling: AMFTA-S (Table 21)
python experiments/run_paper_study.py --block amftas --model logistic

# Post-Process Results into LaTeX Tables and Equivalence Tests
python experiments/paper_stats.py --results results --out tables_from_logs
```

Additional evaluation blocks:
- `--block adaptive`: Tests AGR-tailored attacks.
- `--block normclip`: Evaluates norm rescaling without trust weighting.
- `--block extras`: Evaluates Multi-Krum, FoolsGold, and Coordinate-wise Median.
- `--block alpha01`: Tests severe Non-IID heterogeneity (Dirichlet alpha = 0.1).

---

## Sustainability and Resource Accounting

Derive energy, latency, and communication figures using the audited sustainability models:

```bash
# Compute energy, FLOPs, and communication numbers
python sustainability/resource_model.py

# Run two one-sided equivalence tests (TOST)
python sustainability/equivalence_tests.py

# Regenerate Figures 3, 4, and 5 (Pareto, Energy vs Rho, Tier Budgets)
python sustainability/make_figures.py

# Regenerate Figure 6 (Scaling timings N=50 to N=500)
python sustainability/make_scalability_figure.py

# Verify stored logs against paper tables (42 out of 42 check)
python sustainability/audit_paper_cells.py
```

---

## Testing and Verification

The repository includes a comprehensive unit and integration test suite:

```bash
# Run all 67 tests
pytest

# Run tests with execution timing
pytest -v --durations=10

# Run specific test modules
pytest tests/test_aggregator.py
pytest tests/test_trust_factors.py
pytest tests/test_model_and_data.py
pytest tests/test_integration.py
```

Test coverage includes:
- Aggregator state round-trips and weight constraints
- Borderline and suspect client partitioning logic
- Dirichlet distribution partitioning reproducibility
- Attack determinism under fixed seeds
- End-to-end federated training loops for all aggregators

---

## Scope and Limitations

The empirical claims in this repository adhere to the following documented scope:

1. **Dataset Scope**: Evaluated on NF-TON-IoT NetFlow traffic. Performance across different flow distributions or image benchmarks is not evaluated.
2. **Model Scope**: Evaluated on convex logistic regression models matching the 41-feature tabular NetFlow schema.
3. **Attacker Bound**: Verified up to 30% Byzantine compromise (rho <= 0.30). We do not claim robustness beyond rho = 0.30.
4. **Power Modeling**: Energy figures represent empirically calibrated device power models rather than physical oscilloscope measurements.

---

## Citation

If you use this codebase or baseline implementations in your research, please cite:

```bibtex
@article{ahmad2026sustainable,
  title   = {Sustainable Byzantine-Robust Federated Learning for Resilient Smart-City IoT Systems},
  author  = {Ahmad, Adeel and Akarma, Ali and Mohmand, Muhammad Ismail and Syed, Toqeer Ali and Jan, Salman},
  journal = {Smart Cities},
  year    = {2026}
}
```

---

## License and Governance

- **Code License**: This project is licensed under the [MIT License](LICENSE).
- **Contributing**: Please review [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.
- **Code of Conduct**: Community interactions follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
