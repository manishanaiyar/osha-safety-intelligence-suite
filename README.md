# 🛡️ OSHA Safety Intelligence Suite

> A multimodal AI system for workplace safety — combining XGBoost, BERT, Isolation Forest, Monte Carlo & Reinforcement Learning to predict, classify and prevent industrial incidents.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📋 Overview

This project applies six specialized AI models to 21,578 real OSHA severe injury reports (2015–2017) to build a complete workplace safety intelligence pipeline — from raw incident data to prescriptive intervention recommendations.

Each model addresses a different dimension of the safety problem:

| Model | Task | Key Result |
|-------|------|------------|
| XGBoost Risk Classifier | Predict hospitalization from structured features | AUC = 0.95, Recall = 97.7% |
| Isolation Forest | Detect statistically rare "Black Swan" incidents | 216 anomalies flagged from 21,578 |
| Monte Carlo Simulation | Forecast annual high-risk incident volume | 2.13% error vs real historical data |
| BERT NLP Classifier | Classify risk from raw narrative text | 91% accuracy, 93% HR recall |
| Q-Learning RL Agent | Prescribe optimal safety interventions | 64 states, 20,000 episodes |
| MCDA Integration | Combine all models into unified Priority Index | Critical/Elevated/Moderate/Low tiers |

---

## 📁 Repository Structure

```
osha-safety-intelligence-suite/
│
├── BERT_NLP_Classifier.ipynb          # BERT fine-tuning on incident narratives
├── IsolationForest_Anomaly_Detection.ipynb  # Unsupervised anomaly detection
├── Monte_Carlo.ipynb                  # Stochastic risk forecasting
├── RL_QlearningAgent.ipynb            # Q-Learning prescriptive engine
├── GPT_MCDA_Integration.ipynb         # Multi-criteria decision analysis
├── random_forest_&_decision_tree.ipynb # Baseline classifiers
│
├── severeinjury.csv                   # Raw OSHA dataset (21,578 records)
├── X_tabular.pkl                      # Preprocessed structured features
├── X_text.pkl                         # Preprocessed narrative text
├── X_time.pkl                         # Preprocessed time series data
├── q_table_v2.pkl                     # Trained Q-Table (64 states × 4 actions)
│
└── README.md
```

---

## 🗂️ Dataset

**Source:** OSHA Severe Injury Reports 2015–2017  
**Size:** 21,578 incidents, 26 columns  
**Target:** Binary — Hospitalized (80.08%) vs Minor/Outpatient (19.92%)

**Key columns used:**
- `NarrativeText` — free-text incident description written by OSHA inspectors
- `Hospitalized` — target variable (administrative admission status)
- `EventTitle` — type of incident event
- `Part of Body Title` — body part affected
- `Latitude / Longitude` — GPS coordinates of facility
- `State`, `Primary NAICS` — geographic and industry identifiers

> **Important data quality note:** The `Hospitalized` field measures hospital admission status, not clinical severity. Amputations treated at outpatient facilities are labeled `0` (Minor). This distinction is documented in the BERT notebook error analysis.

---

## 🤖 Model Details

### 1. XGBoost Risk Classifier
**File:** `random_forest_&_decision_tree.ipynb`

Binary classification predicting hospitalization from 473 engineered features.

**Critical bugs identified and fixed:**
- Data leakage from `Amputation` column (100% correlation with target)
- Hidden leakage from `NatureTitle` (outcome variable disguised as input)
- Encoding before train/test split (test categories visible during training)
- GPS coordinates one-hot encoded (destroyed spatial meaning)
- `scale_pos_weight` inverted (0.25 instead of 4.02)

**Results:**
```
Accuracy:    90.64%       AUC:      0.9530
F1 Score:    0.9398       Recall:   0.9123
Cross-Val AUC: 0.9507 ± 0.0039 (5-fold stratified)
```

---

### 2. Isolation Forest Anomaly Detection
**File:** `IsolationForest_Anomaly_Detection.ipynb`

Unsupervised detection of statistically rare incident combinations using path-length isolation scoring across 100 trees.

```
Total incidents scanned: 21,578
Anomalies flagged:       216 (1.00%)
Notable detections:      Chemical burns (sulfuric acid), propane ignition,
                         oil drilling fractures, extension ladder brain injuries
```

**Honest finding:** Feature set is predominantly categorical (one-hot binary). Isolation Forest produces weak geometric separation on binary features. Scores cluster between -0.0214 and 0.0000. For stronger anomaly detection, continuous operational metrics (equipment age, days since inspection, prior near-miss count) would significantly improve signal quality.

---

### 3. Monte Carlo Simulation
**File:** `Monte_Carlo.ipynb`

Stochastic forecasting using a Poisson-Binomial compound process to model annual high-risk incident distribution across 10,000 simulated futures.

**Mathematical foundation:**
```
For each of 10,000 simulated years:
  For each of 12 months:
    monthly_incidents = Poisson(λ=832.49)
    monthly_high_risk = Binomial(incidents, P=0.8008)
  Annual total = sum of 12 months
```

**Results:**
```
Expected annual high-risk incidents:  7,999
95th percentile (safety threshold):   8,146
Backtesting error vs real data:       2.13%  ✅

Stress testing:
  -10% risk reduction → saves ~800 incidents/year
  +10% risk increase  → adds ~800 incidents/year
```

---

### 4. BERT NLP Classifier
**File:** `BERT_NLP_Classifier.ipynb`

Fine-tuned `bert-base-uncased` on OSHA incident narratives for binary risk classification.

**Training configuration:**
```
Base model:    bert-base-uncased (110M parameters)
Epochs:        5
Optimizer:     AdamW  lr=2e-5
Loss:          Weighted Cross Entropy (Minor weight ≈ 4.02×)
Scheduler:     Linear warmup (10%) → linear decay
Max length:    128 tokens
Hardware:      T4 GPU
```

**Results:**
```
Accuracy:          91%
High Risk Recall:  93%     High Risk F1:  0.94
Minor Recall:      85%     Minor F1:      0.80
False Negatives:   249 / 3,456  (7.20%)
False Positives:   126 /   860  (14.65%)
```

**Key finding from error analysis:**
BERT correctly learned the dataset's ground truth — including the counterintuitive pattern that amputation narratives often have LOW High Risk probability. Investigation revealed this reflects the OSHA labeling system: amputations treated at outpatient facilities receive `Hospitalized=0` (Minor). BERT accurately learned administrative admission patterns rather than clinical severity — a finding that reveals a fundamental limitation in using `Hospitalized` as a severity proxy.

---

### 5. Q-Learning Reinforcement Learning Agent
**File:** `RL_QlearningAgent.ipynb`

Prescriptive engine that learns optimal safety interventions through 20,000 episodes of trial and error on OSHA data.

**Environment design:**
```
States (64):   EventTitle × Part of Body Title
               (8 top event types × 8 top body parts)
Actions (4):   Safety_Training | PPE_Renewal |
               Equipment_Service | Protocol_Audit
Rewards:       Derived from real hospitalization rates
               base_reward = 5 + (hosp_rate × 10)
               Range: 5.0 to 15.0
```

**Hyperparameters:**
```
α = 0.10   γ = 0.95   ε = 0.20 → 0.01 (decay=0.9995)
```

**Learned policy:**
```
Safety_Training:    75.0% of states
Equipment_Service:  10.9% of states
PPE_Renewal:         7.8% of states
Protocol_Audit:      6.2% of states
```

**Honest assessment:** The agent experienced suboptimal convergence — Safety_Training accumulated dominant Q-values early in training due to consistent moderate rewards, crowding out other actions. Q-value convergence was confirmed (mean Q-value stabilizes after episode ~15,000). For production deployment, softmax action selection or UCB exploration would address this limitation.

---

### 6. MCDA Integration Layer
**File:** `GPT_MCDA_Integration.ipynb`

Combines all three AI signals into a unified Priority Index using Multi-Criteria Decision Analysis.

**Formula:**
```
Priority Index = (BERT × 0.40) + (MC_normalized × 0.40) + (RL_confidence × 0.20)

Tiers:
  > 0.80  →  🔴 CRITICAL   (immediate action, 24 hours)
  > 0.60  →  🟠 ELEVATED   (action within 48 hours)
  > 0.40  →  🟡 MODERATE   (action within 1 week)
  ≤ 0.40  →  🟢 LOW        (regular review cycle)
```

**Live results on 4 OSHA incidents:**
```
Incident                                    BERT     Priority   Status
──────────────────────────────────────────────────────────────────────
Forklift tipped, pinned, hospitalized      89.37%   0.9210     🔴 CRITICAL
Scaffolding fall, skull fractures          88.72%   0.9077     🔴 CRITICAL
Punch press amputation, treated+released   22.45%   0.6825     🟠 ELEVATED
Auger amputation, outpatient               9.81%    0.6319     🟠 ELEVATED
```

> The outpatient amputation cases score ELEVATED (not LOW) despite low BERT severity scores — because the Monte Carlo component correctly captures that 7,999 similar incidents occur annually, making them a systemic organizational risk regardless of individual case severity.

---

## 🔁 How the Models Work Together

```
Raw OSHA Incident
       │
       ├──► XGBoost ────────────► "Will this be hospitalized?"
       │                           (structured features)
       │
       ├──► Isolation Forest ───► "Is this a rare Black Swan?"
       │                           (anomaly detection)
       │
       ├──► BERT ──────────────► "How severe is the narrative?"
       │                           (NLP text analysis)
       │                                    │
       ├──► Monte Carlo ────────► "How many per year?"          ├──► MCDA ──► Priority Index
       │                           (stochastic forecasting)      │             + Recommended
       │                                                         │             Action
       └──► Q-Learning ─────────► "What should we DO?"    ──────┘
                                   (prescriptive RL)
```

---

## 🚀 Getting Started

**Requirements:**
```
Python 3.10+
torch >= 2.0
transformers >= 4.30
xgboost >= 1.7
scikit-learn >= 1.2
pandas, numpy, matplotlib, seaborn
shap
```

**Installation:**
```bash
pip install torch transformers xgboost scikit-learn shap pandas numpy matplotlib seaborn
```

**Data setup:**
```
1. Download severeinjury.csv from this repository
2. Run notebooks in this order:
   1. random_forest_&_decision_tree.ipynb  (XGBoost)
   2. IsolationForest_Anomaly_Detection.ipynb
   3. Monte_Carlo.ipynb
   4. BERT_NLP_Classifier.ipynb  (requires GPU, ~45 min on T4)
   5. RL_QlearningAgent.ipynb
   6. GPT_MCDA_Integration.ipynb
```

> All notebooks were developed and tested on Google Colab with T4 GPU. Mount your Google Drive and update the `path` variable in each notebook to your working directory.

---

## 📊 Results Summary

```
Model                  Key Metric              Value
─────────────────────────────────────────────────────
XGBoost                AUC (5-fold CV)         0.9507 ± 0.0039
XGBoost                Hospitalization Recall  97.7%
BERT                   Validation Accuracy     91%
BERT                   High Risk Recall        93%
Monte Carlo            Backtesting Error       2.13%
Monte Carlo            Annual Forecast         7,999 ± 90 incidents
RL Agent               Q-Value Convergence     Episode ~15,000
MCDA                   Critical Detection      2/2 hospitalized cases ✅
MCDA                   Elevated Detection      2/2 outpatient cases   ✅
```

---

## 🔍 Key Findings

**1. Data leakage is pervasive in safety datasets.**
Nine distinct leakage sources were identified in the raw OSHA data — from obvious outcome variables to subtle encoding-before-split errors. All were documented and corrected.

**2. Ground truth labels encode administrative outcomes, not clinical severity.**
The `Hospitalized` field measures insurance/admission paperwork, not injury severity. Amputations treated outpatient are labeled Minor. This was discovered through BERT error analysis and has significant implications for any safety ML system using this dataset.

**3. Monte Carlo outperforms point estimates for organizational planning.**
A single prediction ("8,000 incidents") is far less useful than a distribution with confidence intervals. The 95th percentile threshold (8,146) gives safety managers an actionable trigger for declaring an abnormal safety period.

**4. Basic Q-Learning converges but may find suboptimal policies.**
The RL agent correctly converged (confirmed by Q-value stabilization) but developed Safety_Training dominance. This is a known limitation of deterministic reward Q-Learning — documented honestly with proposed solutions.

---

## 👤 Author

**Manisha Naiyar**  
[GitHub](https://github.com/manishanaiyar) · [LinkedIn](https://linkedin.com/in/manishanaiyar)


---

*Built on 21,578 real OSHA workplace injury reports. Every number in this repository is traceable to real data.*
