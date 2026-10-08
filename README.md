# COVID-19 Clinical Severity & Risk Estimation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](LICENSE)

An end-to-end Machine Learning system developed to predict clinical severity and mortality risks for confirmed COVID-19 patients based on baseline demographic information and pre-existing comorbidities. 

The system estimates the risk for three critical clinical endpoints:
1. **ICU Admission Risk** (Intensive Care Unit)
2. **Intubation Risk** (Mechanical Ventilation requirement)
3. **Mortality / Fatality Risk** (Patient Death)

Includes an interactive **Streamlit** clinical decision support application ([`demo.py`](demo.py)).

---

## 👥 Authors & Academic Details

* **Neel Chandrakar** (PES2UG24CS310)
* **Neeraj R Rugi** (PES2UG24CS311)

*Department of Computer Science and Engineering, PES University*  
*Machine Learning Mini Project (5th Semester)*

---

## 📖 Research Background & Clinical Rationale

During respiratory pandemic surges, hospitals face critical shortages of intensive care unit (ICU) beds and mechanical ventilators. Early triaging and risk stratification enable clinical teams to prioritize aggressive interventions and allocate limited life-support infrastructure effectively.

This project is inspired by and grounded in the research methodology published by Stanford University:
> **Zhan, C. & Li, Y.** (Stanford University, CS229 Spring 2020)  
> *Machine Learning-Based Risk Prediction for COVID-19 Patients*  
> [Read Stanford CS229 Project Report](https://cs229.stanford.edu/proj2020spr/report/Zhan_Li.pdf)

---

## 📊 Dataset In-Depth Breakdown

* **Data Source:** [COVID-19 Mexico Patient Health Dataset (Kaggle)](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset) (Published by the Mexican Ministry of Health / Epidemiological Surveillance System).
* **Sample Size After Preprocessing:** **$n = 23,465$** laboratory-confirmed COVID-19 patients.
* **Train / Test Split:** Standard **80% - 20%** train-test split:
  * **Training Set:** $n = 18,772$ patients
  * **Testing Set:** $n = 4,693$ patients

### 1. Predictor Features (11 Input Variables)

All non-continuous features are encoded as binary flags ($0 = \text{No}, 1 = \text{Yes}$), while continuous variables are standardized.

| Feature Name | Type | Distribution / Summary | Clinical Description |
| :--- | :---: | :--- | :--- |
| `Age` | Numeric | Range: 0 – 113 yrs (Mean: 46.52 yrs, Std: ~15.3) | Patient's age in years, normalized with `StandardScaler`. |
| `Gender` | Binary | Female: 13,656 (58.2%), Male: 9,809 (41.8%) | Biological sex (0 = Female, 1 = Male). |
| `Has_Pneumonia` | Binary | No: 16,585 (70.7%), Yes: 6,880 (29.3%) | Confirmed pneumonia diagnosis upon hospital admission. |
| `Has_Diabetes` | Binary | No: 19,119 (81.5%), Yes: 4,346 (18.5%) | Pre-existing diagnosed diabetes mellitus. |
| `Has_COPD` | Binary | No: 22,888 (97.5%), Yes: 577 (2.5%) | Chronic Obstructive Pulmonary Disease. |
| `Has_Asthma` | Binary | No: 22,685 (96.7%), Yes: 780 (3.3%) | History of chronic bronchial asthma. |
| `Is_Immunosuppressed` | Binary | No: 23,037 (98.2%), Yes: 428 (1.8%) | Immunosuppressed due to therapy or congenital condition. |
| `Has_Hypertension` | Binary | No: 18,372 (78.3%), Yes: 5,093 (21.7%) | Pre-existing chronic arterial hypertension. |
| `Has_Cardiovascular` | Binary | No: 22,808 (97.2%), Yes: 657 (2.8%) | Pre-existing cardiovascular diseases. |
| `Is_Smoker` | Binary | No: 21,384 (91.1%), Yes: 2,081 (8.9%) | Active or chronic tobacco smoking history. |
| `Is_Obese` | Binary | No: 18,526 (79.0%), Yes: 4,939 (21.0%) | Diagnosed clinical obesity (BMI $\ge 30$). |

### 2. Clinical Target Variables & Class Imbalance

The three binary target outcomes exhibit substantial **class imbalance**, which represents realistic real-world epidemiological settings:

| Target Outcome | Dataset Column | Total Positive | Dataset Prevalence | Test Set Count ($n = 4,693$) |
| :--- | :---: | :---: | :---: | :---: |
| **ICU Admission** | `Admitted_ICU` | 987 | **4.21%** | 198 Positive / 4,495 Negative |
| **Intubation** | `Is_Intubated` | 991 | **4.22%** | 195 Positive / 4,498 Negative |
| **Mortality (Death)** | `Is_Deceased` | 2,154 | **9.18%** | 427 Positive / 4,266 Negative |

> **Addressing Class Imbalance:**
> Because severe outcomes are rare (only 4.2% require ICU/intubation), predicting positive cases is an extreme needle-in-a-haystack problem. In clinical triage, **Recall (Sensitivity) is paramount**: a False Negative (failing to flag a patient who subsequently deteriorates) is catastrophic, whereas a False Positive (monitoring an uncritical patient) is benign. To prevent models from defaulting to the majority negative class, we employed **`class_weight='balanced'`**, penalizing errors on the minority class inversely proportional to class frequencies.

---

## 🔬 Models Implemented

Four distinct classification algorithms were developed and evaluated in [`model_training.ipynb`](model_training.ipynb):

1. **Logistic Regression (with Balanced Class Weights & Pipeline Scaling)**
   * Feature pipeline: `ColumnTransformer` standardizing `Age` and passthrough for binary flags.
   * Calibrated probabilities, high interpretability, and stable decision boundaries. Deployed to the Streamlit app.
2. **Support Vector Machines (SVM)**
   * Kernel-based boundary separation with balanced class penalty.
3. **Decision Tree Classifier**
   * Non-parametric hierarchical rule-based classification.
4. **Random Forest Classifier**
   * Ensemble method utilizing 100 bagged decision trees to minimize model variance and capture non-linear feature interactions.

---

## 📈 Model Performance & Evaluation Statistics

All models were evaluated on the held-out test set of **4,693 patients** using **Precision**, **Recall**, **F1-score**, **Accuracy**, **ROC-AUC**, and **Precision-Recall AUC (PR-AUC)**.

### 1. ICU Admission Prediction Performance

Random baseline positive rate: **4.2%** ($198 / 4693$).

| Model | Class 1 Recall | Class 1 Precision | Class 1 F1-Score | Overall Accuracy | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.84** (84%) | 0.12 | **0.21** | 0.74 | **0.829** | 0.135 |
| **SVM** | **0.82** (82%) | 0.12 | **0.21** | 0.74 | 0.802 | 0.135 |
| **Decision Tree** | **0.83** (83%) | 0.12 | **0.21** | 0.74 | **0.830** | 0.134 |
| **Random Forest** | **0.83** (83%) | 0.12 | **0.21** | 0.74 | **0.830** | 0.134 |

#### Detailed Classification Breakdown: Logistic Regression (ICU)
```
              precision    recall  f1-score   support

  No ICU (0)       0.99      0.74      0.84      4495
     ICU (1)       0.12      0.84      0.21       198

    accuracy                           0.74      4693
   macro avg       0.56      0.79      0.53      4693
weighted avg       0.95      0.74      0.82      4693

ROC AUC: 0.829
PR-AUC (Average Precision): 0.135 (3.2x lift over random baseline 0.042)
```

---

### 2. Intubation (Mechanical Ventilation) Prediction Performance

Random baseline positive rate: **4.2%** ($195 / 4693$).

| Model | Class 1 Recall | Class 1 Precision | Class 1 F1-Score | Overall Accuracy | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.94** (94%) | 0.14 | **0.24** | 0.75 | **0.877** | 0.141 |
| **SVM** | **0.92** (92%) | 0.14 | **0.24** | 0.76 | 0.846 | 0.141 |
| **Decision Tree** | **0.90** (90%) | 0.13 | **0.23** | 0.75 | 0.852 | 0.150 |
| **Random Forest** | **0.90** (90%) | 0.13 | **0.23** | 0.75 | 0.852 | 0.150 |

#### Detailed Classification Breakdown: Logistic Regression (Intubation)
```
                 precision    recall  f1-score   support

No Intubation (0)     1.00      0.74      0.85      4498
   Intubation (1)     0.14      0.94      0.24       195

         accuracy                           0.75      4693
        macro avg     0.57      0.84      0.54      4693
     weighted avg     0.96      0.75      0.82      4693

ROC AUC: 0.877
PR-AUC (Average Precision): 0.141 (3.4x lift over random baseline 0.042)
```

---

### 3. Mortality (Patient Death) Prediction Performance

Random baseline positive rate: **9.1%** ($427 / 4693$).

| Model | Class 1 Recall | Class 1 Precision | Class 1 F1-Score | Overall Accuracy | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.80** (80%) | **0.25** | **0.38** | **0.76** | **0.848** | 0.250 |
| **SVM** | **0.83** (83%) | 0.23 | **0.37** | 0.74 | 0.811 | 0.250 |
| **Decision Tree** | **0.81** (81%) | 0.23 | **0.35** | 0.73 | 0.839 | **0.298** |
| **Random Forest** | **0.81** (81%) | 0.23 | **0.35** | 0.73 | 0.839 | **0.298** |

#### Detailed Classification Breakdown: Logistic Regression (Mortality)
```
              precision    recall  f1-score   support

 Deceased (0)      0.97      0.75      0.85      4266
 Deceased (1)      0.25      0.80      0.38       427

    accuracy                           0.76      4693
   macro avg       0.61      0.78      0.61      4693
weighted avg       0.91      0.76      0.81      4693

ROC AUC: 0.848
PR-AUC (Average Precision): 0.250 (2.75x lift over random baseline 0.091)
```

---

## 🔍 Feature Importance & Model Weights (Logistic Regression)

Inspecting the learned regression coefficients reveals the impact of individual features on risk escalation:

$$\text{Log-Odds} = \beta_0 + \sum_{i=1}^{11} \beta_i X_i$$

| Feature | ICU Coefficient ($\beta$) | Intubation Coefficient ($\beta$) | Mortality Coefficient ($\beta$) | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Has_Pneumonia** | **+2.925** | **+3.781** | **+1.816** | **Dominant Risk Factor**: By far the strongest predictor for all 3 outcomes; drastically increases odds of intubation ($\approx 43.8\times$ odds). |
| **Age (Standardized)** | **+0.280** | **+0.329** | **+0.731** | Strong positive correlation with mortality; older age heavily accelerates fatality risk. |
| **Is_Immunosuppressed** | +0.285 | +0.125 | **+0.695** | Significantly increases mortality risk. |
| **Has_Diabetes** | +0.119 | +0.096 | **+0.547** | Established metabolic risk factor amplifying death rate. |
| **Is_Obese** | +0.294 | +0.411 | **+0.508** | Consistent risk amplifier across ventilation, ICU, and mortality. |
| **Has_Hypertension** | +0.135 | +0.095 | **+0.346** | Cardiovascular stress contributor. |
| **Has_Asthma** | +0.473 | -0.362 | +0.037 | Correlates with ICU triage, but lower mechanical ventilation rate in sample. |
| **Gender (Male)** | -0.442 | -0.366 | -0.561 | Female baseline representation in cohort. |
| **Model Intercept (Bias)** | **-1.866** | **-2.737** | **-1.374** | Base outcome prevalence log-odds. |

---

## 💡 Understanding the Clinical Metric Trade-offs

1. **Why is Class 1 Recall high (80% - 94%) while Precision is lower (12% - 25%)?**
   * Because only 4.2% of admitted patients require intubation or ICU, a naive model that predicts "No" for everyone would score **95.8% accuracy**, but fail **100% of critical patients**.
   * By setting `class_weight='balanced'`, our models prioritize **Sensitivity (Recall)**. We successfully capture **94 out of every 100 patients requiring mechanical ventilation**.
2. **Why ROC-AUC (0.83 - 0.88) is the primary benchmark:**
   * ROC-AUC demonstrates strong discriminative power across all thresholds, proving that the model consistently ranks high-risk patients above stable ones.
3. **Average Precision (PR-AUC) Lift:**
   * Across all models, PR-AUC is **$3\times$ higher than the random baseline**, providing strong statistical lift in imbalanced clinical classification.

---

## 🖥️ Streamlit Web Application

The interactive web dashboard ([`demo.py`](demo.py)) provides clinicians with real-time risk assessment:

1. **Inputs:** Sliders and checkboxes for Age, Gender, and 9 comorbidities.
2. **Model Evaluation:** Inputs are scaled through the stored pipeline and evaluated against the trained logistic models.
3. **Outputs:** Real-time probability percentage gauges:
   * **ICU Admission Risk score**
   * **Intubation Risk score**
   * **Fatality Risk score**
4. **Alert Badges:** Dynamic thresholds flag higher-risk patients in red (`Flagged: higher risk`) versus stable patients in green (`Not flagged`).

---

## 📁 Repository Structure

```text
ML-Mini-Project/
├── demo.py                          # Streamlit web application
├── model_training.ipynb             # Model training, balancing & evaluation metrics
├── exploratory_data_analysis.ipynb  # EDA & statistical data exploration
├── pair_plot_analysis.ipynb         # Multivariate pair-plot distributions
├── figures/                         # Generated EDA and evaluation visualizations
│   ├── eda_age_outcomes_density.png
│   ├── eda_age_stratification.png
│   ├── eda_correlation_matrix.png
│   ├── eda_odds_ratios_forest.png
│   ├── eda_targets_distribution.png
│   └── pairplot_mortality.png
├── Models/                          # Serialized pipeline artifacts (.joblib)
│   ├── LR_ICU_model.joblib
│   ├── LR_INT_model.joblib
│   └── LR_DEATH_model.joblib
├── InputData.csv                    # Cleaned & processed tabular dataset (n=23,465)
├── patient.csv                      # Raw COVID-19 patient registry dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🚀 How to Run the Project

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/neeraj-r-rugi/ML-Mini-Project.git
cd ML-Mini-Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Model Artifacts
Open and run all cells in [`model_training.ipynb`](model_training.ipynb) to train the models and export the serialized artifacts to `./Models/`:
```bash
jupyter lab model_training.ipynb
```

### 3. Launch the Streamlit App
```bash
streamlit run demo.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 📚 References

1. Zhan, C., & Li, Y. (2020). *Machine Learning-Based Risk Prediction for COVID-19 Patients*. Stanford University CS229 Report. [Link](https://cs229.stanford.edu/proj2020spr/report/Zhan_Li.pdf).
2. General Directorate of Epidemiology, Mexican Ministry of Health. *COVID-19 Mexico Patient Health Dataset*. [Kaggle Dataset](https://www.kaggle.com/datasets/riteshahlawat/covid19-mexico-patient-health-dataset).

---

## 📜 Academic Integrity & License

Developed for the 5th Semester Machine Learning Mini Project at **PES University**, Bengaluru.
All rights reserved for academic evaluation.
