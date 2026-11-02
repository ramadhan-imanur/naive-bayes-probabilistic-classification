# Probabilistic Machine Learning: Gaussian Naïve Bayes Classification on Public Clinical Datasets

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Data-Harvard%20Dataverse-orange.svg)](https://doi.org/10.7910/DVN/HIBCFL)
[![Paper](https://img.shields.io/badge/Research%20Paper-LaTeX%20PDF-red.svg)](paper/final_makalah_naive_bayes.pdf)

An end-to-end mathematical analysis, empirical modeling, and critical evaluation of **Gaussian Naïve Bayes** for clinical diagnosis. This project examines the trade-offs of the *conditional independence assumption*, implements biologically sound preprocessing, and evaluates diagnostic performance across multiple holdout and cross-validation protocols.

---

## 📌 Executive Summary

Naïve Bayes is celebrated in machine learning for its foundational simplicity, computational elegance ($O(nd)$ training complexity), and robust classification boundaries. However, in real-world clinical datasets, physiological features exhibit significant mutual correlation, violating the central *conditional independence assumption*.

This project investigates:
1. **Mathematical Foundations**: Derivation of Bayes' Theorem, log-transformed Maximum A Posteriori (MAP) inference, and Gaussian likelihood parameters.
2. **Empirical Evaluation**: Implementation on the **Pima Indians Diabetes Dataset** (sourced from Harvard Dataverse `doi:10.7910/DVN/HIBCFL` & Kaggle), alongside auxiliary benchmarks (Breast Cancer Wisconsin & Framingham Heart Disease).
3. **Clinical Risk & Error Analysis**: Exploring the tension between *Precision* and *Recall* when false negatives carry severe diagnostic consequences.
4. **Research Paper**: Complete academic paper compiled from LaTeX adhering to rigorous scientific standards.

---

## 📐 Mathematical Formulation

### 1. Bayes' Theorem and Posterior Probability
Given an observation feature vector $\mathbf{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$ and target class $c_k \in \mathcal{C}$:

$$P(c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c_k) P(c_k)}{P(\mathbf{x})} = \frac{P(\mathbf{x} \mid c_k) P(c_k)}{\sum_{j=1}^K P(c_j) P(\mathbf{x} \mid c_j)}$$

### 2. The Conditional Independence Assumption
To avoid estimating the intractable joint distribution in high dimensions, attributes are assumed mutually independent given the class label:

$$P(\mathbf{x} \mid c_k) = \prod_{j=1}^d P(x_j \mid c_k)$$

### 3. Log-Likelihood MAP Decision Rule
To eliminate numerical underflow from multiplying small conditional probabilities:

$$\hat{y} = \arg\max_{c_k \in \mathcal{C}} \left[ \ln P(c_k) + \sum_{j=1}^d \ln P(x_j \mid c_k) \right]$$

### 4. Continuous Feature Likelihood (Gaussian NB)
Continuous attributes are modeled via normal probability density functions parameterized by class mean $\mu_{kj}$ and sample variance $\sigma_{kj}^2$:

$$P(x_j \mid c_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left( -\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2} \right)$$

---

## 📊 Dataset Overview

Primary Dataset: **Pima Indians Diabetes Database**  
* **Source**: Harvard Dataverse ([DOI: 10.7910/DVN/HIBCFL](https://doi.org/10.7910/DVN/HIBCFL)) / [Kaggle Benchmark](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
* **Instances**: 768 samples  
* **Features**: 8 clinical predictor variables  
* **Target**: `Outcome` (0 = Non-diabetic [65.1%], 1 = Diabetic [34.9%])

| Feature | Description | Biological Implication |
| :--- | :--- | :--- |
| `Pregnancies` | Number of times pregnant | Gestational metabolic stress |
| `Glucose` | Plasma glucose concentration (2h OGTT) | Key diagnostic biomarker |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Cardiovascular comorbidity |
| `SkinThickness` | Triceps skin fold thickness (mm) | Subcutaneous adiposity indicator |
| `Insulin` | 2-Hour serum insulin ($\mu$U/mL) | Endogenous insulin production |
| `BMI` | Body mass index ($\text{kg}/\text{m}^2$) | Obesity measure |
| `DiabetesPedigreeFunction` | Genetic pedigree score | Hereditary predisposition |
| `Age` | Age in years | Age-related insulin resistance |

> **Biologically-Informed Preprocessing**: In raw clinical records, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` represent missing physiological measurements (since a zero value is physiologically non-viable). These are imputed using class-conscious median values of valid entries to preserve data distribution without synthetic distortion.

---

## 🧪 Experimental Results

### 1. Model Parameters (Learned Class Distributions)
| Feature | Class 0 (Non-Diabetic) $\mu \pm \sigma$ | Class 1 (Diabetic) $\mu \pm \sigma$ |
| :--- | :---: | :---: |
| `Glucose` | $111.85 \pm 24.38$ | $141.32 \pm 29.88$ |
| `BMI` | $30.85 \pm 6.62$ | $35.27 \pm 6.48$ |
| `Age` | $31.42 \pm 11.96$ | $37.40 \pm 11.08$ |
| `Insulin` | $127.08 \pm 66.38$ | $156.43 \pm 87.45$ |

### 2. Test Set Evaluation (Holdout 20%, Stratified)
```text
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)       TN = 84                  FP = 16
Actual Positive (1)       FN = 22                  TP = 32
```

* **Accuracy**: **75.32%**
* **Precision**: **66.67%**
* **Recall (Sensitivity)**: **59.26%**
* **F1-Score**: **62.75%**

### 3. Stability Benchmark (5-Fold Stratified Cross-Validation)
* **Mean Accuracy**: $74.48\% \pm 3.13\%$
* **Mean Precision**: $64.75\% \pm 5.12\%$
* **Mean Recall**: $59.37\% \pm 6.44\%$
* **Mean F1-Score**: $61.80\% \pm 5.11\%$

---

## 🔍 Critical Discussion: Why Does Naïve Bayes Work Despite Assumption Violation?

A common critique in applied statistical learning is that Naïve Bayes relies on an unrealistic independence assumption. However, empirical and theoretical literature reveals:
1. **Separation of Calibration vs. Decision Boundaries**: Accurate posterior estimation requires true independence, but **classification decision boundaries only require preserving rank order** ($\arg\max$). Even when posterior probabilities are pushed towards extremes (overconfident estimates), the classification boundary frequently remains optimal.
2. **Variance Reduction**: By eliminating the need to estimate pairwise covariance terms, Gaussian Naïve Bayes dramatically reduces parameter variance, protecting the model from overfitting on moderate-sized biomedical datasets.
3. **Medical Tradeoff**: A False Negative ($FN = 22$) means a diabetic patient goes undiagnosed. In real-world deployment, threshold adjustment along the ROC curve can shift sensitivity higher to align with clinical safety requirements.

---

## 📂 Project Structure

```text
├── dataset/
│   ├── diabetes_pima.csv            # Primary Pima Indians Diabetes dataset
│   ├── breast_cancer_wisconsin.csv  # Benchmark dataset (Wisconsin Breast Cancer)
│   ├── heart_disease.csv            # Benchmark dataset (Framingham Heart Study)
│   ├── hasil_eksperimen.json        # Serialized empirical results
│   └── README.md                    # Data provenance and DOI documentation
├── paper/
│   └── final_makalah_naive_bayes.pdf# Full academic research paper (PDF)
├── run_experiment.py                # Standalone reproducible execution script
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # Portfolio documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Virtual environment recommended

### Installation & Execution
```bash
# Clone repository
git clone git@github.com:ramadhan-imanur/naive-bayes-probabilistic-classification.git
cd naive-bayes-probabilistic-classification

# Install dependencies
pip install -r requirements.txt

# Run the experimental pipeline
python3 run_experiment.py
```

---

## 📚 References & Data Citations
1. **OpenXAI Benchmark**: *OpenXAI: Towards a Comprehensive Evaluation Benchmark for Explainable AI*. Harvard Dataverse, DOI: [10.7910/DVN/HIBCFL](https://doi.org/10.7910/DVN/HIBCFL).
2. **National Institute of Diabetes and Digestive and Kidney Diseases**: *Pima Indians Diabetes Database*. UCI Machine Learning Repository / Kaggle.
3. **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. Springer, New York.
4. **Mitchell, T. M.** (1997). *Machine Learning*. McGraw-Hill, New York.

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
