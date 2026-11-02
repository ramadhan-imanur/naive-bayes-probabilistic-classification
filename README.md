# Gaussian Naive Bayes on clinical datasets

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Data-Harvard%20Dataverse-orange.svg)](https://doi.org/10.7910/DVN/HIBCFL)

Mathematical derivation, empirical evaluation, and diagnostic risk analysis for Gaussian Naive Bayes applied to the Pima Indians Diabetes dataset, with auxiliary checks on breast cancer and heart disease data.

---

## Summary

Naive Bayes assumes that all predictors are conditionally independent given the class label. In clinical diagnostic data, this assumption is regularly violated because physiological indicators correlate with each other. Despite this theoretical conflict, the classifier remains widely used for medical risk screening.

This repository implements Gaussian Naive Bayes from scratch and via scikit-learn on the Pima Indians Diabetes dataset (768 records, 8 clinical predictors). It provides:

* Formal derivation of log-likelihood Maximum A Posteriori (MAP) decision rules for continuous features.
* Biologically informed imputation for unrecorded clinical measurements (zero-valued glucose, blood pressure, skin thickness, insulin, and BMI).
* Exact confusion matrix and performance metrics across an 80/20 stratified split and 5-fold cross-validation.
* Error analysis explaining why classification accuracy holds up even when posterior probabilities are uncalibrated due to feature correlation.

---

## Mathematical formulation

### Posterior probability and Bayes theorem
For an observation vector $\mathbf{x} = (x_1, x_2, \dots, x_d) \in \mathbb{R}^d$ and discrete class label $c_k \in \mathcal{C}$:

$$P(c_k \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c_k) P(c_k)}{P(\mathbf{x})} = \frac{P(\mathbf{x} \mid c_k) P(c_k)}{\sum_{j=1}^K P(c_j) P(\mathbf{x} \mid c_j)}$$

### Conditional independence assumption
Estimating a full joint likelihood $P(\mathbf{x} \mid c_k)$ in $d$ dimensions requires either dense data or high-dimensional covariance estimation. Naive Bayes simplifies this by assuming features are conditionally independent given $c_k$:

$$P(\mathbf{x} \mid c_k) = \prod_{j=1}^d P(x_j \mid c_k)$$

### Log-likelihood MAP decision rule
Multiplying many conditional probabilities smaller than 1 leads to floating-point underflow. Taking the natural logarithm converts the product into a sum:

$$\hat{y} = \arg\max_{c_k \in \mathcal{C}} \left[ \ln P(c_k) + \sum_{j=1}^d \ln P(x_j \mid c_k) \right]$$

Because the logarithm is strictly monotonic, this transformation preserves the class ranking.

### Continuous feature likelihood (Gaussian)
For continuous predictors, each class-conditional density is modeled as a univariate normal distribution with class mean $\mu_{kj}$ and variance $\sigma_{kj}^2$:

$$P(x_j \mid c_k) = \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left( -\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2} \right)$$

This specification requires estimating only $2d$ parameters per class (the mean and variance of each feature), avoiding the $d(d+1)/2$ parameters needed for a full covariance matrix.

---

## Dataset and preprocessing

The primary benchmark uses the Pima Indians Diabetes Database, sourced from the Harvard Dataverse ([DOI: 10.7910/DVN/HIBCFL](https://doi.org/10.7910/DVN/HIBCFL)) and Kaggle.

* **Sample count**: 768 records
* **Predictor count**: 8 physiological variables
* **Target variable**: `Outcome` (0 = negative [500 samples, 65.1%], 1 = positive [268 samples, 34.9%])

| Feature | Description | Biological context |
| :--- | :--- | :--- |
| `Pregnancies` | Number of pregnancies | Gestational metabolic load |
| `Glucose` | Plasma glucose concentration (2h oral glucose tolerance test) | Primary diagnostic marker |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | Cardiovascular indicator |
| `SkinThickness` | Triceps skin fold thickness (mm) | Peripheral adiposity |
| `Insulin` | 2-hour serum insulin ($\mu$U/mL) | Endogenous insulin response |
| `BMI` | Body mass index ($\text{kg}/\text{m}^2$) | Weight-to-height ratio |
| `DiabetesPedigreeFunction` | Genetic pedigree score | Family history factor |
| `Age` | Age in years | Age-linked metabolic drift |

In raw records, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are physiologically non-viable. They indicate missing readings rather than true zero measurements. The pipeline imputes these zeros with the median value of non-zero entries for that variable.

---

## Experimental results

### Learned class distributions
Estimating parameters on the training set yields distinct class means for key metabolic markers:

| Feature | Class 0 (non-diabetic) $\mu \pm \sigma$ | Class 1 (diabetic) $\mu \pm \sigma$ |
| :--- | :---: | :---: |
| `Glucose` | $111.85 \pm 24.38$ | $141.32 \pm 29.88$ |
| `BMI` | $30.85 \pm 6.62$ | $35.27 \pm 6.48$ |
| `Age` | $31.42 \pm 11.96$ | $37.40 \pm 11.08$ |
| `Insulin` | $127.08 \pm 66.38$ | $156.43 \pm 87.45$ |

### Test set evaluation (80/20 stratified split)

The held-out test set contains 154 observations (100 negative, 54 positive):

```text
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)       TN = 84                  FP = 16
Actual Positive (1)       FN = 22                  TP = 32
```

* Accuracy: 75.32%
* Precision: 66.67%
* Recall (Sensitivity): 59.26%
* F1 score: 62.75%

### Cross-validation stability (5-fold stratified)
Running 5-fold stratified cross-validation on the entire dataset gives the following performance intervals:

* Accuracy: $74.48\% \pm 3.13\%$
* Precision: $64.75\% \pm 5.12\%$
* Recall: $59.37\% \pm 6.44\%$
* F1 score: $61.80\% \pm 5.11\%$

---

## Analysis and clinical risk

### Calibration versus decision boundaries
Feature correlations (such as `BMI` with `SkinThickness`, or `Glucose` with `Insulin`) cause Naive Bayes to overcount evidence, pushing posterior probabilities toward 0 and 1. While this degrades probability calibration, it does not necessarily hurt classification accuracy. Because the decision rule relies solely on $\arg\max$, the classification remains correct as long as the true class maintains the highest score.

### Variance reduction in small samples
Fitting an unconstrained covariance matrix for 8 features requires estimating 36 distinct covariance terms per class. With only 268 positive cases, empirical covariance estimates become noisy and prone to overfitting. Restricting the covariance matrix to its diagonal reduces the estimation to 8 variances per class, which stabilizes test set predictions.

### Diagnostic trade-offs in screening
In this test set, 22 diabetic individuals were misclassified as healthy (false negatives), while 16 healthy individuals were flagged for diabetes (false positives). In a clinical screening context, false negatives carry greater risk than false positives because untreated diabetes leads to secondary complications. The default decision threshold of 0.5 can be lowered along the ROC curve to increase sensitivity (recall) when minimizing missed cases is the priority.

---

## Project structure

```text
├── dataset/
│   ├── diabetes_pima.csv            # Primary Pima Indians Diabetes dataset
│   ├── breast_cancer_wisconsin.csv  # Benchmark dataset (Wisconsin Breast Cancer)
│   ├── heart_disease.csv            # Benchmark dataset (Framingham Heart Study)
│   ├── hasil_eksperimen.json        # Serialized empirical results
│   └── README.md                    # Data provenance and DOI documentation
├── run_experiment.py                # Standalone reproducible execution script
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # Technical report and documentation
```

---

## Getting started

### Requirements
* Python 3.10 or higher
* Recommended: a clean virtual environment

### Installation and execution
```bash
# Clone the repository
git clone git@github.com:ramadhan-imanur/naive-bayes-probabilistic-classification.git
cd naive-bayes-probabilistic-classification

# Install dependencies
pip install -r requirements.txt

# Run the experiment script
python3 run_experiment.py
```

---

## References
1. OpenXAI Benchmark: *OpenXAI: Towards a Comprehensive Evaluation Benchmark for Explainable AI*. Harvard Dataverse, DOI: [10.7910/DVN/HIBCFL](https://doi.org/10.7910/DVN/HIBCFL).
2. National Institute of Diabetes and Digestive and Kidney Diseases: *Pima Indians Diabetes Database*. UCI Machine Learning Repository / Kaggle.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer, New York.
4. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill, New York.

---

## License
This project is available under the [MIT License](LICENSE).
