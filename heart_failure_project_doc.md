# Heart Failure Risk Analysis with Multiple Linear Regression

This project uses multiple linear regression to analyze and predict mortality risk in heart failure patients based on clinical features. It demonstrates how linear algebra concepts can be applied to real-world medical data to create predictive models.

## Project Overview

Heart disease is one of the leading causes of mortality worldwide. This analysis aims to:

1. Identify which clinical factors are most strongly associated with mortality risk in heart failure patients
2. Create a quantitative model for predicting this risk based on patient data
3. Analyze the effectiveness of linear regression for this type of prediction
4. Visualize and interpret the results in a clinically meaningful way

## Dataset

This project uses the "Heart Failure Clinical Records" dataset from Kaggle, which contains medical records of 299 heart failure patients with 13 clinical features:

- **age**: Age of the patient (years)
- **anaemia**: Decrease of red blood cells or hemoglobin (boolean)
- **creatinine_phosphokinase**: Level of the CPK enzyme in the blood (mcg/L)
- **diabetes**: If the patient has diabetes (boolean)
- **ejection_fraction**: Percentage of blood leaving the heart at each contraction (percentage)
- **high_blood_pressure**: If the patient has hypertension (boolean)
- **platelets**: Platelets in the blood (kiloplatelets/mL)
- **serum_creatinine**: Level of serum creatinine in the blood (mg/dL)
- **serum_sodium**: Level of serum sodium in the blood (mEq/L)
- **sex**: Woman or man (binary)
- **smoking**: If the patient smokes (boolean)
- **time**: Follow-up period (days)
- **DEATH_EVENT**: If the patient died during the follow-up period (boolean, target variable)

### Data Source
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/version/1

## Methodology

The analysis follows these steps:

1. **Data Loading and Exploration**:
   - Basic statistics and distribution analysis
   - Visualization of feature distributions and correlations
   - Assessment of class imbalance

2. **Data Cleaning**:
   - Removal of outliers according to clinical criteria:
     - CPK > 1200 mcg/L
     - Ejection fraction > 70%
     - Platelets < 70,000 or > 440,000 kiloplatelets/mL
     - Serum creatinine ≥ 2.2 mg/dL
     - Serum sodium ≤ 124 mEq/L

3. **Feature Preparation**:
   - Train-test split (75%-25%)
   - Standardization of features to ensure equal scale

4. **Model Building**:
   - Multiple linear regression using sklearn's LinearRegression
   - Evaluation using regression metrics (R², MAE)
   - Classification evaluation using a 0.5 probability threshold

5. **Result Interpretation**:
   - Analysis of model coefficients
   - Identification of key risk factors
   - Visualization of predictions vs. actual outcomes

## Linear Algebra Connection

The multiple linear regression model is fundamentally a linear algebra operation, represented by:

```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```

Where:
- Y is the predicted mortality risk (probability of death)
- X₁, X₂, ..., Xₙ are the patient's clinical measurements
- β₀, β₁, β₂, ..., βₙ are the coefficients determined by the model
- ε is the error term

In matrix form, this becomes:

```
Y = Xβ + ε
```

The solution that minimizes the sum of squared errors is:

```
β = (X^T X)^(-1) X^T Y
```

This involves several key linear algebra operations:
- Matrix multiplication
- Matrix transposition
- Matrix inversion

## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

Install requirements:
```
pip install -r requirements.txt
```

## Usage

1. Download the dataset from Kaggle and save it as `heart_failure_clinical_records_dataset.csv` in the project directory
2. Run the main analysis script:
   ```
   python heart_failure_analysis.py
   ```
3. View the results in the `plots/` and `results/` directories

## Results

The analysis produces several outputs:

1. **Data Visualizations**:
   - Correlation matrix
   - Feature distributions by death event
   - Feature boxplots by death event
   - Categorical feature counts by death event

2. **Model Evaluation**:
   - Regression metrics (R², MAE)
   - Classification metrics (precision, recall, F1-score)
   - Confusion matrices

3. **Interpretation**:
   - Feature coefficients visualization
   - Actual vs. predicted plot
   - Top risk factors identification

## Limitations

1. **Linear Model Limitations**: The linear regression model assumes a linear relationship between features and the target, which may not fully capture complex medical relationships.
2. **Outlier Impact**: The removal of outliers, while necessary for model stability, may exclude important edge cases.
3. **Binary Classification**: Using regression for binary classification (with a threshold) is not ideal, though it provides interpretable coefficients.
4. **Dataset Size**: After cleaning, the dataset size is reduced, which may impact generalizability.

## Future Work

1. Explore more appropriate classification models (logistic regression, random forests)
2. Conduct feature engineering to capture non-linear relationships
3. Perform cross-validation for more robust performance estimation
4. Test alternative approaches to outlier handling
5. Explore model calibration for better probability estimation

## License

This project uses the same license as the dataset (Please refer to the Kaggle dataset page for details). 