import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
import os



plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

def load_data(filepath='heart_failure_clinical_records_dataset.csv'):
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        print("Please download the dataset from Kaggle: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/version/1")
        print("Save it as 'heart_failure_clinical_records_dataset.csv' in the same directory as this script.")
        return None

def explore_data(df):
    print("\n--- Dataset Information ---")
    print(df.info())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Class Distribution ---")
    print(df['DEATH_EVENT'].value_counts())
    print(f"Death rate: {df['DEATH_EVENT'].mean() * 100:.2f}%")
    
    plt.figure(figsize=(14, 10))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Heart Failure Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    
    numeric_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 
                        'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numeric_features):
        plt.subplot(3, 3, i+1)
        sns.histplot(data=df, x=feature, hue='DEATH_EVENT', element='step', 
                    common_norm=False, stat='density', bins=20)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('plots/numeric_features_distribution.png')
    
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numeric_features):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x='DEATH_EVENT', y=feature, data=df)
        plt.title(f'Boxplot of {feature} by Death Event')
    plt.tight_layout()
    plt.savefig('plots/numeric_features_boxplot.png')
    
    categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(categorical_features):
        plt.subplot(2, 3, i+1)
        sns.countplot(x=feature, hue='DEATH_EVENT', data=df)
        plt.title(f'Count of {feature} by Death Event')
    plt.tight_layout()
    plt.savefig('plots/categorical_features_count.png')
    
    print("\nExploratory visualizations saved to 'plots' directory.")

def clean_data(df):
    print("\n--- Data Cleaning ---")
    original_size = df.shape[0]
    
    #Definindo criterio dos outliers para remover
    outlier_criteria = {
        'creatinine_phosphokinase': lambda x: x > 1200,
        'ejection_fraction': lambda x: x > 70,
        'platelets': lambda x: (x < 70000) | (x > 440000),
        'serum_creatinine': lambda x: x >= 2.2,
        'serum_sodium': lambda x: x <= 124
    }
    
    outliers_removed = {}
    
    for feature, criterion in outlier_criteria.items():
        outliers = criterion(df[feature])
        outliers_count = outliers.sum()
        if outliers_count > 0:
            outliers_removed[feature] = outliers_count
            df = df[~outliers]
    
    print("Outliers removed:")
    for feature, count in outliers_removed.items():
        print(f"  - {feature}: {count} records")
    
    print(f"Original dataset size: {original_size}")
    print(f"Cleaned dataset size: {df.shape[0]}")
    print(f"Removed {original_size - df.shape[0]} records ({((original_size - df.shape[0]) / original_size) * 100:.2f}%)")
    
    return df

def prepare_features(df):
    print("\n--- Feature Preparation ---")
    
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    feature_names = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def build_linear_regression_model(X_train, y_train, X_test, y_test, feature_names):
    print("\n--- Linear Regression Model ---")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    print("Regression Metrics:")
    print(f"  R² (Train): {r2_train:.4f}")
    print(f"  R² (Test): {r2_test:.4f}")
    print(f"  Mean Absolute Error (Train): {mae_train:.4f}")
    print(f"  Mean Absolute Error (Test): {mae_test:.4f}")
    
    y_train_pred_binary = (y_train_pred >= 0.5).astype(int)
    y_test_pred_binary = (y_test_pred >= 0.5).astype(int)
    
    print("\nClassification Metrics (using 0.5 threshold):")
    print("Training set:")
    print(classification_report(y_train, y_train_pred_binary))
    print("Confusion Matrix (Training):")
    train_cm = confusion_matrix(y_train, y_train_pred_binary)
    print(train_cm)
    
    print("\nTest set:")
    print(classification_report(y_test, y_test_pred_binary))
    print("Confusion Matrix (Test):")
    test_cm = confusion_matrix(y_test, y_test_pred_binary)
    print(test_cm)
    
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Absolute Value': np.abs(model.coef_)
    }).sort_values('Absolute Value', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
    plt.title('Feature Coefficients in Linear Regression Model', fontsize=16)
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig('plots/feature_coefficients.png')
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Death Event')
    plt.ylabel('Predicted Death Event Probability')
    plt.title('Actual vs. Predicted Death Event', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png')
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Survived', 'Died'], 
                yticklabels=['Survived', 'Died'])
    plt.title('Confusion Matrix - Training Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Survived', 'Died'], 
                yticklabels=['Survived', 'Died'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png')
    
    coefficients.to_csv('results/feature_coefficients.csv', index=False)
    
    print("\nVisualizations and results saved to 'plots' and 'results' directories.")
    
    return model

def interpret_results(feature_coefficients):
    print("\n--- Model Interpretation ---")

    top_positive = feature_coefficients[feature_coefficients['Coefficient'] > 0].head(3)
    top_negative = feature_coefficients[feature_coefficients['Coefficient'] < 0].head(3)
    
    print("Top risk-increasing factors (positive coefficients):")
    for i, (_, row) in enumerate(top_positive.iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
    
    print("\nTop protective factors (negative coefficients):")
    for i, (_, row) in enumerate(top_negative.iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
    
    print("\nClinical interpretation:")
    print("  - The model has identified several key factors associated with mortality risk in heart failure patients.")
    print("  - These findings align with clinical knowledge that factors like serum creatinine levels,")
    print("    age, and ejection fraction are important predictors of heart failure outcomes.")
    print("  - The linear regression approach provides a quantifiable risk score, potentially")
    print("    useful for initial patient risk stratification.")

def main():
    print("==========================================")
    print("Heart Failure Risk Analysis using Multiple Linear Regression")
    print("==========================================")
    
    df = load_data()
    if df is None:
        return
    
    explore_data(df)
    
    df_cleaned = clean_data(df)
    
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_features(df_cleaned)
    
    model = build_linear_regression_model(X_train, y_train, X_test, y_test, feature_names)

    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Absolute Value': np.abs(model.coef_)
    }).sort_values('Absolute Value', ascending=False)
    
    interpret_results(coefficients)
    
    print("\nAnalysis complete. Thank you for using the Heart Failure Risk Analysis tool.")

if __name__ == "__main__":
    main() 