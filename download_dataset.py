import os
import urllib.request
import zipfile
import pandas as pd

def download_heart_failure_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    
    csv_path = "heart_failure_clinical_records_dataset.csv"
    
    print(f"Downloading Heart Failure Clinical Records dataset...")
    
    try:
        urllib.request.urlretrieve(url, csv_path)
        
        df = pd.read_csv(csv_path)
        

        expected_columns = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
            'ejection_fraction', 'high_blood_pressure', 'platelets', 
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 
            'time', 'DEATH_EVENT'
        ]
        
        all_columns_present = all(col in df.columns for col in expected_columns)
        
        if all_columns_present and len(df) > 0:
            print(f"Dataset downloaded successfully with {len(df)} records.")
            print(f"Saved to: {os.path.abspath(csv_path)}")
        else:
            raise Exception("Dataset downloaded but missing expected columns.")
        
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    download_heart_failure_dataset() 