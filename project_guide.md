# Heart Failure Risk Analysis Project Guide

## Project Structure

This project contains the following files:

- `heart_failure_analysis.py`: Main Python script for the analysis
- `download_dataset.py`: Helper script to download the dataset
- `requirements.txt`: Python dependencies required for the project
- `heart_failure_project_doc.md`: Detailed project documentation
- `presentation_outline.md`: Guide for creating the presentation video
- `project_guide.md`: This file - a quick getting started guide

## Getting Started

### 1. Setup Environment

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Option 1: Run the download script
python download_dataset.py

# Option 2: Manual download
# Visit https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/version/1
# Download and save as heart_failure_clinical_records_dataset.csv
```

### 3. Run Analysis

```bash
python heart_failure_analysis.py
```

### 4. Review Results

1. Check the terminal output for model performance metrics
2. Explore the `plots/` directory for visualizations
3. Review the `results/` directory for detailed output files


## Project Goals

1. Demonstrate the application of linear algebra concepts to a real-world problem
2. Analyze which clinical factors predict mortality in heart failure patients
3. Build a quantitative model using multiple linear regression
4. Visualize and interpret results in a clinically meaningful way

## Key Linear Algebra Concepts

The project demonstrates these fundamental linear algebra concepts:

- Matrix operations (multiplication, transposition, inversion)
- Vector operations
- Linear transformations
- Least squares method for solving overdetermined systems

For more details, see the complete documentation in `heart_failure_project_doc.md`. 