# CMPE 256 Project
FALL 2025 CMPE 256 - Recommender Systems Project

## Project Set-Up
If using Google Colab:
```python
# Get data folder
!git clone https://github.com/paulsoriiiano/cmpe-256-project
!mv cmpe-256-project/data .
!rm -rf cmpe-256-project

# Data folder should be available on Colab notebook
df = pd.read_csv("data/train_2_long.csv")
....
```

## Implementation Details
The following `README.md` file is available under this (folder)[harshit/README.md]

### CMPE 256 Project: Music Playlist Continuation

### Overview
This project implements a recommender system to predict the next 20 tracks for a user based on their listening history. It compares four algorithms: Popularity, Item-based CF, SVD, and Item2Vec.

### Folder Structure
- `data/`: Contains raw and processed data.
- `output/`: Contains evaluation results and final submission (`submission.csv`).
- `notebooks/`: Jupyter notebooks (if any).
- `preprocessing.py`: Script to load and clean data.
- `models.py`: Implementation of recommendation algorithms.
- `evaluate.py`: Script to evaluate models.
- `generate_submission.py`: Script to generate final recommendations.

### Instructions

#### 1. Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn scipy gensim matplotlib
```

#### 2. Preprocessing
Run the preprocessing script to clean the data and create train/val splits:
```bash
python3 preprocessing.py
```

#### 3. Evaluation
Run the evaluation script to compare models:
```bash
python3 evaluate.py
```
Results will be saved to `output/evaluation_results.csv`.

### 🚀 Running the Demo

This project includes a polished Streamlit UI for demonstrating the recommender system.

1.  **Setup Environment:**
    ```bash
    ./run_demo.sh
    ```
    This script will automatically:
    *   Activate the virtual environment (or create it if missing).
    *   Install necessary dependencies (`streamlit`, `plotly`, etc.).
    *   Launch the Streamlit app in your browser.

2.  **Manual Run:**
    If you prefer to run it manually:
    ```bash
    source venv/bin/activate
    streamlit run app.py
    ```.

#### 4. Generate Submission
Generate the final recommendations using the best model (ItemCF):
```bash
python3 generate_submission.py
```
The output will be saved to `output/submission.csv`.

### Results
Item-based Collaborative Filtering (ItemCF) was the best-performing model with an NDCG@20 of 0.0523.
