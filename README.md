# Predict Introverts vs. Extroverts - Kaggle Playground Series S5E7

## ⭕ Ongoing project - not complete

This repository contains a Google Colab notebook for the [Kaggle Playground Series Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/overview) competition, focused on predicting whether individuals are introverts or extroverts based on personality traits.

## Overview
The Kaggle Playground Series S5E7 is a binary classification task where the goal is to predict the `Personality` (Introvert or Extrovert) using features like time spent alone, social event attendance, and stage fear. This notebook performs exploratory data analysis (EDA), data preprocessing, and builds a baseline model to establish a starting point for the competition.

### Dataset
- **Source**: Kaggle competition dataset (`train.csv`, `test.csv`, `sample_submission.csv`).
- **Features**:
  - Numerical: `Time_spent_Alone`, `Social_event_attendance`, `Going_outside`, `Friends_circle_size`, `Post_frequency`.
  - Categorical: `Stage_fear`, `Drained_after_socializing`.
  - Target: `Personality` (Introvert/Extrovert).
- **Size**: 18,524 rows, 9 columns (including `id`).
- **Challenges**: Missing values (~5-10% per feature), potential class imbalance.

## Notebook Contents
The notebook (`s5e7_eda.ipynb`) includes:
1. **Data Loading**: Downloads the dataset using the Kaggle API in Google Colab.
2. **Exploratory Data Analysis (EDA)**:
   - Checks dataset shape, missing values, and feature distributions.
   - Visualizes target distribution and feature relationships using `seaborn`.
   - Analyzes correlations between numerical features and the target.
3. **Preprocessing**:
   - Imputes missing values (median for numerical, mode for categorical).
   - Encodes categorical features (`Stage_fear`, `Drained_after_socializing`) using one-hot encoding.
4. **Baseline Model**: Trains a logistic regression model and evaluates AUC-ROC.
5. **Submission**: Prepares predictions for the test set in the required Kaggle format.

## Setup Instructions
To run the notebook in Google Colab:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. **Upload to Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload `s5e7_eda.ipynb` from the cloned repository.
3. **Set Up Kaggle API**:
   - Download your Kaggle API key (`kaggle.json`) from [Kaggle Settings](https://www.kaggle.com/settings).
   - In Colab, run:
     ```python
     from google.colab import files
     files.upload()  # Upload kaggle.json
     !mkdir -p ~/.kaggle
     !cp kaggle.json ~/.kaggle/
     !chmod 600 ~/.kaggle/kaggle.json
     ```
4. **Install Dependencies**:
   - The notebook installs required libraries (e.g., `pandas`, `seaborn`, `scikit-learn`).
   - Install the Kaggle CLI:
     ```python
     !pip install kaggle
     ```
5. **Download Dataset**:
   - The notebook includes code to download the dataset:
     ```python
     !kaggle competitions download -c playground-series-s5e7
     !unzip playground-series-s5e7.zip
     ```

## Running the Notebook
- Open `s5e7_eda.ipynb` in Colab.
- Execute cells sequentially to load data, perform EDA, preprocess, and train the model.
- Modify the notebook to experiment with other models (e.g., `XGBoost`, `CatBoost`) or preprocessing techniques.

## Results (still ongoing project)
- **Baseline AUC-ROC**: [Add your baseline model’s performance here, e.g., 0.75].
- **Leaderboard Rank**: [Add your rank or score after submission, if applicable].
- **Next Steps**: Experiment with feature engineering, advanced models, or hyperparameter tuning to improve performance.

## Requirements
- Python 3.x (provided by Colab)
- Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `kaggle`
- Kaggle API key (`kaggle.json`)

## License
This project is licensed under the MIT License.

## Acknowledgments
- Kaggle for providing the dataset and competition platform.
- Built using Google Colab for accessibility and reproducibility.

For questions or contributions, open an issue or contact me.

Made with ❤️ by scythe410.
