<div align="center">

# 📊 Feature Engineering Technique

![scikit-learn](https://img.shields.io/badge/scikit--learn-0C7BDC?logo=scikit-learn&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) ![numpy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white) ![matplotlib](https://img.shields.io/badge/matplotlib-11557c?logo=matplotlib&logoColor=white)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![Status](https://img.shields.io/badge/status-active-brightgreen) ![GitHub stars](https://img.shields.io/github/stars/NhanPhamThanh-IT/Feature-Engineering-Technique?style=social) ![GitHub forks](https://img.shields.io/github/forks/NhanPhamThanh-IT/Feature-Engineering-Technique?style=social) ![GitHub issues](https://img.shields.io/github/issues/NhanPhamThanh-IT/Feature-Engineering-Technique) ![GitHub pull requests](https://img.shields.io/github/issues-pr/NhanPhamThanh-IT/Feature-Engineering-Technique) ![Last Commit](https://img.shields.io/github/last-commit/NhanPhamThanh-IT/Feature-Engineering-Technique)

</div>

<div align="justify">

## Introduction

Feature engineering is the process of transforming raw data into meaningful features that better represent the underlying problem to predictive models, resulting in improved model accuracy and performance. This repository is a comprehensive resource for mastering feature engineering and feature selection, providing both theoretical background and practical code for real-world machine learning workflows.

Whether you are a student, data scientist, or ML engineer, you will find step-by-step tutorials, reusable Python modules, and best practices to enhance your data preprocessing pipeline.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Quickstart on Windows](#quickstart-on-windows)
- [Features](#features)
- [Methodology](#methodology)
- [Best Practices](#best-practices)
- [Datasets](#datasets)
- [Notebooks](#notebooks)
- [Images & Visualizations](#images--visualizations)
- [API Cheatsheet](#api-cheatsheet)
- [Examples](#examples)
- [Real-World Applications](#real-world-applications)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [References](#references)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Project Structure

```
├── datasets/           # Example datasets for practice
├── docs/               # Documentation and guides
├── images/             # Visualizations and diagrams
├── notebook/           # Jupyter notebooks (theory & practice)
├── src/                # Source code and practical notebooks
│   ├── output/         # Output plots and CSVs
│   └── utils/          # Python modules for feature engineering
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # Project overview
```

## Features

- **Data Exploration**: Summary statistics, visualization, and correlation analysis.
- **Missing Data Handling**: Imputation, removal, and analysis of missing values.
- **Outlier Detection**: Multiple statistical methods for identifying and handling outliers.
- **Rare Value Handling**: Grouping and encoding rare categories.
- **Feature Scaling**: Standardization, normalization, robust scaling, min-max, and more.
- **Discretization**: Binning continuous variables.
- **Feature Encoding**: Label, one-hot, target, and advanced encoders.
- **Feature Transformation**: Log, reciprocal, square root, and other mathematical transforms.
- **Feature Generation**: Creating new features from existing data.
- **Feature Selection**: Filter, wrapper, embedded, shuffling, and hybrid methods.
- **Rich Visualizations**: Plots for EDA, scaling, selection, and more.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/NhanPhamThanh-IT/Feature-Engineering-Technique.git
   cd Feature-Engineering-Technique
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

Note:

- Notebooks use Jupyter. If you don't have it yet: `pip install notebook` (or `pip install jupyterlab`).
- Some utilities use seaborn for plotting. Ensure seaborn is installed (included in requirements).

## Example Workflow

Below is a typical workflow you can follow using this repository:

1. **Load and Explore Data**
   - Use notebooks in `src/` or `notebook/` to load datasets and perform initial EDA (see [output examples](src/output/)).
2. **Handle Missing Data & Outliers**
   - Apply imputation and outlier detection methods from `src/utils/`.
3. **Feature Engineering**
   - Scale, encode, transform, and generate new features using provided modules and notebooks.
4. **Feature Selection**
   - Use filter, wrapper, embedded, and hybrid methods to select the best features.
5. **Modeling & Evaluation**
   - Use the processed data for machine learning models and visualize results.

## Usage

Explore the Jupyter notebooks in the `notebook/` and `src/` folders for step-by-step tutorials and code examples. Each notebook focuses on a specific aspect of feature engineering or selection, with explanations and visualizations.

Example:

```sh
jupyter notebook notebook/feature-engineering/01_What-is-Feature-Engineering.ipynb
```

You can also use the utility Python modules in `src/utils/` for your own projects:

```python
from src.utils.missing_data import *
from src.utils.outlier import *
# ...and more
```

When running notebooks from `src/`, outputs (plots/CSV) are saved into `src/output/` when an `output_path` argument is provided.

## Quickstart on Windows

Use a virtual environment and run notebooks from this repo.

1. Create and activate a venv (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
# Optional editors/notebook tools
pip install notebook
```

3. Launch Jupyter and open any notebook:

```powershell
jupyter notebook
# or
jupyter lab
```

4. Open, for example, `notebook/feature-engineering/04_Feature-Scaling.ipynb` and run the cells.

## Advanced Tips

- Try combining multiple feature selection methods for robust results.
- Use the visualizations in `images/` and `src/output/` to compare before/after effects of each technique.
- For large datasets, consider sampling or using efficient pandas/numpy operations to speed up processing.
- Explore the [docs/A-Short-Guide-for-Feature-Engineering-and-Feature-Selection.pdf](docs/A-Short-Guide-for-Feature-Engineering-and-Feature-Selection.pdf) for deeper theory.

## Example Outputs

You can find example plots and CSVs generated by the notebooks in `src/output/`, such as:

- Correlation heatmaps
- Boxplots and barplots
- Missing value summaries
- Feature importance rankings

## How to Extend

- Add your own datasets to `datasets/` and update the notebook paths.
- Create new utility functions in `src/utils/` for custom feature engineering needs.
- Share your results and improvements via pull requests!

## Datasets

Sample datasets are provided in the `datasets/` folder, including:

- Titanic
- Pima Indians Diabetes
- Housing

Notes:

- `datasets/titanic.csv` contains typical Titanic passenger fields (Pclass, Sex, Age, Fare, Survived, etc.).
- `datasets/pima-indians-diabetes.data.csv` is the classic Pima dataset for binary classification.
- `datasets/housing.data.txt` is a Boston Housing-style dataset for regression tasks.
- `datasets/SampleFile.csv` is a small sample to test I/O.

You can add your own files to `datasets/` and update notebook paths accordingly.

## Notebooks

Theory and practice notebooks are organized as follows:

- `notebook/feature-engineering/` — Theory, concepts, and real-world examples
- `src/` — Practical, hands-on feature engineering with code and outputs

Highlights in `src/` include:

- `1_Data_Explore.ipynb` — EDA walkthrough with plots and correlations.
- `2.1_Missing_Data.ipynb` — Strategies for NA detection and imputation.
- `2.2_Outlier.ipynb` — Outlier detection via IQR, MAD, mean±k·std, and handling.
- `2.3_Rare_Values.ipynb` — Grouping rare categories and mode imputation.
- `3.1_Feature_Scaling.ipynb` — Standardization, normalization, robust/min-max scaling.
- `3.2_Discretisation.ipynb` — ChiMerge and tree-based discretization.
- `3.3_Feature_Encoding.ipynb` — One-hot, target/mean encoding, category encoders.
- `3.4_Feature_Transformation.ipynb` — Log, reciprocal, sqrt, power transforms + Q-Q diagnostics.
- `3.5_Feature_Generation.ipynb` — Create new features and interactions.
- `4.1`–`4.5` — Feature selection (filter, wrapper, embedded, shuffling, hybrid).

## Images & Visualizations

## API Cheatsheet

A quick tour of the reusable utilities under `src/utils/`.

- `data_explore.py`

  - `get_dtypes(df, drop_col=[])` → (str_cols, num_cols, all_cols)
  - `describe(df, output_path=None)` → DataFrame; saves `describe.csv` if path given
  - Plotters: `discrete_var_barplot`, `discrete_var_countplot`, `discrete_var_boxplot`,
    `continuous_var_distplot`, `scatter_plot`, `correlation_plot`, `heatmap`

- `missing_data.py`

  - `check_missing(df, output_path=None)` → counts and proportions (CSV if path)
  - `drop_missing(df, axis=0)` → drop rows/cols with NA
  - `add_var_denote_NA(df, NA_col=[...])` → adds `<col>_is_NA` indicator
  - `impute_NA_with_arbitrary(df, value, NA_col=[...])` → `<col>_<value>`
  - `impute_NA_with_avg(df, strategy='mean', NA_col=[...])` → `<col>_impute_*`
  - `impute_NA_with_end_of_distribution(df, NA_col=[...])` → mean+3·std
  - `impute_NA_with_random(df, NA_col=[...], random_state=0)` → `<col>_random`

- `outlier.py`

  - Detection: `outlier_detect_arbitrary`, `outlier_detect_IQR`, `outlier_detect_mean_std`, `outlier_detect_MAD`
  - Handling: `impute_outlier_with_arbitrary`, `impute_outlier_with_avg`, `windsorization`, `drop_outlier`

- `rare_values.py`

  - `GroupingRareValues(threshold=0.01)` → `.fit/.transform` to map infrequent categories to 'rare'
  - `ModeImputation(threshold=0.01)` → replace infrequent categories with column mode

- `encoding.py`

  - `MeanEncoding(cols=[...])` → target/mean encoding; `.fit(X, y)` then `.transform(X)`

- `discretization.py`

  - `ChiMerge(col='feature', num_of_bins=10, confidenceVal=3.841)` → supervised binning; `.fit(X, y)`
  - `DiscretizeByDecisionTree(col='feature', max_depth=3 or [1,2,3])` → tree-based score binning

- `filter_method.py`

  - `constant_feature_detect(df, threshold=0.98)` → quasi-constant columns
  - `corr_feature_detect(df, threshold=0.8)` → groups of correlated features
  - `mutual_info(X, y, select_k=10 or 0<k<1)`; `chi_square_test(...)`
  - `univariate_roc_auc(Xtr, ytr, Xte, yte, threshold)`; `univariate_mse(...)`

- `embedded_method.py`

  - `rf_importance(...)` and `gbt_importance(...)` → fit and plot top feature importances

- `feature_shuffle.py`

  - `feature_shuffle_rf(X_train, y_train, ...)` → permutation importance (AUC drop)

- `hybrid_method.py`
  - `recursive_feature_elimination_rf(...)` and `recursive_feature_addition_rf(...)`

Tip: Most functions copy input DataFrames and return new objects; original inputs remain unchanged unless noted.

## Examples

Minimal example with Titanic data using a few utilities:

```python
import pandas as pd
from src.utils.data_explore import get_dtypes, correlation_plot
from src.utils.missing_data import check_missing, impute_NA_with_avg
from src.utils.outlier import outlier_detect_IQR, windsorization

df = pd.read_csv('datasets/titanic.csv')

# 1) Explore dtypes
str_cols, num_cols, _ = get_dtypes(df, drop_col=['PassengerId', 'Name'])

# 2) Missing summary and simple impute for Age
miss = check_missing(df)
df2 = impute_NA_with_avg(df, strategy='median', NA_col=['Age'])

# 3) Outliers on Fare
mask, bounds = outlier_detect_IQR(df2, col='Fare', threshold=1.5)
df3 = windsorization(df2, col='Fare', bounds=bounds)

# 4) Correlation plot for numerics
correlation_plot(df3[num_cols], output_path='src/output')
```

Run inside your virtual environment. Check `src/output/` for saved plots.

The `images/` and `src/output/` folders contain diagrams and plots illustrating key concepts and results.

## Methodology

This project follows a modular and reproducible approach to feature engineering. Each notebook and utility script is self-contained, with clear explanations and code comments. The workflow typically includes:

1. **Data Exploration**: Understand the dataset, visualize distributions, and identify potential issues.
2. **Data Cleaning**: Handle missing values, outliers, and rare categories.
3. **Feature Engineering**: Apply scaling, encoding, transformation, and generation techniques.
4. **Feature Selection**: Use statistical and model-based methods to select the most relevant features.
5. **Evaluation**: Visualize results and compare model performance before and after feature engineering.

## Best Practices

- Always visualize your data before and after each transformation.
- Use domain knowledge to guide feature creation and selection.
- Avoid data leakage by separating training and test data during preprocessing.
- Document every step for reproducibility.
- Experiment with multiple techniques and compare their impact on model performance.

## Real-World Applications

Feature engineering is used in a wide range of domains, including:

- **Finance**: Credit scoring, fraud detection, risk modeling
- **Healthcare**: Disease prediction, patient risk stratification
- **Retail**: Customer segmentation, recommendation systems
- **Cybersecurity**: Intrusion detection, anomaly detection
- **Natural Language Processing**: Text classification, sentiment analysis
- **Computer Vision**: Image retrieval, object detection

## Troubleshooting

- If you encounter errors with missing packages, ensure you have installed all dependencies from `requirements.txt`.
- For issues with Jupyter notebooks, restart the kernel and clear outputs.
- If a dataset is missing, check the `datasets/` folder or update the file path in the notebook.
- If plots don't render, ensure a GUI backend is available or run in Jupyter. For headless runs, save plots to files.
- Some functions print to stdout (e.g., outlier counts). This is expected for quick diagnostics.
- When using supervised techniques (mean encoding, ChiMerge, feature importances), split data properly to avoid leakage.

## FAQ

**Q: Can I use my own datasets?**
A: Yes! Simply place your dataset in the `datasets/` folder and update the notebook paths as needed.

**Q: Do I need a GPU?**
A: No, all code is designed to run efficiently on CPU for small to medium datasets.

**Q: How do I cite this project?**
A: Please reference the GitHub repository URL in your work.

**Q: Which Python versions are supported?**
A: Python 3.8+ is recommended. The notebooks have been tested on Python 3.11 as well.

**Q: Do I need to install Jupyter?**
A: Yes, to run notebooks. Install via `pip install notebook` or use VS Code's Jupyter extension.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your branch and open a pull request

Please follow the existing code style and add documentation for any new features.

Suggested contribution ideas:

- Add more datasets and corresponding tutorial notebooks.
- Extend utilities with additional encoders or scalers (with tests).
- Add unit tests for core functions and CI workflow (GitHub Actions).
- Improve visualizations and examples in `src/output/`.

## References

- [A Short Guide for Feature Engineering and Feature Selection (PDF)](docs/A-Short-Guide-for-Feature-Engineering-and-Feature-Selection.pdf)
- [Featuretools](https://www.featuretools.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Acknowledgments

- Open-source contributors and the data science community
- Authors of [A Short Guide for Feature Engineering and Feature Selection](docs/A-Short-Guide-for-Feature-Engineering-and-Feature-Selection.pdf)
- Scikit-learn, pandas, numpy, matplotlib, and other library maintainers

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Conclusion

Feature engineering is a critical step in building effective machine learning models. This repository provides both theoretical background and practical tools to help you master feature engineering and selection. Explore, experiment, and contribute to make the most of your data science journey!

</div>

---

<div align="center">

🩷 _Created by Nhan Pham Thanh — 2025. Feel free to contribute!_ 🩷

</div>
