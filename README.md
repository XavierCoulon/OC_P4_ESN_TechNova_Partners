# ESN TechNova Partners - Employee Analysis Project

## 🎯 Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) on employee data from multiple sources to identify key differences and patterns among employees.

## 📊 Datasets

The analysis combines data from three main sources:

-   **SIRH Dataset** (`extrait_sirh.csv`) - Human Resources Information System data
-   **Evaluation Dataset** (`extrait_eval.csv`) - Employee performance evaluations
-   **Survey Dataset** (`extrait_sondage.csv`) - Employee satisfaction surveys

## 🏗️ Project Structure

```
├── data/
│   ├── raw/                   # Original CSV datasets
│   ├── processed/             # Cleaned and merged datasets
│   └── external/              # External reference data
├── notebooks/
│   ├── exploratory/           # EDA notebooks
│   │   └── 01_exploratory_data_analysis.ipynb
│   ├── modeling/              # Machine learning notebooks
│   └── reports/               # Final analysis notebooks
├── main.py                    # Main Python script
├── pyproject.toml            # Project dependencies (uv)
└── README.md                 # Project documentation
```

## 🚀 Getting Started

### Prerequisites

-   Python 3.12+
-   [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/XavierCoulon/OC_P4_ESN_TechNova_Partners.git
cd OC_P4_ESN_TechNova_Partners
```

2. Install dependencies:

```bash
uv sync
```

3. Start Jupyter Lab:

```bash
uv run jupyter lab
```

## 📈 Analysis Workflow

1. **Data Loading**: Load and inspect individual datasets (SIRH, Evaluation, Survey)
2. **Data Quality Assessment**: Check for missing values, duplicates, and data types
3. **Join Strategy**: Identify common columns and merge datasets
4. **Central Dataset Creation**: Create unified employee dataset
5. **Statistical Analysis**: Generate descriptive statistics
6. **Visualizations**: Create charts to highlight employee differences
7. **Data Cleaning**: Use pandas `.apply()` method for data preprocessing

## 🔍 Key Analysis Areas

-   **Quantitative vs Quantitative**: Correlation analysis, scatter plots
-   **Quantitative vs Qualitative**: Box plots, grouped analysis
-   **Qualitative vs Qualitative**: Cross-tabulation, stacked charts

## 📦 Dependencies

Key packages used in this project:

-   `pandas` - Data manipulation and analysis
-   `numpy` - Numerical computing
-   `matplotlib` - Static visualizations
-   `seaborn` - Statistical visualizations
-   `scikit-learn` - Machine learning tools
-   `jupyterlab` - Interactive development environment

## 🎓 Educational Context

This project is part of the OpenClassrooms Data Scientist program, focusing on exploratory data analysis techniques and employee data insights.

## 📝 License

This project is for educational purposes.

## 👤 Author

Xavier Coulon - [GitHub](https://github.com/XavierCoulon)
