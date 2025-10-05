import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import zscore


# Function to analyze dataset characteristics
def analyze_dataset(df, name):
    print(f"=== {name} ANALYSIS ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # Identify quantitative and qualitative columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"\nQuantitative columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Qualitative columns ({len(categorical_cols)}): {categorical_cols}")

    if numerical_cols:
        print(f"\nNumerical summary:")
        print(df[numerical_cols].describe())

    if categorical_cols:
        print(f"\nCategorical summary:")
        for col in categorical_cols:
            print(f"{col}: {df[col].nunique()} unique values")
            if df[col].nunique() <= 10:
                print(f"  Values: {df[col].value_counts().to_dict()}")

    print("=" * 50)
    return numerical_cols, categorical_cols


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def identify_feature_types(df, exclude_cols=None):
    """
    Identify different types of features in the dataset.

    Parameters:
    -----------
    df : DataFrame
                    Input dataset
    exclude_cols : list, optional
                    Columns to exclude from analysis

    Returns:
    --------
    dict
                    Dictionary containing different feature types:
                    - numerical_continuous: Continuous numerical features
                    - numerical_discrete: Discrete numerical features
                    - categorical_ordinal: Ordinal categorical features
                    - categorical_nominal: Nominal categorical features
                    - binary: Binary features (2 unique values)
                    - id_columns: Identifier columns

    Example:
    --------
    >>> feature_types = identify_feature_types(df, exclude_cols=['employee_id'])
    >>> print(feature_types['binary'])
    ['gender', 'remote_work']
    """
    if exclude_cols is None:
        exclude_cols = []

    # Work with columns excluding specified ones
    working_cols = [col for col in df.columns if col not in exclude_cols]
    working_df = df[working_cols]

    feature_types = {
        "numerical_continuous": [],
        "numerical_discrete": [],
        "categorical_ordinal": [],
        "categorical_nominal": [],
        "binary": [],
        "id_columns": [],
    }

    for col in working_cols:
        # Identify ID columns
        if any(
            keyword in col.lower() for keyword in ["id", "code", "number"]
        ) and col.lower() not in ["age", "note_evaluation"]:
            feature_types["id_columns"].append(col)
            continue

        # Identify binary variables
        if working_df[col].nunique() == 2:
            feature_types["binary"].append(col)
            continue

        # Identify numerical vs categorical
        if pd.api.types.is_numeric_dtype(working_df[col]):
            # Check if it's discrete (like ratings) or continuous
            unique_count = working_df[col].nunique()
            value_range = (
                working_df[col].max() - working_df[col].min()
                if working_df[col].max() is not pd.NA
                else 0
            )

            if unique_count <= 10 and value_range <= 10:
                # Likely ordinal (ratings, scores)
                feature_types["categorical_ordinal"].append(col)
            elif unique_count <= 20:
                feature_types["numerical_discrete"].append(col)
            else:
                feature_types["numerical_continuous"].append(col)
        else:
            # Categorical variables
            unique_count = working_df[col].nunique()
            if unique_count <= 20:  # Manageable for encoding
                feature_types["categorical_nominal"].append(col)

    return feature_types


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """
    Détection des outliers avec la méthode Z-score.
    Z = (x - moyenne) / écart-type
    Tout point avec |Z| > threshold est considéré comme un outlier.
    """
    col_data = pd.to_numeric(df[column], errors="coerce")
    mean = np.nanmean(col_data)
    std = np.nanstd(col_data)

    if std == 0 or np.isnan(std):
        return pd.DataFrame(columns=[column]), pd.Series(
            np.zeros(len(df)), index=df.index
        )

    # Calcul manuel des Z-scores → évite tout conflit de typage
    z_scores = (col_data - float(mean)) / float(std)
    z_scores = z_scores.abs()

    # Masque des outliers
    outlier_mask = z_scores > threshold
    outliers = df.loc[outlier_mask, [column]]

    return outliers, z_scores


def create_correlation_matrix(df, method="pearson", threshold=0.8):
    """
    Create correlation matrix and identify highly correlated features.

    Parameters:
    -----------
    df : DataFrame
                    Input dataset with numerical features only
    method : str, default='pearson'
                    Correlation method: 'pearson' or 'spearman'
    threshold : float, default=0.8
                    Correlation threshold for identifying high correlations

    Returns:
    --------
    tuple
                    (correlation_matrix, highly_correlated_pairs)
                    - correlation_matrix: DataFrame with correlation values
                    - highly_correlated_pairs: List of dicts with highly correlated pairs

    Example:
    --------
    >>> corr_matrix, high_corr = create_correlation_matrix(df_numerical, threshold=0.7)
    >>> print(f"Found {len(high_corr)} highly correlated pairs")
    """
    # Calculate correlation matrix
    corr_matrix = df.corr(method=method)

    # Find highly correlated pairs
    highly_correlated = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                highly_correlated.append(
                    {
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    return corr_matrix, sorted(
        highly_correlated, key=lambda x: abs(x["correlation"]), reverse=True
    )


def apply_binary_encoding(df, columns, mapping_dict=None):
    """
    Apply binary encoding to specified columns.

    Parameters:
    -----------
    df : DataFrame
                    Input dataset
    columns : list
                    List of columns to encode
    mapping_dict : dict, optional
                    Custom mapping {column: {value1: 0, value2: 1}}

    Returns:
    --------
    tuple
                    (encoded_df, encoding_info)
                    - encoded_df: DataFrame with binary encoded columns
                    - encoding_info: Dict with encoding details for each column

    Example:
    --------
    >>> custom_mapping = {'gender': {'Male': 0, 'Female': 1}}
    >>> df_encoded, info = apply_binary_encoding(df, ['gender'], custom_mapping)
    """
    df_encoded = df.copy()
    encoding_info = {}

    for col in columns:
        if col in df_encoded.columns:
            unique_vals = df_encoded[col].unique()

            if len(unique_vals) != 2:
                print(f"⚠️ Warning: {col} has {len(unique_vals)} unique values, not 2")
                continue

            if mapping_dict and col in mapping_dict:
                encoding_map = mapping_dict[col]
            else:
                # Default binary encoding
                encoding_map = {unique_vals[0]: 0, unique_vals[1]: 1}

            df_encoded[col] = df_encoded[col].map(encoding_map)
            encoding_info[col] = {"method": "binary", "mapping": encoding_map}
            print(f"✅ Binary encoded {col}: {encoding_map}")

    return df_encoded, encoding_info


def apply_label_encoding(df, columns):
    """
    Apply label encoding to specified columns (for ordinal data).

    Parameters:
    -----------
    df : DataFrame
                    Input dataset
    columns : list
                    List of columns to encode

    Returns:
    --------
    tuple
                    (encoded_df, encoding_info)
                    - encoded_df: DataFrame with label encoded columns
                    - encoding_info: Dict with encoding details for each column

    Example:
    --------
    >>> df_encoded, info = apply_label_encoding(df, ['note_evaluation'])
    >>> print(info['note_evaluation']['mapping'])
    """
    df_encoded = df.copy()
    encoding_info = {}

    for col in columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoding_info[col] = {
                "method": "label_encoder",
                "classes": le.classes_.tolist(),
                "mapping": {cls: idx for idx, cls in enumerate(le.classes_)},
            }
            print(f"✅ Label encoded {col}: {len(le.classes_)} classes")

    return df_encoded, encoding_info


def apply_onehot_encoding(df, columns, drop_first=True, prefix=None):
    """
    Apply one-hot encoding to specified columns using sklearn OneHotEncoder (for nominal data).

    Parameters:
    -----------
    df : DataFrame
                    Input dataset
    columns : list
                    List of columns to encode
    drop_first : bool, default=True
                    Whether to drop first category to avoid multicollinearity
    prefix : list, optional
                    Custom prefixes for each column

    Returns:
    --------
    tuple
                    (encoded_df, encoding_info)
                    - encoded_df: DataFrame with one-hot encoded columns (dtype: int64)
                    - encoding_info: Dict with encoding details for each column

    Example:
    --------
    >>> df_encoded, info = apply_onehot_encoding(df, ['departement'], drop_first=True)
    >>> print(info['departement']['new_columns'])
    """
    df_encoded = df.copy()
    encoding_info = {}

    for i, col in enumerate(columns):
        if col in df_encoded.columns:
            # Set prefix
            if prefix and i < len(prefix):
                col_prefix = prefix[i]
            else:
                col_prefix = col

            # Get original categories before encoding
            original_categories = df_encoded[col].unique().tolist()

            # Initialize OneHotEncoder
            encoder = OneHotEncoder(
                drop="first" if drop_first else None,
                sparse_output=False,  # Return dense array instead of sparse
                dtype=np.int64,  # Ensure integer output (0/1)
            )

            # Fit and transform the column
            encoded_array = encoder.fit_transform(df_encoded[[col]])

            # Get feature names from the encoder (this handles the drop_first automatically)
            try:
                # For newer sklearn versions
                new_column_names = encoder.get_feature_names_out([col]).tolist()
            except AttributeError:
                # Fallback for older sklearn versions
                fitted_categories = np.array(encoder.categories_[0])
                if drop_first and len(fitted_categories) > 1:
                    categories_to_use = fitted_categories[1:]
                else:
                    categories_to_use = fitted_categories
                new_column_names = [f"{col_prefix}_{cat}" for cat in categories_to_use]

            # Create DataFrame with encoded columns
            encoded_df = pd.DataFrame(
                encoded_array, columns=new_column_names, index=df_encoded.index
            )

            # Drop original column and add encoded columns
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

            # Store encoding information
            try:
                categories_list = np.array(encoder.categories_[0]).tolist()
            except:
                categories_list = original_categories

            encoding_info[col] = {
                "method": "one_hot_sklearn",
                "encoder": encoder,
                "original_categories": original_categories,
                "new_columns": new_column_names,
                "drop_first": drop_first,
                "categories": categories_list,
            }
            print(
                f"✅ One-hot encoded {col}: {len(new_column_names)} new columns (dtype: int64)"
            )

    return df_encoded, encoding_info


def apply_ordinal_encoding(df, columns, ordinal_mappings=None):
    """
    Apply ordinal encoding with custom order to specified columns.

    Parameters:
    -----------
    df : DataFrame
                    Input dataset
    columns : list
                    List of columns to encode
    ordinal_mappings : dict, optional
                    Custom ordinal mappings {column: [ordered_categories]}

    Returns:
    --------
    tuple
                    (encoded_df, encoding_info)
                    - encoded_df: DataFrame with ordinal encoded columns
                    - encoding_info: Dict with encoding details for each column

    Example:
    --------
    >>> ordinal_maps = {'education': ['Primary', 'Secondary', 'Bachelor', 'Master', 'PhD']}
    >>> df_encoded, info = apply_ordinal_encoding(df, ['education'], ordinal_maps)
    """
    df_encoded = df.copy()
    encoding_info = {}

    for col in columns:
        if col in df_encoded.columns:
            if ordinal_mappings and col in ordinal_mappings:
                # Use custom ordering
                ordered_categories = ordinal_mappings[col]
                mapping = {cat: idx for idx, cat in enumerate(ordered_categories)}
            else:
                # Use natural ordering for numerical-like data
                unique_vals = sorted(df_encoded[col].unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}

            df_encoded[col] = df_encoded[col].map(mapping)
            encoding_info[col] = {
                "method": "ordinal",
                "mapping": mapping,
                "ordered_categories": list(mapping.keys()),
            }
            print(f"✅ Ordinal encoded {col}: {len(mapping)} levels")

    return df_encoded, encoding_info


def analyze_feature_scaling(X, numerical_cols=None, verbose=True):
    """
    Analyse les échelles des features numériques pour vérifier la nécessité d'un scaling.

    Parameters
    ----------
    X : pd.DataFrame
            DataFrame contenant les features.
    numerical_cols : list, optional
            Liste des colonnes numériques à analyser. Si None, toutes les colonnes numériques sont utilisées.
    verbose : bool
            Si True, affiche le rapport complet.

    Returns
    -------
    scale_info : dict
            Dictionnaire contenant les statistiques des features et recommandations de scaling.
    """
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=[float, int]).columns.tolist()

    if verbose:
        print("FEATURE SCALING ANALYSIS:")
        print("=" * 50)
        print(f"Numerical features: {len(numerical_cols)}")

    # Stats descriptives
    feature_stats = X[numerical_cols].describe()
    if verbose:
        print("\n📊 Key Feature Statistics:")
        print(feature_stats.loc[["mean", "std", "min", "max"]].round(2))

    # Analyse des échelles
    means = X[numerical_cols].mean()
    stds = X[numerical_cols].std()
    scale_ratio = means.max() / means.min() if means.min() > 0 else float("inf")

    if verbose:
        print(f"\n🔍 Scale Analysis:")
        print(f"   Range of means: {means.min():.2f} to {means.max():.2f}")
        print(f"   Range of stds:  {stds.min():.2f} to {stds.max():.2f}")
        print(f"   Mean scale ratio: {scale_ratio:.1f}")

    recommendation = (
        "StandardScaler RECOMMENDED" if scale_ratio > 10 else "StandardScaler optional"
    )

    if verbose:
        print(
            f"   ✅ {recommendation} - significant scale differences detected"
            if scale_ratio > 10
            else f"   ℹ️  {recommendation} - scales are relatively similar"
        )
        print("\n💡 Logistic Regression is sensitive to feature scales.")
        print(
            "   StandardScaler will help with convergence and coefficient interpretation."
        )

    return {
        "feature_stats": feature_stats,
        "means": means,
        "stds": stds,
        "scale_ratio": scale_ratio,
        "recommendation": recommendation,
    }


def compare_models(models_dict, model_names=None, display_charts=True):
    """
    Compare multiple machine learning models with comprehensive analysis.

    Parameters:
    -----------
    models_dict : dict
                    Dictionary with model results in format:
                    {
                                    'model_name': {
                                                    'train': {'accuracy': float, 'precision': float, 'recall': float, 'f1_score': float},
                                                    'test': {'accuracy': float, 'precision': float, 'recall': float, 'f1_score': float}
                                    }
                    }
    model_names : list, optional
                    Custom names for models. If None, uses keys from models_dict
    display_charts : bool, default=True
                    Whether to display comparison charts

    Returns:
    --------
    dict
                    Dictionary containing:
                    - comparison_train: DataFrame with training metrics
                    - comparison_test: DataFrame with test metrics
                    - best_model: Name of best performing model (by test F1-score)
                    - overfitting_analysis: DataFrame with overfitting analysis

    Example:
    --------
    >>> results = {
    ...     'Logistic Regression': {
    ...         'train': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1_score': 0.84},
    ...         'test': {'accuracy': 0.82, 'precision': 0.81, 'recall': 0.82, 'f1_score': 0.81}
    ...     },
    ...     'Random Forest': {
    ...         'train': {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.88, 'f1_score': 0.87},
    ...         'test': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1_score': 0.84}
    ...     }
    ... }
    >>> comparison = compare_models(results)
    """

    if model_names is None:
        model_names = list(models_dict.keys())

    print("MODEL COMPARISON SUMMARY:")
    print("=" * 60)

    # Create comparison dataframes
    train_data = []
    test_data = []

    for model_name in model_names:
        if model_name in models_dict:
            # Training metrics
            train_metrics = models_dict[model_name]["train"]
            train_data.append(
                [
                    model_name,
                    train_metrics["accuracy"],
                    train_metrics["precision"],
                    train_metrics["recall"],
                    train_metrics["f1_score"],
                ]
            )

            # Test metrics
            test_metrics = models_dict[model_name]["test"]
            test_data.append(
                [
                    model_name,
                    test_metrics["accuracy"],
                    test_metrics["precision"],
                    test_metrics["recall"],
                    test_metrics["f1_score"],
                ]
            )

    # Create DataFrames
    columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    comparison_train = pd.DataFrame(train_data, columns=columns)
    comparison_test = pd.DataFrame(test_data, columns=columns)

    # Display comparison tables
    print("📊 TRAINING SET Performance:")
    print(comparison_train.round(4).to_string(index=False))

    print(f"\n📊 TEST SET Performance:")
    print(comparison_test.round(4).to_string(index=False))

    # Overfitting analysis
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    print(
        f"{'Model':<25} {'Train F1':<10} {'Test F1':<10} {'Difference':<12} {'Status'}"
    )
    print("-" * 70)

    overfitting_data = []
    for i, model_name in enumerate(model_names):
        if model_name in models_dict:
            # Use iloc to get numeric values directly
            train_f1_val = comparison_train.iloc[i]["F1-Score"]
            test_f1_val = comparison_test.iloc[i]["F1-Score"]

            # Ensure we have numeric values
            train_f1 = float(train_f1_val) if pd.notna(train_f1_val) else 0.0
            test_f1 = float(test_f1_val) if pd.notna(test_f1_val) else 0.0
            diff = train_f1 - test_f1

            if abs(diff) < 0.01:
                status = "✅ Excellent"
            elif diff > 0.10:
                status = "🚨 High overfitting"
            elif diff > 0.05:
                status = "⚠️ Moderate overfitting"
            elif diff > 0.02:
                status = "⚡ Minor overfitting"
            else:
                status = "ℹ️ Normal"

            print(
                f"{model_name:<25} {train_f1:<10.4f} {test_f1:<10.4f} {diff:<12.4f} {status}"
            )
            overfitting_data.append(
                {
                    "Model": model_name,
                    "Train_F1": train_f1,
                    "Test_F1": test_f1,
                    "Difference": diff,
                    "Status": status,
                }
            )

    overfitting_df = pd.DataFrame(overfitting_data)

    # Find best model
    best_model_idx = comparison_test["F1-Score"].idxmax()
    best_model = comparison_test.loc[best_model_idx, "Model"]
    best_f1 = comparison_test.loc[best_model_idx, "F1-Score"]

    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Test F1-Score: {best_f1:.4f}")

    # Baseline comparison (assumes first model is baseline)
    if len(comparison_test) > 1:
        baseline_f1_val = comparison_test.iloc[0]["F1-Score"]
        baseline_f1 = float(baseline_f1_val) if pd.notna(baseline_f1_val) else 0.0
        print(f"\n📈 Improvements over baseline (Test Set):")
        for idx in range(len(comparison_test)):
            if idx > 0:  # Skip baseline
                row = comparison_test.iloc[idx]
                improvement = float(row["F1-Score"]) - baseline_f1
                improvement_pct = (
                    (improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0
                )
                print(f"   {row['Model']}: +{improvement:.4f} ({improvement_pct:.1f}%)")

    # Generate visualization if requested
    if display_charts:
        _create_model_comparison_charts(comparison_train, comparison_test, model_names)

    print(f"\n🎯 Model comparison completed!")

    return {
        "comparison_train": comparison_train,
        "comparison_test": comparison_test,
        "best_model": best_model,
        "overfitting_analysis": overfitting_df,
    }


def get_classification_report_table(y_true, y_pred, model_name="Model"):
    """
    Create a nicely formatted classification report as a pandas DataFrame table.

    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    model_name: Name of the model for display (used in the DataFrame name attribute)

    Returns:
    pd.DataFrame: Formatted classification report table
    """
    # Convert classification report to DataFrame for better formatting
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Round values for better display
    report_df = report_df.round(4)

    # Remove the Support column and rename remaining columns for clarity
    report_df = report_df.drop("support", axis=1)
    report_df.columns = ["Precision", "Recall", "F1-Score"]

    # Create more descriptive index
    index_mapping = {
        "0": "Class 0 (Stayed)",
        "1": "Class 1 (Left)",
        "accuracy": "Accuracy",
        "macro avg": "Macro Average",
        "weighted avg": "Weighted Average",
    }

    report_df.index = pd.Index(
        [index_mapping.get(str(idx), str(idx)) for idx in report_df.index]
    )

    # Add model name as DataFrame name for reference
    report_df.name = f"Classification Report - {model_name}"

    return report_df


def _create_model_comparison_charts(comparison_train, comparison_test, model_names):
    """
    Create visualization charts for model comparison.

    Parameters:
    -----------
    comparison_train : DataFrame
                    Training metrics comparison
    comparison_test : DataFrame
                    Test metrics comparison
    model_names : list
                    List of model names
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.figure(figsize=(15, 10))

    # Create short names for better visualization
    model_names_short = [
        name.split("(")[0].strip() if "(" in name else name[:10] for name in model_names
    ]

    # Subplot 1: Training vs Test F1-Score
    plt.subplot(2, 3, 1)
    train_f1_scores = comparison_train["F1-Score"]
    test_f1_scores = comparison_test["F1-Score"]

    x = np.arange(len(model_names_short))
    width = 0.35

    bars1 = plt.bar(
        x - width / 2,
        train_f1_scores,
        width,
        label="Training",
        alpha=0.8,
        color="skyblue",
    )
    bars2 = plt.bar(
        x + width / 2, test_f1_scores, width, label="Test", alpha=0.8, color="orange"
    )

    plt.ylabel("F1-Score")
    plt.title("Training vs Test F1-Score")
    plt.xticks(x, model_names_short, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.005,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Subplot 2: All metrics comparison (Test Set)
    plt.subplot(2, 3, 2)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x_metrics = np.arange(len(metrics))
    width_metrics = 0.25

    colors = ["red", "orange", "green", "blue", "purple"][: len(model_names)]
    for i, model in enumerate(comparison_test["Model"]):
        values = [comparison_test.loc[i, metric] for metric in metrics]
        plt.bar(
            x_metrics + i * width_metrics,
            values,
            width_metrics,
            label=model_names_short[i],
            alpha=0.8,
            color=colors[i],
        )

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("All Metrics Comparison (Test Set)")
    plt.xticks(x_metrics + width_metrics, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Overfitting visualization
    plt.subplot(2, 3, 3)
    overfitting_diffs = [
        comparison_train.loc[i, "F1-Score"] - comparison_test.loc[i, "F1-Score"]
        for i in range(len(comparison_train))
    ]

    bars = plt.bar(
        model_names_short,
        overfitting_diffs,
        alpha=0.7,
        color=[
            "green" if diff < 0.02 else "orange" if diff < 0.05 else "red"
            for diff in overfitting_diffs
        ],
    )
    plt.ylabel("Training - Test F1 Difference")
    plt.title("Overfitting Analysis")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(
        y=0.02,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="Minor overfitting threshold",
    )
    plt.axhline(
        y=0.05,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Moderate overfitting threshold",
    )
    plt.legend(fontsize=8)

    # Add value labels
    for bar, diff in zip(bars, overfitting_diffs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{diff:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


def confusion_matrix_analysis(y_true, y_pred, model_name="Model"):
    # Confusion Matrix for Logistic Regression
    print(f"\n=== CONFUSION MATRIX ANALYSIS for {model_name} ===")
    print("=" * 50)

    # Calculate confusion matrix
    cm_lr = confusion_matrix(y_true, y_pred)
    class_names = sorted(y_true.unique())

    print(f"📊 Confusion Matrix:")
    print(f"   True labels (rows) vs Predicted labels (columns)")
    print(f"   Classes: {class_names}")
    print()
    print(cm_lr)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Calculate confusion matrix metrics for each class
    print(f"\n📈 Confusion Matrix Analysis:")
    for i, class_name in enumerate(class_names):
        tp = cm_lr[i, i]  # True positives
        fn = cm_lr[i, :].sum() - tp  # False negatives
        fp = cm_lr[:, i].sum() - tp  # False positives
        tn = cm_lr.sum() - tp - fn - fp  # True negatives

        # Class-specific metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision

        print(f"\n   Class {class_name}:")
        print(f"     True Positives:  {tp}")
        print(f"     False Positives: {fp}")
        print(f"     False Negatives: {fn}")
        print(f"     True Negatives:  {tn}")
        print(f"     Sensitivity (Recall): {sensitivity:.4f}")
        print(f"     Specificity:          {specificity:.4f}")
        print(f"     Precision:            {precision_class:.4f}")

    print(f"\n🎯 Confusion Matrix completed!")
