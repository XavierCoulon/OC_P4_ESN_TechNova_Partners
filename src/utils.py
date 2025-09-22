"""
Encoding Utilities for Feature Engineering

This module contains encoding functions for categorical and binary variables
used in machine learning preprocessing.

Created for: OC Project 4 - ESN TechNova Partners
Author: Data Science Team
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

warnings.filterwarnings("ignore")
import warnings

warnings.filterwarnings("ignore")


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

    return corr_matrix, highly_correlated


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


def remove_highly_correlated_features(df, correlation_threshold=0.9):
    """
    Remove highly correlated features to avoid multicollinearity.

    Parameters:
    -----------
    df : DataFrame
        Input dataset with numerical features
    correlation_threshold : float, default=0.9
        Threshold for removing features

    Returns:
    --------
    tuple
        (df_reduced, removed_features)
        - df_reduced: DataFrame with highly correlated features removed
        - removed_features: List of removed feature names

    Example:
    --------
    >>> df_clean, removed = remove_highly_correlated_features(df, threshold=0.85)
    >>> print(f"Removed {len(removed)} highly correlated features")
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Find highly correlated feature pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Identify features to remove
    to_remove = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > correlation_threshold)
    ]

    # Remove features
    df_reduced = df.drop(columns=to_remove)

    return df_reduced, to_remove


# Version info
__version__ = "1.0.0"
__author__ = "Data Science Team"
__description__ = "Encoding utilities for categorical feature preprocessing"


def validate_data_quality(X, y, feature_name="X", target_name="y"):
    """
    Comprehensive data quality validation for ML datasets.

    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series or DataFrame
        Target variable
    feature_name : str, default="X"
        Name for feature matrix in output
    target_name : str, default="y"
        Name for target variable in output

    Returns:
    --------
    dict
        Dictionary with validation results and recommendations

    Example:
    --------
    >>> validation_results = validate_data_quality(X, y)
    >>> if validation_results['passed']:
    ...     print("Data quality validation passed!")
    """
    validation_results = {"passed": True, "issues": [], "warnings": [], "summary": {}}

    # Check for missing values
    X_missing = X.isnull().sum().sum()
    y_missing = y.isnull().sum() if hasattr(y, "isnull") else 0

    validation_results["summary"]["missing_values"] = {
        feature_name: X_missing,
        target_name: y_missing,
    }

    if X_missing > 0 or y_missing > 0:
        validation_results["passed"] = False
        validation_results["issues"].append(
            f"Missing values found: {feature_name}={X_missing}, {target_name}={y_missing}"
        )

    # Check data types
    numerical_features = len(X.select_dtypes(include=[np.number]).columns)
    categorical_features = len(X.select_dtypes(exclude=[np.number]).columns)

    validation_results["summary"]["feature_types"] = {
        "numerical": numerical_features,
        "categorical": categorical_features,
        "total": X.shape[1],
    }

    # Check for infinite values
    if np.isinf(X.select_dtypes(include=[np.number])).any().any():
        validation_results["passed"] = False
        validation_results["issues"].append("Infinite values found in feature matrix")

    # Check feature variability
    zero_variance_features = []
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].var() == 0:
            zero_variance_features.append(col)

    if zero_variance_features:
        validation_results["passed"] = False
        validation_results["issues"].append(
            f"Zero variance features: {zero_variance_features}"
        )

    # Target variable analysis
    if hasattr(y, "nunique"):
        target_unique = y.nunique()
        target_distribution = y.value_counts(normalize=True).to_dict()
    else:
        target_unique = len(np.unique(y))
        unique_vals, counts = np.unique(y, return_counts=True)
        target_distribution = {
            val: count / len(y) for val, count in zip(unique_vals, counts)
        }

    validation_results["summary"]["target_analysis"] = {
        "unique_values": target_unique,
        "distribution": target_distribution,
    }

    # Check for class imbalance (for classification)
    if target_unique <= 10:  # Likely classification
        min_class_proportion = min(target_distribution.values())
        if min_class_proportion < 0.05:  # Less than 5%
            validation_results["warnings"].append(
                f"Severe class imbalance detected: minimum class = {min_class_proportion:.1%}"
            )
        elif min_class_proportion < 0.1:  # Less than 10%
            validation_results["warnings"].append(
                f"Class imbalance detected: minimum class = {min_class_proportion:.1%}"
            )

    return validation_results


def print_feature_engineering_summary(
    feature_types, encoding_info=None, removed_features=None
):
    """
    Print a comprehensive summary of feature engineering steps.

    Parameters:
    -----------
    feature_types : dict
        Output from identify_feature_types()
    encoding_info : dict, optional
        Encoding information from encoding functions
    removed_features : list, optional
        List of features removed during preprocessing

    Example:
    --------
    >>> print_feature_engineering_summary(feature_types, encoding_info, removed_features)
    """
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    # Feature type summary
    print("\n📊 FEATURE TYPE ANALYSIS:")
    total_features = sum(len(features) for features in feature_types.values())
    for feature_type, features in feature_types.items():
        if features:
            print(
                f"  • {feature_type.replace('_', ' ').title()}: {len(features)} features"
            )
            if len(features) <= 5:
                print(f"    - {', '.join(features)}")
            else:
                print(f"    - {', '.join(features[:3])} ... and {len(features)-3} more")

    print(f"\n  📈 Total features analyzed: {total_features}")

    # Encoding summary
    if encoding_info:
        print("\n🔧 ENCODING APPLIED:")
        for feature, info in encoding_info.items():
            method = info["method"]
            if method == "binary":
                print(f"  • {feature}: Binary encoding → {info['mapping']}")
            elif method == "label_encoder":
                print(f"  • {feature}: Label encoding → {len(info['classes'])} classes")
            elif method == "one_hot":
                print(
                    f"  • {feature}: One-hot encoding → {len(info['new_columns'])} columns"
                )
            elif method == "ordinal":
                print(
                    f"  • {feature}: Ordinal encoding → {len(info['mapping'])} levels"
                )

    # Removed features summary
    if removed_features:
        print(f"\n🗑️ FEATURES REMOVED:")
        print(f"  • Highly correlated features: {len(removed_features)}")
        if len(removed_features) <= 5:
            print(f"    - {', '.join(removed_features)}")
        else:
            print(
                f"    - {', '.join(removed_features[:3])} ... and {len(removed_features)-3} more"
            )

    print("\n" + "=" * 60)


# Constants and configuration
DEFAULT_TARGET_VARIABLE = "a_quitte_l_entreprise"
DEFAULT_CORRELATION_THRESHOLD = 0.9
DEFAULT_VARIANCE_THRESHOLD = 0.0

# Version info
__version__ = "1.0.0"
__author__ = "Data Science Team"
__description__ = "Feature engineering utilities for employee turnover prediction"
