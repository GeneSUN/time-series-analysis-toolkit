from dataclasses import MISSING, dataclass, field
from typing import List, Optional, Iterable
import pandas as pd
import numpy as np

import copy
import warnings
from typing import Dict, List, Union
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.metrics import calculate_metrics

@dataclass
class FeatureConfig:
    """FeatureConfig describe data, not how to transform it."""

    continuous_features: List
    date: List = None
    target: str = None

    categorical_features: List = None
    boolean_features: List[str] = None 
    
    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": (
                        "These features are not the target but may influence it. "
                        "Each must already be declared in either 'continuous_features' or 'categorical_features'."
                    ) } )
    original_target = None

    def __post_init__(self):
        """Validate feature configuration after initialization."""
        
        all_features = ( self.categorical_features + self.continuous_features + self.boolean_features )

        # --- 1. Ensure target/date are not included in features
        if self.target in all_features:
            raise ValueError( f"Target column `{self.target}` must not appear in categorical, continuous, or boolean features." )

        if self.date in all_features:
            raise ValueError( f"Date column `{self.date}` must not appear in categorical, continuous, or boolean features." )

        # --- 2. Validate exogenous features
        extra_exog = set(self.exogenous_features) - set(all_features)
        if extra_exog:
            raise ValueError(
                f"Exogenous features not present in declared features: {sorted(extra_exog)}"
            )

        # --- 3. Ensure no overlaps between feature types
        overlaps = (
                        set(self.continuous_features) & set(self.categorical_features)
                    ) | (
                        set(self.continuous_features) & set(self.boolean_features)
                    ) | (
                        set(self.categorical_features) & set(self.boolean_features)
                    )

        if overlaps:
            raise ValueError(
                f"Features cannot belong to multiple types. Overlapping features: {sorted(overlaps)}"
            )
            # --- 6. Default original_target if missing
        if self.original_target is None:
            self.original_target = self.target

    def get_X_y(
        self, 
        df: pd.DataFrame, 
        categorical: bool = False, 
        exogenous: bool = False
    ):
        """
        Slice a dataframe into X, y, y_orig using the configured feature groups.

        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe containing features and target(s).
        categorical : bool
            If True, include categorical + boolean features in X (in addition to continuous).
        exogenous : bool
            If True, keep `exogenous_features` in X; otherwise drop them.

        Returns
        -------
        (X, y, y_orig)
            X : pd.DataFrame of selected features
            y : pd.DataFrame with the target column (or None if missing)
            y_orig : pd.DataFrame with the original target column (or None if missing)
        """
        # 1) Build list of expected features
        feature_list = list(self.continuous_features)  # start with continuous
        if categorical:
            feature_list += self.categorical_features + self.boolean_features
        if not exogenous:
            feature_list = [f for f in feature_list if f not in set(self.exogenous_features)]

        # 2) Validate all required features exist
        missing = set(feature_list) - set(df.columns)
        if missing:
            warnings.warn(f"Missing features dropped: { missing }")
        feature_list = [f for f in feature_list if f in df.columns]

        # 3) Slice
        X = df.loc[:, feature_list].copy()
        y = df[[self.target]].copy() if self.target in df.columns else None
        y_orig = df[[self.original_target]].copy() if self.original_target in df.columns else None

        return X, y, y_orig



@dataclass
class ModelConfig:

    # Models
    model: BaseEstimator = field(
        default=MISSING, metadata={"help": "Sci-kit Learn Compatible model instance"}
    )

    name: str = field(
        default=None,
        metadata={
            "help": "Name or identifier for the model. If left None, will use the string representation of the model"
        },
    )

    # Continuous features preprocessing
    apply_normalization: bool = field(
        default=False,
        metadata={"help": "Whether to normalize continuous features before fitting"}
    )

    scaler_model: Optional[BaseEstimator] = field(
        default=None,
        metadata={"help": "Scaler object to be used for continuous features (e.g., StandardScaler, MinMaxScaler)"}
    )

    # Categorical features preprocessing
    apply_categorical_encoding: bool = field(
        default=False,
        metadata={"help": "Whether to encode categorical features before fitting"}
    )

    categorical_encoder_model: Optional[BaseEstimator] = field(
        default=None,
        metadata={"help": "Encoder object for categorical features (e.g., OneHotEncoder, OrdinalEncoder)"}
    )

    def __post_init__(self):

        if self.model is MISSING:
            raise ValueError("`model` must be provided.")

        if self.apply_normalization and self.scaler_model is None:
            warnings.warn(
                            "`apply_normalization=True` but no scaler_model was provided. "
                            "Defaulting to StandardScaler()."
                        )
            self.scaler_model = StandardScaler()

        if self.apply_categorical_encoding and self.categorical_encoder_model is None:
            raise ValueError("`apply_categorical_encoding=True` but no categorical_encoder_model provided.")

        if self.name is None:
            self.name = str(self.model)

        # Type safety checks
        if self.scaler_model is not None and not isinstance(self.scaler_model, BaseEstimator):
            raise TypeError("scaler_model must be a scikit-learn estimator.")
        if self.categorical_encoder_model is not None and not isinstance(self.categorical_encoder_model, BaseEstimator):
            raise TypeError("categorical_encoder_model must be a scikit-learn estimator.")


    def clone(self):
      # creates a fresh copy of the estimator with the same hyperparameters but without any fitted state.
        self.model = clone(self.model)
        return self

class ModelWorkflow:
    def __init__(
        self,
        model_config,
        feature_config,
        target_transformer: object = None,
    ) -> None:
        """
        Execution workflow wrapping a scikit-learn style estimator
        with preprocessing and optional target transformation.
        """
        self.model_config = model_config
        self.feature_config = feature_config
        self.target_transformer = target_transformer

        # clone model
        self._model = clone(self.model_config.model)

        # runtime state
        self._scaler = None
        self._cat_encoder = None
        self._encoded_categorical_features = None

        # keep track of last datasets for evaluation
        self._last_train_df = None
        self._last_test_df = None
        self._last_y_pred = None

        # initialize scaler if needed
        if self.model_config.apply_normalization:
            self._scaler = clone(self.model_config.scaler_model)

        # initialize categorical encoder if needed
        if self.model_config.apply_categorical_encoding:
            if self.model_config.categorical_encoder_model is None:
                warnings.warn(
                    "`apply_categorical_encoding=True` but no encoder set. "
                    "Defaulting to OneHotEncoder(handle_unknown='ignore')."
                )
                self._cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                self._cat_encoder = clone(self.model_config.categorical_encoder_model)
            self._encoded_categorical_features = copy.deepcopy(
                self.feature_config.categorical_features
            )

    # -------------------------------
    # Categorical encoding
    # -------------------------------
    def _encode_categorical_fit(self, X: pd.DataFrame) -> None:
        if self._cat_encoder is not None and self._encoded_categorical_features:
            self._cat_encoder.fit(X[self._encoded_categorical_features])

    def _encode_categorical_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._cat_encoder is None or not self._encoded_categorical_features:
            return X
        encoded = self._cat_encoder.transform(X[self._encoded_categorical_features])
        encoded_cols = self._cat_encoder.get_feature_names_out(self._encoded_categorical_features)
        X_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)

        # drop original categorical cols, add encoded ones
        X = X.drop(columns=self._encoded_categorical_features)
        return pd.concat([X, X_encoded], axis=1)

    # -------------------------------
    # Scaling
    # -------------------------------
    def _scale_features_fit(self, X: pd.DataFrame) -> None:
        if self._scaler is not None and self.feature_config.continuous_features:
            self._scaler.fit(X[self.feature_config.continuous_features])

    def _scale_features_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or not self.feature_config.continuous_features:
            return X
        scaled = self._scaler.transform(X[self.feature_config.continuous_features])
        X_scaled = pd.DataFrame(scaled, columns=self.feature_config.continuous_features, index=X.index)
        X = X.copy()
        X[self.feature_config.continuous_features] = X_scaled
        return X

    # -------------------------------
    # Target transformation
    # -------------------------------
    def _transform_target_fit(self, y: pd.Series) -> np.ndarray:
        if self.target_transformer is None:
            return y.values
        return self.target_transformer.fit_transform(y.values.reshape(-1, 1)).ravel()

    def _inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        if self.target_transformer is None:
            return y_pred
        return self.target_transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    # -------------------------------
    # Public API
    # -------------------------------
    def fit(self, df: pd.DataFrame, categorical=True, exogenous=True):
        """Full workflow: extract features & target, preprocess, fit model."""
        X, y, _ = self.feature_config.get_X_y(df, categorical=categorical, exogenous=exogenous)
        self._last_train_df = df.copy()
        return self.fit_Xy(X, y)

    def fit_Xy(self, X: pd.DataFrame, y: pd.Series):
        """Lower-level fit if you already have X and y."""
        # categorical encoding
        self._encode_categorical_fit(X)
        X_proc = self._encode_categorical_transform(X)

        # scaling
        self._scale_features_fit(X_proc)
        X_proc = self._scale_features_transform(X_proc)

        # target transform
        y_proc = self._transform_target_fit(y)

        # fit model
        self._model.fit(X_proc, y_proc)
        return self

    def predict(self, df: pd.DataFrame, categorical=True, exogenous=True) -> np.ndarray:
        """Predict from raw dataframe and store test set + predictions."""
        X, _, _ = self.feature_config.get_X_y(df, categorical=categorical, exogenous=exogenous)
        y_pred = self.predict_X(X)
        self._last_test_df = df.copy()
        self._last_y_pred = pd.Series(y_pred.ravel(), index=df.index)
        return y_pred

    def predict_X(self, X: pd.DataFrame) -> np.ndarray:
        """Predict from pre-extracted features."""
        X_proc = self._encode_categorical_transform(X)
        X_proc = self._scale_features_transform(X_proc)
        y_pred = self._model.predict(X_proc)
        return self._inverse_transform_target(y_pred)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances or coefficients if available."""
        if hasattr(self._model, "feature_importances_"):
            values = self._model.feature_importances_
            features = self._model.feature_names_in_
        elif hasattr(self._model, "coef_"):
            values = self._model.coef_
            features = self._model.feature_names_in_
        else:
            raise AttributeError("Model has no feature_importances_ or coef_")

        return pd.DataFrame({"feature": features, "importance": values})

    def evaluate(self, decimals: int = 3) -> Dict[str, float]:
        """
        Evaluate using the last train/test/predict calls.
        Requires fit() and predict() to have been called first.
        """
        if self._last_train_df is None or self._last_test_df is None or self._last_y_pred is None:
            raise ValueError("You must call fit() and predict() before evaluate().")

        # Extract y_true and y_train
        _, y_test, _ = self.feature_config.get_X_y(self._last_test_df)
        _, y_train, _ = self.feature_config.get_X_y(self._last_train_df)

        return calculate_metrics(
            y=y_test,
            y_pred=self._last_y_pred,
            name=self.model_config.name,
            y_train=y_train,
            decimals=decimals,
        )
