"""
Curve fitting models for workforce management FTE prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import curve_fit, minimize
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
import logging

logger = logging.getLogger(__name__)


class FTEPredictionModel:
    """
    Curve fitting models for predicting FTE requirements based on workforce metrics.
    Supports various functional forms and statistical validation.
    """

    def __init__(self):
        """Initialize the FTE prediction model."""
        self.models = {}
        self.model_performance = {}
        self.fitted_parameters = {}
        self.scaler = StandardScaler()

    @staticmethod
    def linear_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Linear model: y = ax + b"""
        return a * x + b

    @staticmethod
    def exponential_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Exponential model: y = a * exp(b * x) + c"""
        return a * np.exp(b * x) + c

    @staticmethod
    def power_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Power model: y = a * x^b + c"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return a * np.power(np.abs(x), b) + c

    @staticmethod
    def logarithmic_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Logarithmic model: y = a * log(b * x) + c"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return a * np.log(b * np.abs(x) + 1e-10) + c

    @staticmethod
    def polynomial_model(x: np.ndarray, *params) -> np.ndarray:
        """Polynomial model: y = a_n*x^n + ... + a_1*x + a_0"""
        result = np.zeros_like(x)
        for i, param in enumerate(params):
            result += param * (x ** i)
        return result

    @staticmethod
    def erlang_c_model(x: np.ndarray, intensity: float, service_rate: float) -> np.ndarray:
        """
        Erlang C model for call center staffing.
        x: offered traffic (calls * AHT / period)
        Returns: minimum agents needed for given service level
        """
        # Simplified Erlang C calculation
        # In practice, this would use full Erlang C formula
        return np.ceil(x / service_rate + intensity * np.sqrt(x / service_rate))

    def fit_model(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        model_type: str = "auto",
        degree: int = 2
    ) -> Dict:
        """
        Fit a curve to the data using specified model type.

        Args:
            x_data: Independent variable data
            y_data: Dependent variable (FTE) data
            model_type: Type of model ('linear', 'exponential', 'power', 'log', 'polynomial', 'auto')
            degree: Degree for polynomial models

        Returns:
            Dictionary with fitted model results
        """
        # Clean data
        mask = ~(np.isnan(x_data) | np.isnan(y_data) | np.isinf(x_data) | np.isinf(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]

        if len(x_clean) < 3:
            raise ValueError("Not enough valid data points for curve fitting")

        results = {
            'model_type': model_type,
            'fitted': False,
            'parameters': None,
            'r_squared': 0,
            'rmse': np.inf,
            'mae': np.inf,
            'aic': np.inf,
            'bic': np.inf
        }

        try:
            if model_type == "auto":
                # Try multiple models and select best
                best_model = self._auto_select_model(x_clean, y_clean, degree)
                return best_model

            elif model_type == "linear":
                popt, pcov = curve_fit(self.linear_model, x_clean, y_clean)
                y_pred = self.linear_model(x_clean, *popt)
                model_func = self.linear_model

            elif model_type == "exponential":
                # Provide initial guess
                p0 = [1, 0.1, np.mean(y_clean)]
                popt, pcov = curve_fit(self.exponential_model, x_clean, y_clean, p0=p0, maxfev=5000)
                y_pred = self.exponential_model(x_clean, *popt)
                model_func = self.exponential_model

            elif model_type == "power":
                p0 = [1, 1, 0]
                popt, pcov = curve_fit(self.power_model, x_clean, y_clean, p0=p0, maxfev=5000)
                y_pred = self.power_model(x_clean, *popt)
                model_func = self.power_model

            elif model_type == "logarithmic":
                p0 = [1, 1, 0]
                popt, pcov = curve_fit(self.logarithmic_model, x_clean, y_clean, p0=p0, maxfev=5000)
                y_pred = self.logarithmic_model(x_clean, *popt)
                model_func = self.logarithmic_model

            elif model_type == "polynomial":
                # Use polynomial fitting
                coeffs = np.polyfit(x_clean, y_clean, degree)
                y_pred = np.polyval(coeffs, x_clean)
                popt = coeffs[::-1]  # Reverse for our polynomial model format
                model_func = lambda x, *p: self.polynomial_model(x, *p)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Calculate performance metrics
            r_squared = r2_score(y_clean, y_pred)
            rmse = np.sqrt(mean_squared_error(y_clean, y_pred))
            mae = mean_absolute_error(y_clean, y_pred)

            # Calculate AIC and BIC
            n = len(y_clean)
            k = len(popt)
            mse = mean_squared_error(y_clean, y_pred)
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)

            results.update({
                'fitted': True,
                'parameters': popt,
                'parameter_covariance': pcov if 'pcov' in locals() else None,
                'r_squared': r_squared,
                'rmse': rmse,
                'mae': mae,
                'aic': aic,
                'bic': bic,
                'model_function': model_func,
                'x_data': x_clean,
                'y_data': y_clean,
                'y_predicted': y_pred
            })

            logger.info(f"Fitted {model_type} model: R² = {r_squared:.3f}, RMSE = {rmse:.3f}")

        except Exception as e:
            logger.warning(f"Failed to fit {model_type} model: {str(e)}")
            results['error'] = str(e)

        return results

    def _auto_select_model(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        max_degree: int = 3
    ) -> Dict:
        """
        Automatically select the best model from multiple options.

        Args:
            x_data: Independent variable data
            y_data: Dependent variable data
            max_degree: Maximum polynomial degree to test

        Returns:
            Best fitted model results
        """
        models_to_try = ["linear", "exponential", "power", "logarithmic"]

        # Add polynomial models
        for degree in range(2, max_degree + 1):
            models_to_try.append(f"polynomial_{degree}")

        best_model = None
        best_score = -np.inf

        for model_type in models_to_try:
            try:
                if model_type.startswith("polynomial_"):
                    degree = int(model_type.split("_")[1])
                    result = self.fit_model(x_data, y_data, "polynomial", degree)
                else:
                    result = self.fit_model(x_data, y_data, model_type)

                if result['fitted']:
                    # Use AIC for model selection (lower is better)
                    score = -result['aic']

                    if score > best_score:
                        best_score = score
                        best_model = result
                        best_model['selected_by'] = 'auto_aic'

            except Exception as e:
                logger.debug(f"Auto model selection failed for {model_type}: {str(e)}")
                continue

        if best_model is None:
            raise ValueError("No model could be fitted to the data")

        logger.info(f"Auto-selected model: {best_model['model_type']} (AIC: {best_model['aic']:.2f})")

        return best_model

    def predict(
        self,
        model_result: Dict,
        x_new: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Make predictions using a fitted model.

        Args:
            model_result: Result from fit_model
            x_new: New x values for prediction

        Returns:
            Predicted y values
        """
        if not model_result['fitted']:
            raise ValueError("Model has not been fitted")

        model_func = model_result['model_function']
        parameters = model_result['parameters']

        if isinstance(x_new, (int, float)):
            x_new = np.array([x_new])

        try:
            predictions = model_func(x_new, *parameters)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def calculate_prediction_intervals(
        self,
        model_result: Dict,
        x_new: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals for the model.

        Args:
            model_result: Result from fit_model
            x_new: New x values
            confidence_level: Confidence level for intervals

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        if not model_result['fitted']:
            raise ValueError("Model has not been fitted")

        predictions = self.predict(model_result, x_new)

        # Estimate prediction standard error from residuals
        residuals = model_result['y_data'] - model_result['y_predicted']
        residual_std = np.std(residuals)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)

        margin_of_error = z_score * residual_std

        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error

        return lower_bound, upper_bound

    def validate_model(
        self,
        model_result: Dict,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Validate model performance on test data.

        Args:
            model_result: Fitted model result
            x_test: Test x data
            y_test: Test y data

        Returns:
            Validation metrics
        """
        if not model_result['fitted']:
            raise ValueError("Model has not been fitted")

        # Make predictions
        y_pred = self.predict(model_result, x_test)

        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        max_error = np.max(np.abs(y_test - y_pred))

        validation_results = {
            'r_squared': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'max_error': max_error,
            'n_test_points': len(y_test)
        }

        logger.info(f"Model validation: R² = {r2:.3f}, RMSE = {rmse:.3f}, MAPE = {mape:.1f}%")

        return validation_results

    def fit_multiple_models(
        self,
        df: pd.DataFrame,
        target_column: str,
        predictor_columns: List[str]
    ) -> Dict[str, Dict]:
        """
        Fit models for multiple predictor-target relationships.

        Args:
            df: Input DataFrame
            target_column: Target variable (FTE)
            predictor_columns: List of predictor variables

        Returns:
            Dictionary of fitted models for each predictor
        """
        models = {}

        for predictor in predictor_columns:
            if predictor not in df.columns or target_column not in df.columns:
                continue

            # Prepare data
            data_subset = df[[predictor, target_column]].dropna()

            if len(data_subset) < 5:
                logger.warning(f"Not enough data for {predictor} -> {target_column}")
                continue

            x_data = data_subset[predictor].values
            y_data = data_subset[target_column].values

            try:
                # Fit model with auto-selection
                model_result = self.fit_model(x_data, y_data, model_type="auto")
                model_result['predictor'] = predictor
                model_result['target'] = target_column

                models[predictor] = model_result

                logger.info(f"Fitted model for {predictor} -> {target_column}: "
                           f"{model_result['model_type']} (R² = {model_result['r_squared']:.3f})")

            except Exception as e:
                logger.error(f"Failed to fit model for {predictor} -> {target_column}: {str(e)}")

        return models

    def generate_model_summary(self, model_result: Dict) -> Dict:
        """
        Generate a comprehensive summary of the fitted model.

        Args:
            model_result: Result from fit_model

        Returns:
            Model summary dictionary
        """
        if not model_result['fitted']:
            return {'error': 'Model not fitted'}

        summary = {
            'model_type': model_result['model_type'],
            'performance': {
                'r_squared': model_result['r_squared'],
                'rmse': model_result['rmse'],
                'mae': model_result['mae'],
                'aic': model_result['aic'],
                'bic': model_result['bic']
            },
            'parameters': {
                f'param_{i}': param for i, param in enumerate(model_result['parameters'])
            },
            'data_points': len(model_result['x_data']),
            'quality_assessment': self._assess_model_quality(model_result)
        }

        return summary

    def _assess_model_quality(self, model_result: Dict) -> str:
        """Assess the quality of the fitted model."""
        r2 = model_result['r_squared']

        if r2 >= 0.9:
            return "excellent"
        elif r2 >= 0.8:
            return "very_good"
        elif r2 >= 0.7:
            return "good"
        elif r2 >= 0.5:
            return "fair"
        else:
            return "poor"