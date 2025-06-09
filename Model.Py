"""
Weather-Based Commodity Price Forecasting Model
==============================================

A hybrid machine learning model that forecasts commodity prices based on weather predictions.
Supports multiple commodities including agricultural products and energy commodities.

Author: Jeremiah D. Rhoads in tandem with AI Assistant 
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Deep learning (optional, requires tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

class WeatherCommodityForecaster:
    """
    Hybrid model for forecasting commodity prices based on weather predictions.
    
    Combines multiple approaches:
    - Random Forest and Gradient Boosting for non-linear relationships
    - Ridge/ElasticNet for linear relationships
    - ARIMA for time series patterns
    - LSTM for deep learning (optional)
    """
    
    def __init__(self, commodity_type='agricultural'):
        self.commodity_type = commodity_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.ensemble_weights = {}
        self.is_fitted = False
        
        # Weather features relevant to different commodity types
        self.weather_features = {
            'agricultural': [
                'temperature_avg', 'temperature_min', 'temperature_max',
                'precipitation', 'humidity', 'wind_speed',
                'growing_degree_days', 'frost_days', 'drought_index'
            ],
            'energy': [
                'temperature_avg', 'temperature_min', 'temperature_max',
                'wind_speed', 'solar_radiation', 'heating_degree_days',
                'cooling_degree_days'
            ]
        }
        
    def generate_synthetic_data(self, n_samples=1000, start_date='2020-01-01'):
        """
        Generate synthetic weather and commodity price data for demonstration.
        In production, replace with real data sources.
        """
        np.random.seed(42)
        
        # Generate date range
        dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
        
        # Generate weather features
        data = {
            'date': dates,
            'temperature_avg': 15 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 3, n_samples),
            'temperature_min': 10 + 8 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 2, n_samples),
            'temperature_max': 20 + 12 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 4, n_samples),
            'precipitation': np.maximum(0, np.random.exponential(2, n_samples)),
            'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 10, n_samples),
            'wind_speed': np.maximum(0, np.random.gamma(2, 2, n_samples)),
            'solar_radiation': 200 + 100 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 20, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived weather features
        df['growing_degree_days'] = np.maximum(0, (df['temperature_max'] + df['temperature_min']) / 2 - 10)
        df['frost_days'] = (df['temperature_min'] < 0).astype(int)
        df['heating_degree_days'] = np.maximum(0, 18 - df['temperature_avg'])
        df['cooling_degree_days'] = np.maximum(0, df['temperature_avg'] - 18)
        df['drought_index'] = self._calculate_drought_index(df['precipitation'].values)
        
        # Generate commodity prices with weather relationships
        if self.commodity_type == 'agricultural':
            # Wheat price example - affected by temperature, precipitation, and growing conditions
            base_price = 200
            price_trend = 0.02 * np.arange(n_samples)  # Long-term trend
            seasonal_effect = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365 + np.pi/4)
            
            # Weather effects
            temp_effect = -2 * (df['temperature_avg'] - 20) ** 2 / 100  # Optimal around 20°C
            precip_effect = 5 * np.log1p(df['precipitation']) - 10 * (df['precipitation'] > 10)  # Good rain, but not floods
            drought_effect = -30 * df['drought_index']
            frost_effect = -15 * df['frost_days']
            
            commodity_price = (base_price + price_trend + seasonal_effect + 
                             temp_effect + precip_effect + drought_effect + frost_effect +
                             np.random.normal(0, 10, n_samples))
            
        else:  # energy
            # Natural gas price example - affected by temperature (heating/cooling demand)
            base_price = 3.5
            price_trend = 0.001 * np.arange(n_samples)
            seasonal_effect = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 365 + np.pi)
            
            # Weather effects
            heating_effect = 0.05 * df['heating_degree_days']
            cooling_effect = 0.03 * df['cooling_degree_days']
            wind_effect = -0.02 * df['wind_speed']  # Wind power competition
            
            commodity_price = (base_price + price_trend + seasonal_effect +
                             heating_effect + cooling_effect + wind_effect +
                             np.random.normal(0, 0.3, n_samples))
        
        df['commodity_price'] = np.maximum(0, commodity_price)
        
        return df
    
    def _calculate_drought_index(self, precipitation, window=30):
        """Calculate a simple drought index based on precipitation deficit."""
        df_precip = pd.DataFrame({'precip': precipitation})
        rolling_precip = df_precip['precip'].rolling(window=window, min_periods=1).mean()
        long_term_avg = df_precip['precip'].expanding().mean()
        drought_index = np.maximum(0, (long_term_avg - rolling_precip) / long_term_avg)
        return drought_index.fillna(0).values
    
    def create_features(self, df):
        """Create additional features for modeling."""
        df = df.copy()
        
        # Lag features
        for col in self.weather_features[self.commodity_type]:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag7'] = df[col].shift(7)
                df[f'{col}_ma7'] = df[col].rolling(7).mean()
                df[f'{col}_ma30'] = df[col].rolling(30).mean()
        
        # Price lag features
        if 'commodity_price' in df.columns:
            df['price_lag1'] = df['commodity_price'].shift(1)
            df['price_lag7'] = df['commodity_price'].shift(7)
            df['price_ma7'] = df['commodity_price'].rolling(7).mean()
            df['price_ma30'] = df['commodity_price'].rolling(30).mean()
            df['price_volatility'] = df['commodity_price'].rolling(30).std()
        
        # Time features
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year
        
        # Seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def prepare_data(self, df, target_col='commodity_price'):
        """Prepare data for modeling."""
        df_features = self.create_features(df)
        
        # Select feature columns (exclude date and target)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', target_col] and not col.endswith('_target')]
        
        X = df_features[feature_cols].dropna()
        y = df_features.loc[X.index, target_col]
        
        return X, y, feature_cols
    
    def fit(self, df, target_col='commodity_price', test_size=0.2):
        """Fit the hybrid model."""
        print("Preparing data...")
        X, y, self.feature_cols = self.prepare_data(df, target_col)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        print("Training models...")
        
        # 1. Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)
        
        # 2. Gradient Boosting
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.models['gb'].fit(X_train, y_train)
        
        # 3. Ridge Regression
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(X_train_scaled, y_train)
        
        # 4. Elastic Net
        self.models['elastic'] = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        self.models['elastic'].fit(X_train_scaled, y_train)
        
        # 5. ARIMA model for time series component
        try:
            # Use price series for ARIMA
            price_series = df.set_index('date')[target_col].dropna()
            self.models['arima'] = ARIMA(price_series, order=(2, 1, 2))
            self.models['arima'] = self.models['arima'].fit()
        except Exception as e:
            print(f"ARIMA model failed: {e}")
            self.models['arima'] = None
        
        # 6. LSTM model (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            try:
                self.models['lstm'] = self._build_lstm_model(X_train_scaled, y_train)
            except Exception as e:
                print(f"LSTM model failed: {e}")
                self.models['lstm'] = None
        
        # Calculate ensemble weights based on cross-validation performance
        self._calculate_ensemble_weights(X_train, X_train_scaled, y_train, tscv)
        
        # Store test data for evaluation
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        # Feature importance
        if 'rf' in self.models:
            self.feature_importance['rf'] = dict(zip(
                self.feature_cols, self.models['rf'].feature_importances_
            ))
        
        self.is_fitted = True
        print("Model training completed!")
        
    def _build_lstm_model(self, X_train, y_train, sequence_length=30):
        """Build and train LSTM model."""
        # Reshape data for LSTM
        X_lstm = []
        y_lstm = []
        
        for i in range(sequence_length, len(X_train)):
            X_lstm.append(X_train[i-sequence_length:i])
            y_lstm.append(y_train.iloc[i])
        
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0)
        
        return model
    
    def _calculate_ensemble_weights(self, X_train, X_train_scaled, y_train, tscv):
        """Calculate ensemble weights based on cross-validation performance."""
        cv_scores = {}
        
        for name, model in self.models.items():
            if model is None:
                continue
                
            if name in ['ridge', 'elastic']:
                X_cv = X_train_scaled
            elif name == 'arima':
                continue  # Skip ARIMA for CV (time series specific)
            elif name == 'lstm':
                continue  # Skip LSTM for CV (complex reshaping needed)
            else:
                X_cv = X_train
            
            try:
                scores = cross_val_score(model, X_cv, y_train, cv=tscv, scoring='r2')
                cv_scores[name] = np.mean(scores)
            except Exception as e:
                print(f"CV failed for {name}: {e}")
                cv_scores[name] = 0
        
        # Normalize weights
        total_score = sum(max(0, score) for score in cv_scores.values())
        if total_score > 0:
            self.ensemble_weights = {
                name: max(0, score) / total_score 
                for name, score in cv_scores.items()
            }
        else:
            # Equal weights if all models perform poorly
            self.ensemble_weights = {
                name: 1/len(cv_scores) for name in cv_scores.keys()
            }
        
        print("Ensemble weights:", self.ensemble_weights)
    
    def predict(self, X):
        """Make predictions using the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = {}
        
        # Scale features
        if hasattr(self.scalers['standard'], 'transform'):
            X_scaled = self.scalers['standard'].transform(X)
        else:
            X_scaled = X
        
        # Get predictions from each model
        for name, model in self.models.items():
            if model is None:
                continue
                
            try:
                if name in ['ridge', 'elastic']:
                    predictions[name] = model.predict(X_scaled)
                elif name in ['rf', 'gb']:
                    predictions[name] = model.predict(X)
                elif name == 'arima':
                    # For ARIMA, we'd need the time series context
                    # This is simplified - in practice, you'd need proper time series forecasting
                    predictions[name] = np.full(len(X), model.fittedvalues.iloc[-1])
                elif name == 'lstm' and TENSORFLOW_AVAILABLE:
                    # Simplified LSTM prediction - needs proper sequence handling
                    predictions[name] = model.predict(X_scaled[:, np.newaxis, :])[:, 0]
            except Exception as e:
                print(f"Prediction failed for {name}: {e}")
                continue
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.zeros(len(X))
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = self.ensemble_weights.get(name, 0)
                ensemble_pred += weight * pred
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            return ensemble_pred
        else:
            raise ValueError("No models available for prediction")
    
    def evaluate(self):
        """Evaluate model performance on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(self.X_test)
        
        metrics = {
            'MAE': mean_absolute_error(self.y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'R2': r2_score(self.y_test, y_pred),
            'MAPE': np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        }
        
        return metrics
    
    def plot_results(self, df=None, n_days_forecast=30):
        """Plot model results and forecasts."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted
        y_pred = self.predict(self.X_test)
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--')
        axes[0, 0].set_xlabel('Actual Price')
        axes[0, 0].set_ylabel('Predicted Price')
        axes[0, 0].set_title('Actual vs Predicted Prices')
        
        # 2. Time series plot
        test_dates = df.iloc[-len(self.y_test):]['date'] if df is not None else range(len(self.y_test))
        axes[0, 1].plot(test_dates, self.y_test.values, label='Actual', linewidth=2)
        axes[0, 1].plot(test_dates, y_pred, label='Predicted', linewidth=2)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].set_title('Price Time Series')
        axes[0, 1].legend()
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Feature importance
        if self.feature_importance:
            importance_df = pd.DataFrame.from_dict(
                self.feature_importance['rf'], orient='index', columns=['Importance']
            ).sort_values('Importance', ascending=True).tail(10)
            
            axes[1, 0].barh(importance_df.index, importance_df['Importance'])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 10 Feature Importances (Random Forest)')
        
        # 4. Residuals
        residuals = self.y_test.values - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Price')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
        
        # Print evaluation metrics
        metrics = self.evaluate()
        print("\nModel Performance Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    def forecast_future(self, df, n_days=30):
        """
        Generate future forecasts based on weather predictions.
        Note: This requires future weather data as input.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get the last known data point
        last_date = df['date'].max()
        last_price = df['commodity_price'].iloc[-1]
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1), 
            periods=n_days, 
            freq='D'
        )
        
        # For demonstration, generate synthetic future weather
        # In practice, this would come from weather forecast APIs
        np.random.seed(42)
        future_weather = {
            'date': future_dates,
            'temperature_avg': 15 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 2, n_days),
            'temperature_min': 10 + 8 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 1.5, n_days),
            'temperature_max': 20 + 12 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 2.5, n_days),
            'precipitation': np.maximum(0, np.random.exponential(2, n_days)),
            'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 8, n_days),
            'wind_speed': np.maximum(0, np.random.gamma(2, 2, n_days)),
            'solar_radiation': 200 + 100 * np.sin(2 * np.pi * np.arange(n_days) / 365) + np.random.normal(0, 15, n_days),
        }
        
        future_df = pd.DataFrame(future_weather)
        
        # Calculate derived features
        future_df['growing_degree_days'] = np.maximum(0, (future_df['temperature_max'] + future_df['temperature_min']) / 2 - 10)
        future_df['frost_days'] = (future_df['temperature_min'] < 0).astype(int)
        future_df['heating_degree_days'] = np.maximum(0, 18 - future_df['temperature_avg'])
        future_df['cooling_degree_days'] = np.maximum(0, future_df['temperature_avg'] - 18)
        future_df['drought_index'] = self._calculate_drought_index(future_df['precipitation'].values)
        
        # Add placeholder commodity price (will be predicted)
        future_df['commodity_price'] = last_price
        
        # Combine with historical data for feature creation
        combined_df = pd.concat([df, future_df], ignore_index=True)
        combined_features = self.create_features(combined_df)
        
        # Extract future features
        future_features = combined_features.iloc[-n_days:][self.feature_cols].fillna(method='ffill').fillna(0)
        
        # Make predictions
        future_predictions = self.predict(future_features)
        
        # Create forecast DataFrame
        forecast_df = future_df.copy()
        forecast_df['predicted_price'] = future_predictions
        
        return forecast_df[['date', 'predicted_price']]

def main():
    """Main function to demonstrate the model."""
    print("Weather-Based Commodity Price Forecasting Model")
    print("=" * 50)
    
    # Initialize model for agricultural commodity (wheat)
    model = WeatherCommodityForecaster(commodity_type='agricultural')
    
    # Generate synthetic data
    print("Generating synthetic data...")
    df = model.generate_synthetic_data(n_samples=1000)
    
    # Fit the model
    print("Training the hybrid model...")
    model.fit(df)
    
    # Evaluate performance
    print("\nEvaluating model performance...")
    model.plot_results(df)
    
    # Generate future forecasts
    print("\nGenerating 30-day forecast...")
    forecast = model.forecast_future(df, n_days=30)
    print(forecast.head(10))
    
    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'].tail(60), df['commodity_price'].tail(60), 
             label='Historical', linewidth=2)
    plt.plot(forecast['date'], forecast['predicted_price'], 
             label='Forecast', linewidth=2, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Commodity Price')
    plt.title('Commodity Price Forecast')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("\nModel demonstration completed!")

if __name__ == "__main__":
    main()
