# Weather-Driven-Commodity-Pricing-Model
A hybrid ML model that forecasts commodity prices based on weather predictions. Supports multiple commodities, including agricultural products and energy commodities.

The model combines multiple modeling approaches: 
- Random Forest & Gradient Boosting (non-linear relationships)
- Ridge & Elastic Net Regression (linear relationships)
- ARIMA (time series analysis)
- LSTM neural networks (deep learning aspect)

Regarding the weather, features are tailored for different commodity types:
- Agricultural: temp, precipitation, humidity, growing degree days, frost days, drought index)
- Energy: temperature, wind speed, solar radiation, heating/cooling degree days

