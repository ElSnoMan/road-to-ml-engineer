# Sequences, Time Series and Prediction

## Week 1 - Intro to Time Series

### What is a Time Series? What are some terms to be familiar with?
1. Trend
2. Seasonality
3. Noise
4. Autocorrelation

### Forecasting
* Naive Forecasting
* Training and Measuring Forecasting Models
    * Fixed Partitioning
    * Roll-Forward Partitioning

### Metrics for Evaluation
* Errors `forecasts - actual`
* MSE    `np.square(errors).mean()`
* RMSE   `np.sqrt(MSE)`
* MAE    `np.abs(errors).mean()`
* MAPE   `np.abs(errors/x_valid).mean()`

### Techniques to improve performance
* Moving Average
* Differencing
* Windows (trailing vs centered)
