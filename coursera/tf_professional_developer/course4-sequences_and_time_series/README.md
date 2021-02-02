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
* Be aware of Sequence Bias!

## Week 2 - Neural Networks on Time Series Data

### Windowed Datasets
Instead of loading the entire Time Series, we can capture a "window" of time to train and predict on.

```python
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
```

### Train, Compile, Evaluate, Predict
* How to split Time Series data into train, validation and test sets
* Used MSE as our loss function
* Tuning parameters in our optimizer (in this case, SGD aka Scholastic Gradient Descent)
* Inspect the weights and biases in a layer after training
* Compilation and Training were basically the same as previous models
* Chart the predictions against the actual values

### Forecasting with DNNs
Most of this was the same as models from previous courses, but the biggest takeaway for me was the `LearningRateScheduler` callback which helped me find the optimal learning reate for my SGD optimizer! Once we found the best one, it greatly improved the MAE.
