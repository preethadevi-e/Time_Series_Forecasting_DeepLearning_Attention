# Time_Series_Forecasting_DeepLearning_Attention
Advanced Time Series Forecasting with Attention Mechanisms

Project Overview : 
    This project implements and evaluates deep learning models for multivariate time series forecasting. A baseline LSTM model is compared with an attention-augmented LSTM to demonstrate how attention improves prediction accuracy and interpretability.

Objectives : 
    1.Forecast future values from multivariate time series data
    2.Implement a baseline LSTM/GRU sequence-to-sequence model
    3.Enhance the model using a custom attention mechanism
    4.Compare model performance using RMSE and MAE
    5.Interpret attention weights to identify important historical time steps

Dataset : 
    Electricity Consumption Dataset (multivariate)

Source: statsmodels
    Features include multiple electricity usage measurements over time

Methodology :
1. Data Preprocessing
    Missing value handling
    Feature normalization using Min-Max scaling
    Time-series windowing (sliding window approach)
2. Models Implemented
    Baseline Model: LSTM sequence-to-sequence model
    Advanced Model: LSTM with custom attention layer
3. Training & Validation
    Time-based train/validation split
    Rolling/expanding window evaluation strategy
4. Evaluation Metrics
    Root Mean Squared Error (RMSE)
    Mean Absolute Error (MAE)

Attention Mechanism : 
    The attention layer dynamically assigns importance weights to past time steps, allowing the model to focus on the most relevant historical information for each prediction. This improves both forecasting accuracy and model interpretability.

Results : 
    The attention-based LSTM achieved lower RMSE and MAE compared to the baseline model
    Attention weights showed higher importance for recent and peak-usage time steps
    Demonstrated clear performance improvement over standard LSTM

Project Structure : 
    attention_time_series.py
    README.md

How to Run :
    pip install tensorflow scikit-learn pandas numpy statsmodels
    python attention_time_series.py

Conclusion :
    Attention mechanisms significantly enhance deep learning–based time series forecasting by improving accuracy and providing insight into which historical patterns influence predictions.
