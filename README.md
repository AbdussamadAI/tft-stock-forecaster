# TFT Stock Forecasting Tool

A Streamlit application for stock market forecasting using a simplified Temporal Fusion Transformer (TFT) model.

## Features

- Select from different stock markets (US, London, Pakistan)
- Choose historical data period (1 month to 5 years)
- Set custom forecast horizon (1-30 days)
- Interactive visualizations of historical data and forecasts
- Technical indicators including MA, RSI, MACD, and Bollinger Bands
- Real-time model training with progress tracking

## Technical Details

- Built with Streamlit, PyTorch, and yfinance
- Uses a simplified TFT model architecture
- Robust data preprocessing with handling for NaN and infinity values
- Adaptive sequence length for different data sizes
- OHLC (Open, High, Low, Close) price forecasting

## Requirements

See requirements.txt for the full list of dependencies.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run tft_forecast_app.py
```
