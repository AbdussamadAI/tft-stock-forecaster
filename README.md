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

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key (Optional for AI Analysis)

To enable AI-powered stock analysis, you need to configure your OpenAI API key:

**Option A: Using .env file (Recommended for local development)**
```bash
# Create a .env file in the project root
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

**Option B: Using environment variable**
```bash
export OPENAI_API_KEY=your_api_key_here
```

**Option C: For Streamlit Cloud deployment**
- Go to your app settings in Streamlit Cloud dashboard
- Add `OPENAI_API_KEY` as a secret
- Set the value to your OpenAI API key

### 3. Run the Application
```bash
streamlit run tft_forecast_app.py
```

### 4. Test OpenAI Integration (Optional)
```bash
python3 test_openai.py
```

**Note:** The app will work without an OpenAI API key, but the AI analysis feature will use a simplified template-based analysis instead.
