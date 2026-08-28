# Quick Start Guide

## Get Your App Running in 3 Minutes!

### 1. Install Dependencies (30 seconds)
```bash
cd /Users/abdulsamad/Desktop/MS-deployment/tft-stock-forecaster
pip install -r requirements.txt
```

### 2. Set Up API Key (30 seconds)
Add your DeepSeek API key to the `.env` file:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

To verify it works:
```bash
python3 test_deepseek.py
```

Expected output:
```
✓ API key loaded from .env file
✓ DeepSeek client initialized successfully
✓ API call successful!
✓ All tests passed!
```

### 3. Run the App (10 seconds)
```bash
streamlit run tft_forecast_app.py
```

Your browser will open automatically at `http://localhost:8501`

## How to Use

1. **Select a stock** (e.g., AAPL for Apple)
2. **Choose time period** (e.g., 1 Year)
3. **Set forecast days** (e.g., 5 days)
4. Wait for the forecast to generate (~30 seconds)
5. **Click "Generate AI Analysis"** for AI-powered insights! 🤖

## What You Get

### Without API Key:
- Stock price charts ✅
- TFT forecasts ✅
- Technical indicators ✅
- Template analysis ⚠️

### With API Key:
- Stock price charts ✅
- TFT forecasts ✅
- Technical indicators ✅
- **AI-powered analysis** ✅🎉
  - Forecast interpretation
  - Model reliability assessment
  - Investment recommendations
  - Risk analysis

## Troubleshooting

### App won't start?
```bash
pip install --upgrade streamlit
streamlit run tft_forecast_app.py
```

### API not working?
```bash
python3 test_deepseek.py
```

### Need help?
Check `SETUP_GUIDE.md` for detailed instructions.

## Cost

DeepSeek's `deepseek-chat` model is very cost-effective:
- Each AI analysis costs a fraction of a cent
- DeepSeek's pricing is among the most affordable for this type of analysis

## What's Next?

Try analyzing different stocks:
- **Tech**: AAPL, MSFT, GOOGL, NVDA
- **Index**: SPY, QQQ
- **International**: Add .L for London stocks

Have fun! 🚀📈
