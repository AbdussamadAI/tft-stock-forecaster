# Setup Guide for TFT Stock Forecaster with DeepSeek Integration

## Overview
This guide will help you set up the TFT Stock Forecaster app with DeepSeek API integration for AI-powered stock analysis.

## Prerequisites
- Python 3.8 or higher
- pip package manager
- DeepSeek API key (get one at https://platform.deepseek.com/api_keys)

## Step-by-Step Setup

### 1. Clone/Download the Repository
If you haven't already, ensure you have all the project files in your local directory.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- yfinance
- pandas, numpy
- plotly
- scikit-learn
- torch (PyTorch)
- python-dotenv
- openai (DeepSeek's API is OpenAI-compatible, so we use the OpenAI SDK)

### 3. Configure DeepSeek API Key

#### Method 1: Using .env File (Recommended for Local)
1. Create a file named `.env` in the project root directory
2. Add your DeepSeek API key:
```
DEEPSEEK_API_KEY=sk-your-actual-api-key-here
```
3. Save the file

**Note:** The `.env` file is already in `.gitignore`, so it won't be committed to git.

#### Method 2: Using Environment Variables
```bash
# For macOS/Linux
export DEEPSEEK_API_KEY=sk-your-actual-api-key-here

# For Windows (Command Prompt)
set DEEPSEEK_API_KEY=sk-your-actual-api-key-here

# For Windows (PowerShell)
$env:DEEPSEEK_API_KEY="sk-your-actual-api-key-here"
```

### 4. Verify Your Setup
Run the test script to ensure everything is configured correctly:
```bash
python3 test_deepseek.py
```

You should see:
```
✓ API key loaded from .env file
✓ DeepSeek client initialized successfully
✓ API call successful!
✓ All tests passed! Your DeepSeek API key is configured correctly.
```

### 5. Run the Application
```bash
streamlit run tft_forecast_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Using the App

### Basic Features (No API Key Required)
- Select stock symbols from different markets (US, London, Pakistan)
- Choose historical data periods
- Set forecast horizons (1-30 days)
- View interactive price charts
- Train TFT model and view forecasts
- See technical indicators and feature importance

### AI Analysis Feature (Requires API Key)
1. After generating a forecast, scroll down to the "AI Stock Analysis" section
2. Click the "Generate AI Analysis" button
3. Wait for the AI to analyze the forecast (takes ~10-30 seconds)
4. View the comprehensive analysis including:
   - Forecast trend interpretation
   - Model reliability assessment
   - Key feature analysis
   - Investment recommendations
   - Risk factors

## Troubleshooting

### Issue: "No valid DeepSeek API key available"
**Solution:**
- Verify your `.env` file exists and contains the correct API key
- Check that the key starts with `sk-`
- Run `python3 test_deepseek.py` to diagnose the issue

### Issue: "DeepSeek API error: Incorrect API key provided"
**Solution:**
- Your API key is invalid or expired
- Generate a new key at https://platform.deepseek.com/api_keys
- Update your `.env` file with the new key

### Issue: "DeepSeek API error: You exceeded your current quota"
**Solution:**
- You've used up your DeepSeek API credits
- Top up your balance at https://platform.deepseek.com/top_up
- The app will still work, but AI analysis will use fallback template

### Issue: Module import errors
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub (make sure `.env` is in `.gitignore`)
2. Go to https://share.streamlit.io/
3. Connect your GitHub repository
4. In the app settings, add your secrets:
   - Click "Advanced settings"
   - Add `DEEPSEEK_API_KEY = "your-key-here"` to the secrets section
5. Deploy!

## Cost Considerations

The AI analysis feature uses DeepSeek's `deepseek-chat` model, which is very cost-effective:
- Approximate cost: a fraction of a cent per analysis
- A small credit balance can generate thousands of analyses

If you don't want to use DeepSeek, the app will automatically fall back to template-based analysis.

## Support

For issues or questions:
1. Check this guide first
2. Run `python3 test_deepseek.py` to diagnose API issues
3. Check the terminal/console for error messages
4. Review the Streamlit app logs

## Security Notes

- Never commit your `.env` file to git
- Never share your DeepSeek API key publicly
- Rotate your API key if it's been exposed
- Monitor your DeepSeek usage at https://platform.deepseek.com/usage
