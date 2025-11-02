# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

TFT Stock Forecasting Tool - A Streamlit application that uses a simplified Temporal Fusion Transformer (TFT) model for stock market forecasting with AI-powered analysis using OpenAI.

## Common Commands

### Development
```bash
# Run the application locally
streamlit run tft_forecast_app.py

# Test OpenAI API integration
python3 test_openai.py

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
```bash
# Create and configure .env file for local development
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Export environment variable (alternative to .env)
export OPENAI_API_KEY=your_api_key_here
```

## Architecture

### Core Components

**tft_forecast_app.py** (main application, ~700+ lines)
- Streamlit UI with sidebar for stock selection, period, and forecast configuration
- Simplified TFT model implementation (`TemporalFusionTransformer` class) - embedded neural network with 3-layer architecture (input → hidden → output)
- Custom `StockDataset` class for PyTorch data handling
- Data pipeline: yfinance → technical indicators → normalization → sequences → training/validation split
- Training loop with progress tracking and best model checkpointing
- Prediction generation with synthetic OHLC forecast based on model output

**tft_analysis_llm.py** (AI analysis module)
- `TFTStockAnalysisLLM` class that interfaces with OpenAI API
- Uses GPT-4o-mini for cost-effective stock analysis
- Fallback mechanism: if API key is missing/invalid, generates template-based analysis instead
- Analyzes forecast data, model metrics, and feature importance to provide investment recommendations

### Data Flow

1. **Input**: User selects stock symbol (with market suffix: `.L` for London, `.KA` for Pakistan), historical period (1mo to 5y), and forecast days (1-30)
2. **Data Loading**: yfinance fetches OHLC + volume data with 1-hour caching
3. **Feature Engineering**: `add_technical_indicators()` creates 10+ features (MA5/MA20, RSI, MACD, Bollinger Bands, volume metrics, price momentum)
4. **Preprocessing**: 
   - StandardScaler normalization
   - Robust handling of NaN/infinity values (bfill/ffill, then zero-fill)
   - Adaptive sequence length based on data availability
   - Creates sliding windows for training
5. **Model Training**: 
   - 80/20 train/val split
   - MSE loss with Adam optimizer
   - Dimension matching logic handles multi-day forecasts (reshapes outputs to match target shape)
   - Tracks best validation loss for model selection
6. **Prediction**: 
   - Uses last validation sequence
   - Generates forecast with randomized OHLC values based on predicted trend direction
7. **AI Analysis** (optional): OpenAI analyzes forecast + metrics + feature importance → investment recommendation

### Key Design Patterns

**Robust Error Handling**: The app is designed to run on Streamlit Cloud with limited resources. It includes:
- PyTorch import error handling with dummy classes fallback
- Data validation with NaN/infinity cleaning at multiple stages
- Adaptive sequence length when insufficient data available
- Dimension mismatch handling in training loop
- API key validation with masked logging

**Fallback Mechanisms**:
- If OpenAI API fails → template-based analysis
- If data is insufficient → reduced sequence length
- If validation set can't be created → clones training data

**Streamlit Deployment Optimizations**:
- `torch._classes = None` to prevent Streamlit Cloud errors
- CPU-only PyTorch to reduce deployment size
- No external `src/` dependencies
- All model code embedded in main app file

## Important Technical Details

### Model Architecture
The TFT model is simplified (not the full academic implementation):
- Input layer: maps features → hidden_size (default 128)
- Hidden layer: hidden_size → hidden_size with ReLU
- Output layer: hidden_size → 4 (OHLC prediction)
- No actual attention mechanism despite the TFT name

### Training Parameters (Fixed)
- Sequence length: 20 days
- Hidden size: 128
- Attention heads: 4 (not actually used)
- Layers: 2
- Epochs: 20
- Batch size: 64
- Learning rate: 0.001

### Feature Set (15 features)
Core: Open, High, Low, Close, Volume
Technical: MA5, MA20, RSI, MACD, Signal, BB_Upper, BB_Lower, Volume_Change, Price_Change, Price_Change_5d

### Dimension Handling in Training
The model outputs shape `[batch_size, 4]` but targets are `[batch_size, forecast_days, 4]`. The training loop includes logic to expand/reshape outputs to match targets using unsqueeze and expand operations.

## Environment Variables

**OPENAI_API_KEY** (optional):
- Required for AI-powered analysis feature
- Store in `.env` file for local development (already in .gitignore)
- Use Streamlit Cloud secrets for deployment
- App works without it (falls back to template analysis)

## Deployment

### Streamlit Cloud
- Uses Procfile: `web: sh setup.sh && streamlit run tft_forecast_app.py`
- setup.sh creates Streamlit config at runtime
- Add OPENAI_API_KEY to Streamlit Cloud app secrets
- Dependencies: requirements.txt (CPU-only PyTorch recommended)

### Local Development
1. Install requirements: `pip install -r requirements.txt`
2. Configure API key: create `.env` with OPENAI_API_KEY
3. Verify setup: `python3 test_openai.py`
4. Run: `streamlit run tft_forecast_app.py`

## Code Modification Guidelines

### When Editing tft_forecast_app.py
- Keep `st.set_page_config()` at the very top (Streamlit requirement)
- Maintain robust NaN/infinity handling in any data processing code
- Preserve dimension matching logic in training loop
- Don't assume sequence length - use adaptive logic
- Use `.bfill().ffill()` instead of deprecated `fillna(method=...)`

### When Editing tft_analysis_llm.py
- Always provide fallback analysis path
- Mask API keys in logs: `f"{key[:3]}...{key[-3:]}"`
- Keep model parameter as `gpt-4o-mini` (cost-effective)
- Handle OpenAI API exceptions gracefully

### Adding New Technical Indicators
1. Add calculation in `add_technical_indicators()`
2. Include in `features` list in `prepare_tft_data()`
3. Update input_size calculation (currently `len(features)`)
4. Ensure NaN handling for new indicator

## Testing

**Manual Testing Workflow**:
1. Run app locally: `streamlit run tft_forecast_app.py`
2. Test different stock symbols across markets (US: AAPL, London: HSBA.L, Pakistan: HBL.KA)
3. Test different time periods (1mo through 5y) to verify adaptive sequence length
4. Verify forecasts generate without errors
5. Test AI analysis with valid/invalid/missing API key

**API Testing**:
```bash
python3 test_openai.py
```
Expected output: ✓ all tests passed

## File Structure

```
.
├── tft_forecast_app.py          # Main Streamlit application
├── tft_analysis_llm.py          # OpenAI analysis module
├── test_openai.py               # API key verification script
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for environment variables
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── Procfile                     # Deployment configuration
├── setup.sh                     # Streamlit Cloud setup script
├── README.md                    # User documentation
├── SETUP_GUIDE.md              # Detailed setup instructions
└── QUICKSTART.md               # Quick start guide
```

## Stock Market Symbols

The app supports three markets with different suffixes:
- **US (NYSE/NASDAQ)**: No suffix (e.g., `AAPL`, `MSFT`, `GOOGL`)
- **London (LSE)**: `.L` suffix (e.g., `HSBA.L`, `BP.L`, `GSK.L`)
- **Pakistan (PSX)**: `.KA` suffix (e.g., `HBL.KA`, `UBL.KA`, `ENGRO.KA`)

Symbol validation happens via yfinance - invalid symbols return empty dataframe and trigger error message.
