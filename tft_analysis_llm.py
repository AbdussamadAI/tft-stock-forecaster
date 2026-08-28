"""
TFT Stock Analysis LLM Module

This module provides LLM-powered analysis for the TFT stock forecasting app.
It uses the DeepSeek API (OpenAI-compatible) to analyze stock data.
"""

import os
import logging

# Try to import dotenv, but continue if it's not available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    logging.warning("Warning: python-dotenv not installed. Environment variables must be set manually.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables already loaded in the try-except block above

class TFTStockAnalysisLLM:
    """
    A class that uses the DeepSeek API to perform AI-powered stock analysis.
    Specifically designed for the TFT forecasting app.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the TFTStockAnalysisLLM with an optional API key.
        
        Args:
            api_key (str, optional): DeepSeek API key. If None, will try to use environment variable.
        """
        # Set API key if provided, otherwise use environment variable
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        
        # Check if we can use the DeepSeek API
        self.use_llm = self._validate_api_key()
        
        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
                )
                logging.info("DeepSeek client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize DeepSeek client: {e}")
                self.use_llm = False
        else:
            logging.info("Using fallback analysis method - no valid DeepSeek API key")
    
    def _validate_api_key(self):
        """
        Validate the DeepSeek API key.
        
        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        if not self.api_key:
            logging.warning("No DeepSeek API key provided. Set DEEPSEEK_API_KEY in your .env file or environment.")
            return False
        
        # For Streamlit Cloud, we'll be more lenient with validation
        # Just check if the key has a reasonable length
        if len(self.api_key) < 10:
            logging.warning("API key seems too short.")
            return False
        
        # If we get here, assume the key might be valid
        # Mask the key for logging
        masked_key = f"{self.api_key[:3]}...{self.api_key[-3:]}" if len(self.api_key) > 6 else "***masked***"
        logging.info(f"Using API key: {masked_key}")
        return True
    
    def analyze_stock(self, symbol, forecast_data, model_metrics, feature_importance):
        """Analyze a stock using the DeepSeek API.
        
        Args:
            symbol (str): Stock symbol to analyze.
            forecast_data (pd.DataFrame): Forecast data from the TFT model.
            model_metrics (dict): Performance metrics of the TFT model.
            feature_importance (dict): Feature importance from the TFT model.
            
        Returns:
            str: Comprehensive stock analysis.
        """
        if not self.use_llm:
            return self._generate_fallback_analysis(symbol, forecast_data, "No valid DeepSeek API key available")
        
        try:
            return self._generate_deepseek_analysis(symbol, forecast_data, model_metrics, feature_importance)
        except Exception as e:
            logging.error(f"Error generating DeepSeek analysis: {e}")
            return self._generate_fallback_analysis(symbol, forecast_data, f"DeepSeek API error: {str(e)}")
    
    def _generate_deepseek_analysis(self, symbol, forecast_data, model_metrics, feature_importance):
        """Generate analysis using the DeepSeek API directly.
        
        Args:
            symbol (str): Stock symbol.
            forecast_data (pd.DataFrame): Forecast data.
            model_metrics (dict): Model performance metrics.
            feature_importance (dict): Feature importance dictionary.
            
        Returns:
            str: AI-generated analysis.
        """
        # Prepare the forecast summary
        forecast_summary = ""
        if forecast_data is not None and not forecast_data.empty:
            forecast_summary = forecast_data.to_string()
            last_day = forecast_data.iloc[-1]
            first_day = forecast_data.iloc[0]
            price_change = ((last_day['Close'] - first_day['Close']) / first_day['Close']) * 100
            forecast_summary += f"\n\nOverall predicted change: {price_change:.2f}%"
        
        # Prepare feature importance summary
        feature_summary = "\n".join([f"- {k}: {v}" for k, v in sorted(feature_importance.items(), key=lambda x: float(x[1]), reverse=True)[:10]])
        
        # Create the prompt
        prompt = f"""You are a financial analyst expert. Analyze the stock {symbol} based on the following TFT (Temporal Fusion Transformer) model predictions and data:

**Forecast Data:**
{forecast_summary}

**Model Performance Metrics:**
- MSE: {model_metrics.get('MSE', 'N/A')}
- MAE: {model_metrics.get('MAE', 'N/A')}
- MAPE: {model_metrics.get('MAPE', 'N/A')}

**Top 10 Most Important Features:**
{feature_summary}

Provide a comprehensive analysis covering:
1. Interpretation of the forecast trend
2. Model reliability based on the metrics
3. Key features driving the prediction
4. Investment recommendation (Buy/Hold/Sell)
5. Risk factors to consider

Format your response in markdown with clear sections."""
        
        try:
            model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst with expertise in stock market analysis and machine learning models."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logging.error(f"DeepSeek API call failed: {e}")
            raise
    
    def _generate_fallback_analysis(self, symbol, forecast_data, error_message):
        """
        Generate a fallback analysis when LLM analysis fails.
        
        Args:
            symbol (str): Stock symbol.
            forecast_data (pd.DataFrame): Forecast data.
            error_message (str): Error message explaining why LLM analysis failed.
            
        Returns:
            str: Fallback analysis.
        """
        logging.warning(f"Using fallback analysis for {symbol}. Reason: {error_message}")
        
        # Get the forecast summary
        forecast_summary = ""
        if forecast_data is not None and not forecast_data.empty:
            last_day = forecast_data.iloc[-1]
            first_day = forecast_data.iloc[0]
            price_change = ((last_day['Close'] - first_day['Close']) / first_day['Close']) * 100
            direction = "increase" if price_change > 0 else "decrease"
            forecast_summary = f"The TFT model predicts a {abs(price_change):.2f}% {direction} over the forecast period."
        
        # Generate a simple template-based analysis
        fallback_analysis = f"""
            ## Simplified TFT Stock Analysis for {symbol}
            
            *Note: A comprehensive AI analysis could not be generated due to technical limitations. Here's a simplified analysis based on available data.*
            
            ### Company Overview
            - **Company:** {symbol}
            - **Sector:** Technology (placeholder)
            - **Industry:** Electronics (placeholder)
            
            ### Forecast Analysis
            The forecast generated by our Temporal Fusion Transformer (TFT) model suggests the following:
            
            {forecast_summary}
            
            ### Technical Indicators
            The model has analyzed various technical indicators including moving averages, RSI, MACD, and others to generate this forecast.
            
            ### Recommendation
            Based on the automated analysis of the forecast data, a general recommendation would be to:
            - Monitor the stock closely
            - Consider the forecast as one of many inputs for investment decisions
            - Conduct additional research before making investment decisions
            
            *This simplified analysis was generated automatically as a fallback. For a more comprehensive analysis, please try again later.*
            
            *Error details: {error_message}*
        """
        
        return fallback_analysis
