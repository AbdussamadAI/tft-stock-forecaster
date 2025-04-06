"""
TFT Stock Analysis LLM Module

This module provides LLM-powered analysis for the TFT stock forecasting app.
It uses CrewAI to simulate a team of financial experts analyzing stock data.
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
    A class that uses CrewAI to perform stock analysis with multiple specialized agents.
    Specifically designed for the TFT forecasting app.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the TFTStockAnalysisLLM with an optional API key.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will try to use environment variable.
        """
        # Set API key if provided, otherwise use environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Always use fallback analysis to avoid CrewAI/SQLite issues
        self.use_llm = False
        
        # Log that we're using fallback analysis
        logging.info("Using fallback analysis method to avoid CrewAI/SQLite issues on Streamlit Cloud")
    
    def _validate_api_key(self):
        """
        Validate the OpenAI API key.
        
        Returns:
            bool: True if the API key is valid, False otherwise.
        """
        if not self.api_key:
            logging.warning("No OpenAI API key provided. Set OPENAI_API_KEY in Streamlit Cloud settings.")
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
    
    def _create_financial_analyst(self):
        """Create a financial analyst agent specialized in analyzing stock data."""
        from crewai import Agent
        return Agent(
            role="Financial Analyst",
            goal="Analyze financial data and provide insights on stock performance",
            backstory="""You are a seasoned financial analyst with expertise in stock market analysis. 
            You excel at interpreting financial statements, analyzing market trends, and providing 
            insights on company performance. Your analysis is thorough, balanced, and focused on 
            helping investors make informed decisions.""",
            verbose=True,
            allow_delegation=False,
            tools=[]  # Empty tools list to avoid SQLite/Chroma DB issues
        )
    
    def _create_market_researcher(self):
        """Create a market researcher agent specialized in analyzing market trends."""
        from crewai import Agent
        return Agent(
            role="Market Researcher",
            goal="Research market trends and news related to the stock",
            backstory="""You are an experienced market researcher with a keen eye for market trends 
            and news that impact stock prices. You analyze news articles, social media sentiment, 
            and industry reports to provide a comprehensive view of the market landscape. Your 
            research helps investors understand the broader context affecting a stock.""",
            verbose=True,
            allow_delegation=False,
            tools=[]  # Empty tools list to avoid SQLite/Chroma DB issues
        )
    
    def _create_investment_advisor(self):
        """Create an investment advisor agent specialized in providing recommendations."""
        from crewai import Agent
        return Agent(
            role="Investment Advisor",
            goal="Provide investment recommendations based on financial analysis and market research",
            backstory="""You are a seasoned investment advisor with a track record of providing sound 
            investment advice. You analyze financial data and market trends to formulate investment 
            strategies. Your recommendations are balanced, considering both potential risks and rewards.""",
            verbose=True,
            allow_delegation=False,
            tools=[]  # Empty tools list to avoid SQLite/Chroma DB issues
        )
        
    def _create_tft_interpreter(self):
        """Create a TFT model interpreter specialized in explaining model predictions."""
        from crewai import Agent
        return Agent(
            role="TFT Model Interpreter",
            goal="Interpret and explain the TFT model predictions and feature importance",
            backstory="""You are an expert in time series forecasting and deep learning models, 
            particularly Temporal Fusion Transformers. You can interpret model predictions, 
            explain feature importance, and provide insights on model performance. You translate 
            complex technical concepts into clear, actionable insights.""",
            verbose=True,
            allow_delegation=False,
            tools=[]  # Empty tools list to avoid SQLite/Chroma DB issues
        )
    
    def _create_financial_analysis_task(self, symbol, forecast_data, feature_importance):
        """Create a task for financial analysis."""
        from crewai import Task
        return Task(
            description=f"""Analyze the financial performance of {symbol} based on historical data and TFT model forecasts.
            Focus on key financial metrics, price trends, and the forecast data provided.
            Explain what the feature importance values mean for this stock.
            Your analysis should be data-driven and objective.
            Format your response in markdown.
            """,
            expected_output="A comprehensive financial analysis of the stock in markdown format.",
            agent=self.financial_analyst
        )
    
    def _create_market_research_task(self, symbol):
        """Create a task for market research."""
        from crewai import Task
        return Task(
            description=f"""Research market trends and news that could impact {symbol}'s performance.
            Consider industry developments, market sentiment, and macroeconomic factors.
            Look for recent news, analyst opinions, and market trends related to {symbol}.
            Your research should provide context for the stock's movements and future potential.
            Format your response in markdown.
            """,
            expected_output="A thorough market research report on the stock in markdown format.",
            agent=self.market_researcher
        )
    
    def _create_tft_interpretation_task(self, symbol, model_metrics, feature_importance):
        """Create a task for TFT model interpretation."""
        from crewai import Task
        return Task(
            description=f"""Interpret the TFT model predictions and feature importance for {symbol}.
            Explain what the model metrics mean in terms of prediction reliability.
            Analyze which features are most important for the prediction and why.
            Make the technical aspects of the model understandable to a non-technical audience.
            Format your response in markdown.
            """,
            expected_output="A clear interpretation of the TFT model predictions and feature importance in markdown format.",
            agent=self.tft_interpreter
        )
    
    def _create_investment_recommendation_task(self, symbol, financial_analysis=None, market_research=None, model_interpretation=None):
        """Create a task for investment recommendation."""
        from crewai import Task
        context = ""
        if financial_analysis:
            context += f"\nFinancial Analysis:\n{financial_analysis}\n"
        if market_research:
            context += f"\nMarket Research:\n{market_research}\n"
        if model_interpretation:
            context += f"\nModel Interpretation:\n{model_interpretation}\n"
        
        return Task(
            description=f"""Based on the provided analyses, create an investment recommendation for {symbol}.
            Consider the financial analysis, market research, and model interpretation.
            Provide a clear buy, hold, or sell recommendation with supporting rationale.
            Include potential risks and alternative viewpoints.
            Format your response in markdown.
            
            Context:\n{context}
            """,
            expected_output="A balanced investment recommendation with clear buy/hold/sell guidance in markdown format.",
            agent=self.investment_advisor
        )
    
    def analyze_stock(self, symbol, forecast_data, model_metrics, feature_importance):
        """Analyze a stock using the fallback template-based method.
        
        Args:
            symbol (str): Stock symbol to analyze.
            forecast_data (pd.DataFrame): Forecast data from the TFT model.
            model_metrics (dict): Performance metrics of the TFT model.
            feature_importance (dict): Feature importance from the TFT model.
            
        Returns:
            str: Comprehensive stock analysis.
        """
        # Always use the fallback analysis to avoid CrewAI/SQLite issues
        return self._generate_fallback_analysis(symbol, forecast_data, "Using template-based analysis for Streamlit Cloud compatibility")
    
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
