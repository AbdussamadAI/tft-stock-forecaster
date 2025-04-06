"""
TFT Stock Analysis LLM Module

This module provides LLM-powered analysis for the TFT stock forecasting app.
It uses CrewAI to simulate a team of financial experts analyzing stock data.
"""

import os
from datetime import datetime
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
        self.use_llm = self._validate_api_key()
        
        # Only create agents if we have a valid API key
        if self.use_llm:
            try:
                # Import CrewAI here to avoid errors if not using LLM
                from crewai import Agent, Crew, Process
                
                # Create the specialized agents
                self.financial_analyst = self._create_financial_analyst()
                self.market_researcher = self._create_market_researcher()
                self.investment_advisor = self._create_investment_advisor()
                self.tft_interpreter = self._create_tft_interpreter()
            except Exception as e:
                logging.error(f"Error creating CrewAI agents: {str(e)}")
                self.use_llm = False
    
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
        """Analyze a stock using CrewAI agents.
        
        Args:
            symbol (str): Stock symbol to analyze.
            forecast_data (pd.DataFrame): Forecast data from the TFT model.
            model_metrics (dict): Performance metrics of the TFT model.
            feature_importance (dict): Feature importance from the TFT model.
            
        Returns:
            str: Comprehensive stock analysis.
        """
        if not self.use_llm:
            return self._generate_fallback_analysis(symbol, forecast_data, "API key validation failed or CrewAI not available")
        
        try:
            # Set API key explicitly for this session
            os.environ["OPENAI_API_KEY"] = self.api_key
            
            # Import CrewAI here to avoid errors if not using LLM
            from crewai import Crew, Process
            
            # Create the tasks
            financial_analysis_task = self._create_financial_analysis_task(symbol, forecast_data, feature_importance)
            market_research_task = self._create_market_research_task(symbol)
            tft_interpretation_task = self._create_tft_interpretation_task(symbol, model_metrics, feature_importance)
            
            # Assign agents to tasks
            financial_analysis_task.agent = self.financial_analyst
            market_research_task.agent = self.market_researcher
            tft_interpretation_task.agent = self.tft_interpreter
            
            # Create the initial crew for sequential tasks with memory disabled to avoid SQLite issues
            initial_crew = Crew(
                agents=[self.financial_analyst, self.market_researcher, self.tft_interpreter],
                tasks=[financial_analysis_task, market_research_task, tft_interpretation_task],
                verbose=True,
                process=Process.sequential,
                memory=False  # Disable memory to avoid SQLite/Chroma DB issues
            )
            
            # Run the initial crew
            result = initial_crew.kickoff()
            logging.info(f"Initial crew result type: {type(result)}")
            
            # Handle different result formats from CrewAI
            if isinstance(result, list) and len(result) > 0:
                financial_analysis = result[0]
                market_research = result[1] if len(result) > 1 else None
                model_interpretation = result[2] if len(result) > 2 else None
            else:
                # If result is not a list, it might be a single output or a different format
                financial_analysis = result
                market_research = None
                model_interpretation = None
            
            # Create the investment recommendation task
            investment_recommendation_task = self._create_investment_recommendation_task(
                symbol,
                financial_analysis=financial_analysis,
                market_research=market_research,
                model_interpretation=model_interpretation
            )
            
            # Assign the investment advisor to the task
            investment_recommendation_task.agent = self.investment_advisor
            
            # Create the recommendation crew with memory disabled to avoid SQLite issues
            recommendation_crew = Crew(
                agents=[self.investment_advisor],
                tasks=[investment_recommendation_task],
                verbose=True,
                memory=False  # Disable memory to avoid SQLite/Chroma DB issues
            )
            
            # Run the recommendation crew
            final_result = recommendation_crew.kickoff()
            
            # Debug information about the output types
            logging.info(f"Financial analysis type: {type(financial_analysis)}")
            logging.info(f"Market research type: {type(market_research)}")
            logging.info(f"Model interpretation type: {type(model_interpretation)}")
            logging.info(f"Final result type: {type(final_result)}")
            
            # Extract content from different output types
            def extract_content(output):
                if hasattr(output, 'raw_output'):
                    return output.raw_output
                elif hasattr(output, 'output'):
                    return output.output
                elif hasattr(output, 'result'):
                    return output.result
                elif isinstance(output, str):
                    return output
                else:
                    return str(output)
            
            # Combine all analyses into a comprehensive report
            comprehensive_analysis = f"""
            # Comprehensive TFT Stock Analysis for {symbol}
            
            ## Financial Analysis
            {extract_content(financial_analysis)}
            
            ## Market Research
            {extract_content(market_research) if market_research else "No market research available."}
            
            ## TFT Model Interpretation
            {extract_content(model_interpretation) if model_interpretation else "No model interpretation available."}
            
            ## Investment Recommendation
            {extract_content(final_result)}
            
            *Analysis generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
            """
            
            return comprehensive_analysis
            
        except Exception as e:
            logging.error(f"Error during LLM call: {str(e)}")
            return self._generate_fallback_analysis(symbol, forecast_data, f"Error during LLM call: {str(e)}")
    
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
