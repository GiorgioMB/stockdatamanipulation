"""
This module contains the DataFetcher class which is designed to retrieve financial information. 
It uses the yfinance library to gather historical data for specified companies, including data on similar 
companies based on their sector and beta value. It also provides functionalities to fetch company tickers and 
historical stock price data, enhancing the ease of financial data analysis.
"""
from requests.exceptions import RequestException
import yfinance as yf
import pandas as pd
import requests

class Fetcher:
    """
    Serves as a comprehensive data retrieval utility aimed at gathering 
    financial information about a specified company. 
    """
    def __init__(self, ticker:str, period:str = 'max'):
        self.ticker = ticker
        self.df, self.actual_ticker = self.get_historical_data(self.ticker, period)    
    def get_ticker(self, company_name:str)->str:
        """
        This function returns the ticker of a company, given the name.
        Input:
            -company_name: string of lenght > 4
        """
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like"
            " Gecko) Chrome/108.0.0.0 Safari/537.36"
        )
        params = {"q": company_name, "quotes_count": 1, "country": "United States"}
        res = requests.get(
            url=yfinance, params=params, headers={"User-Agent": user_agent}, timeout = 50
        )
        data = res.json()
        if "quotes" in data and len(data["quotes"]) > 0:
            company_code = data["quotes"][0]["symbol"]
            return company_code
        raise AssertionError("Company may not be listed")
    def get_historical_data(self, symbol:str, timeframe:str)->tuple[pd.DataFrame, str]:
        """
        This function returns a PANDAS dataframe about the price of the company and the 
        stock symbol of the company input. The company must be listed in the NYSE.
        
        Input:
            -symbol: string with the name (or ticker) of the company in question; 
            capitalization is not regarded.
        """
        try:
            stock_symbol = symbol
            stock_symbol = self.get_ticker(stock_symbol)
            if stock_symbol is None:
                return None
            stock = yf.Ticker(stock_symbol)
            historical_data = stock.history(period=timeframe)
            historical_data["Volatility"] = historical_data["Close"].rolling(window=30).std()
            historical_data = historical_data.dropna()
            return historical_data, stock_symbol
        except RequestException as e:     
            print(f"Network-related error occurred: {e}")
        return None, None
