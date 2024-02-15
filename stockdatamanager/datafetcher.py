"""
This module contains the Fetcher class which is designed to retrieve financial information. 
It uses the yfinance library to gather historical data for specified companies.
"""
from requests.exceptions import RequestException
import yfinance as yf
import pandas as pd
import requests
from typing import Tuple


class Fetcher:
    """
    Serves as a comprehensive data retrieval utility aimed at gathering 
    financial information about a specified company. 
    """
    def __init__(self, ticker:str, period:str = 'max'):
        self.ticker = ticker
        self.df, self.yf_stock = self.get_historical_data(self.ticker, period)
        self.income_statement, self.balance_sheet, self.cashflow = self.get_financial_statements(self.yf_stock)    
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
            if not stock_symbol:
                return None
            stock = yf.Ticker(stock_symbol)
            historical_data = stock.history(period=timeframe)
            historical_data["Volatility"] = historical_data["Close"].rolling(window=30).std()
            historical_data = historical_data.dropna()
            return historical_data, stock
        except RequestException as e:     
            print(f"Network-related error occurred: {e}")
        return None, None
    
    def get_financial_statements(self, stock:yf.Ticker) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Given a yf.Ticker, it returns the three financial statements of the company, if present
        """
        try:
            income_statement = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            return income_statement, balance_sheet, cashflow
        except RequestException as e:
            print(f"Network-related error occurred: {e}")
            return None, None, None
    
    def get_income_statement(self, symbol:str= None):
        """Returns the income stament
        if "symbol" is not specified, it returns the income statement of the company the 
        Fetcher has been initialized to
        """
        if not symbol:
            return self.income_statement
        else:
            stock_symbol = symbol
            stock_symbol = self.get_ticker(stock_symbol)
            stock = yf.Ticker(stock_symbol)
            return stock.financials
    
    def get_balance_sheet(self, symbol:str= None):
        """Returns the balance sheet
        if "symbol" is not specified, it returns the income statement of the company the 
        Fetcher has been initialized to
        """
        if not symbol:
            return self.balance_sheet
        else:
            stock_symbol = symbol
            stock_symbol = self.get_ticker(stock_symbol)
            stock = yf.Ticker(stock_symbol)
            return stock.balance_sheet
    
    def get_cashflow(self, symbol:str= None):
        """Returns the cash flow
        if "symbol" is not specified, it returns the income statement of the company the 
        Fetcher has been initialized to
        """
        if not symbol:
            return self.cashflow
        else:
            stock_symbol = symbol
            stock_symbol = self.get_ticker(stock_symbol)
            stock = yf.Ticker(stock_symbol)
            return stock.cashflow
    
    def get_beta_value(self) -> float:
        """Returns the beta value of the stock"""
        return self.yf_stock.info['beta']
    
    def get_alpha_value(self) -> float:
        """Returns the alpha value of the stock"""
        return self.yf_stock.info['alpha']
    
    def get_risk_free_rate(self, horizon:str = 'month') -> float:
        """Returns the risk free rate"""
        if horizon == 'month':
            return yf.Ticker('^IRX').history(period='max')
        elif horizon == '5 years':
            return yf.Ticker('^FVX').history(period='max')
        elif horizon == '10 years':
            return yf.Ticker('^TNX').history(period='max')
        elif horizon == '30 years':
            return yf.Ticker('^TYX').history(period='max')
        else:
            raise ValueError('Invalid horizon, please choose between "month", "5 years", "10 years" and "30 years"')

    def get_sp500(self) -> pd.DataFrame:
        """Returns the S&P 500"""
        return yf.Ticker('^GSPC').history(period='max')