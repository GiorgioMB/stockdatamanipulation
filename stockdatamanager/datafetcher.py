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
    A financial data retrieval utility designed to fetch and analyze financial 
    information for a specified company. This class offers methods to retrieve 
    historical stock data, financial statements (income statement, balance sheet, 
    and cash flow statement), and compute various financial metrics and ratios 
    critical for financial analysis.

    Attributes:
        ticker (str): The stock ticker symbol for the company.
        df (pd.DataFrame): DataFrame containing the company's historical stock data.
        yf_stock (yf.Ticker): yfinance Ticker object for accessing Yahoo Finance API data.
        income_statement (pd.DataFrame): Company's income statement.
        balance_sheet (pd.DataFrame): Company's balance sheet.
        cashflow (pd.DataFrame): Company's cash flow statement.

    Methods offer capabilities to:
    - Retrieve the company's ticker symbol based on its name.
    - Fetch historical price data and compute volatility.
    - Access detailed financial statements.
    - Calculate key financial metrics and ratios such as EPS, ROA, ROE, and more.
    - Evaluate the company's liquidity, leverage, and profitability.

    Usage:
    fetcher = Fetcher(ticker='AAPL')
    print(fetcher.get_pe_ratio())
    print(fetcher.get_current_ratio())

    Note:
    This class requires an internet connection to fetch data from Yahoo Finance.
    """
    def __init__(self, ticker:str = None, period:str = 'max'):
        if ticker:
            self.ticker = ticker
            self.df, self.yf_stock = self.get_historical_data(self.ticker, period)
            self.income_statement, self.balance_sheet, self.cashflow = self.get_financial_statements(self.yf_stock)
        else:
            print("Warning, no ticker has been specified. Only get_ticker(), get_risk_free_rate() and get_sp500() methods will work.")    
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
    def get_historical_data(self, symbol:str, timeframe:str)-> Tuple[pd.DataFrame, str]:
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
    
    def get_income_statement(self, symbol:str= None) -> pd.DataFrame:
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
    
    def get_balance_sheet(self, symbol:str= None) -> pd.DataFrame:
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
    
    def get_cashflow(self, symbol:str= None) -> pd.DataFrame:
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
    
    def get_pe_ratio(self) -> float:
        """Returns the price to earnings ratio"""
        return self.yf_stock.info['trailingPE']
    
    def get_ebitda(self) -> float:
        """Returns the EBITDA of the stock"""
        return self.income_statement.loc['EBITDA'].iloc[0]
    
    def get_ebit(self) -> float:
        """Returns the EBIT of the stock"""
        return self.income_statement.loc['EBIT'].iloc[0]

    def get_normalized_ebitda(self) -> float:
        """Returns the normalized EBITDA of the stock"""
        return self.income_statement.loc['Normalized EBITDA'].iloc[0]
    
    def get_gross_profit(self) -> float:
        """Returns the gross profit of the stock"""
        return self.income_statement.loc['Gross Profit'].iloc[0]
    
    def get_gross_profit_margin(self) -> float:
        """Returns the gross profit margin of the stock"""
        revenue = self.income_statement.loc['Total Revenue'].iloc[0]
        gross_profit = self.get_gross_profit()
        return (gross_profit / revenue) * 100
    
    def get_cogs_trend(self) -> float:
        """Returns the cost of goods sold"""
        cogs = self.income_statement.loc['Cost of Revenue']
        return cogs.pct_change()
    
    def get_roa(self) -> float:
        """Returns the return on assets"""
        net_income = self.income_statement.loc['Net Income'].iloc[0]
        total_assets = self.balance_sheet.loc['Total Assets'].iloc[0]
        return (net_income / total_assets) * 100
    
    def get_roe(self) -> float:
        """Returns the return on equity"""
        net_income = self.income_statement.loc['Net Income'].iloc[0]
        total_equity = self.balance_sheet.loc['Total Equity'].iloc[0]
        return (net_income / total_equity) * 100
    
    def get_total_current_assets(self) -> float:
        """Returns the current assets"""
        values_to_sum = []
        current_assets = 0
        for value in self.balance_sheet.index:
            if 'Current' in value and 'Assets' in value:
                values_to_sum.append(value)
        for value in values_to_sum:
            current_assets += self.balance_sheet.loc[value].iloc[0]
        return current_assets
        
    def get_total_current_liabilities(self) -> float:
        """Returns the current liabilities"""
        values_to_sum = []
        current_liabilities = 0
        for value in self.balance_sheet.index:
            if 'Current' in value and 'Liabilities' in value:
                values_to_sum.append(value)
        for value in values_to_sum:
            current_liabilities += self.balance_sheet.loc[value].iloc[0]
        return current_liabilities
    
    def get_current_ratio(self) -> float:
        """Returns the current ratio"""
        current_assets = self.get_total_current_assets()
        current_liabilities = self.get_total_current_liabilities()
        return (current_assets / current_liabilities) * 100

    def get_quick_ratio(self) -> float:
        """Returns the quick ratio"""
        current_assets = self.get_total_current_assets()
        inventory = self.balance_sheet.loc['Inventory'].iloc[0]
        current_liabilities = self.get_total_current_liabilities()
        return ((current_assets - inventory) / current_liabilities) * 100

    def get_debt_to_equity(self) -> float:
        """Returns the debt to equity ratio"""
        total_debt = self.balance_sheet.loc['Total Debt'].iloc[0]
        total_equity = self.balance_sheet.loc['Stockholders Equity'].iloc[0]
        return (total_debt / total_equity) * 100
    
    def get_interest_coverage(self) -> float:
        """Returns the interest coverage ratio"""
        ebit = self.get_ebit()
        interest_expense = self.income_statement.loc['Interest Expense'].iloc[0]
        return (ebit / interest_expense) * 100
    
    def get_cash_ratio(self) -> float:
        """Returns the cash ratio"""
        cash = self.balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
        current_liabilities = self.get_total_current_liabilities()
        return (cash / current_liabilities) * 100
    
    def get_net_working_capital(self) -> float:
        """Returns the net working capital"""
        current_assets = self.get_total_current_assets()
        current_liabilities = self.get_total_current_liabilities()
        return current_assets - current_liabilities
    
    def get_net_working_capital_ratio(self) -> float:
        """Returns the net working capital ratio"""
        net_working_capital = self.get_net_working_capital()
        current_liabilities = self.get_total_current_liabilities()
        return net_working_capital/ current_liabilities
    
    def get_eps(self) -> float:
        """Returns the earnings per share"""
        return self.income_statement.loc['Basic EPS'].iloc[0]
    
    def get_debt_ratio(self) -> float:
        """Returns the debt ratio"""
        total_debt = self.balance_sheet.loc['Total Debt'].iloc[0]
        total_assets = self.balance_sheet.loc['Total Assets'].iloc[0]
        return (total_debt / total_assets) * 100
    
    def get_equity_multiplier(self) -> float:
        """Returns the equity multiplier"""
        total_assets = self.balance_sheet.loc['Total Assets'].iloc[0]
        total_equity = self.balance_sheet.loc['Stockholders Equity'].iloc[0]
        return total_assets / total_equity
    
    def get_price_to_book(self) -> float:
        """Returns the price to book ratio"""
        book_value = self.balance_sheet.loc['Stockholders Equity'].iloc[0] / self.yf_stock.info['sharesOutstanding']
        return self.yf_stock.info['marketCap'] / book_value
    
    def get_price_to_sales(self) -> float:
        """Returns the price to sales ratio"""
        revenue = self.income_statement.loc['Total Revenue'].iloc[0]
        return self.yf_stock.info['marketCap'] / revenue


    def get_risk_free_rate(self, horizon:str = '13 weeks') -> float:
        """Returns the risk free rate, given an horizon.
        Accepted inputs are:
            -13 weeks
            -5 years
            -10 years
            -30 years
        """
        if horizon == '13 weeks':
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
    
    def get_dividend_yield(self) -> float:
        """Returns the dividend yield"""
        return self.yf_stock.info['dividendYield']
    
    def get_payout_ratio(self) -> float:
        """Returns the payout ratio"""
        return self.yf_stock.info['payoutRatio']

    def get_earnings_growth_rate(self) -> float:
        """Returns the growth rate"""
        return self.yf_stock.info['earningsQuarterlyGrowth']
    
