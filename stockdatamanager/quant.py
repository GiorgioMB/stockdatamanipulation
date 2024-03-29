from .datafetcher import Fetcher
import pandas as pd
import numpy as np
import yfinance as yf
import re
from typing import Union
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
class Greeks:
  """
  Class to calculate the greeks for an option
  Inputs:
  - ticker: str, the ticker of the stock
  - call: bool, whether the option is a call or a put
  - identification: Union[str, int], the contract symbol of the option or the index of the option in the option chain
                                     Note: to find the contract identifier, it is written as ticker-YY-MM-DD-C(or P)-strike price (omit the - and the strike price is in a thousands format, so 1000 is 1.00)
  - verbose: int, the verbosity of the output
  """
  def __init__(self, 
               ticker: str,
               call: bool,
               identification: Union[str, int],
               verbose: bool = False):
    self.fetcher = Fetcher(ticker)
    self.df, self.yf_ticker = self.fetcher.df, self.fetcher.yf_stock
    if type(identification) == str:
     self.option_chain = self.yf_ticker.option_chain().calls if call else self.yf_ticker.option_chain().puts
     self.option = self.option_chain[self.option_chain['contractSymbol'] == identification]
    elif type(identification) == int:
      self.option = self.yf_ticker.option_chain().calls.iloc[identification] if call else self.yf_ticker.option_chain().puts.iloc[identification]
    else:
      raise ValueError('identification must be either a string or an integer')
    self.verbose = verbose
    self.date = self.calculate_expiration()
    self.call = call

  def from_name_to_datestr(self, s: str) -> str:
    """Helper function to convert the contract symbol to a date string"""
    if self.verbose:
      print(f"Processing {s}")
    match = re.search(r'[A-Za-z](\d{2})(\d{2})(\d{2})[CP]', s)
    if match:
        year, month, day = match.groups()
        return f"20{year}-{month}-{day}"
    else:
        return "No date found"
  
  def calculate_expiration(self) -> pd.Timestamp:
    """Function to calculate the expiration date of the option"""
    if self.verbose:
      print("Calculating expiration date")
    to_process = self.option['contractSymbol']
    date_str = self.from_name_to_datestr(to_process)
    date = pd.to_datetime(date_str)
    return date

  def calculate_delta(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the delta of the option fed into the class, using the Black-Scholes formula.
    The delta is calculated as the first derivative of the option price with respect to the price of the underlying asset.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating delta")
    option = self.option
    dataframe = self.df
    if self.call:
      S = dataframe['Close'].iloc[-1]
      K = option['strike']
      ##convert self.date to days until expiration
      T = (self.date - pd.to_datetime('today')).days / 365
      r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
      sigma = option['impliedVolatility']
      d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
      delta = norm.cdf(d1)
      return delta
    else:
      S = dataframe['Close'].iloc[-1]
      K = option['strike']
      T = (self.date - pd.to_datetime('today')).days / 365
      r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
      sigma = option['impliedVolatility']
      d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
      delta = norm.cdf(d1) - 1
      return delta
    
  def calculate_gamma(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the gamma of the option fed into the class, using the Black-Scholes formula.
    The gamma is calculated as the second derivative of the option price with respect to the price of the underlying asset.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating gamma")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma
  
  def calculate_vega(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the vega of the option fed into the class, using the Black-Scholes formula.
    The vega is calculated as the first derivative of the option price with respect to the implied volatility.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating vega")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega
  
  def calculate_vanna(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the vanna of the option fed into the class, using the Black-Scholes formula.
    The vanna is calculated as the first derivative of the delta with respect to the implied volatility.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating vanna")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    vanna = -norm.pdf(d1) * d1 / sigma
    return vanna
  
  def calculate_theta(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the theta of the option fed into the class, using the Black-Scholes formula.
    The theta is calculated as the first derivative of the option price with respect to time.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating theta")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    dividends = self.fetcher.get_dividend_yield()
    r_adj = r - dividends
    d1 = (np.log(S / K) + (r_adj + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if self.call:
        theta = - (S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = - (S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)    
    theta_per_day = theta / 365
    return theta_per_day
  
  def calculate_rho(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the rho of the option fed into the class, using the Black-Scholes formula.
    The rho is calculated as the first derivative of the option price with respect to the risk-free rate.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating rho")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if self.call:
      rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
      rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return rho
  
  def calculate_vomma(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the vomma of the option fed into the class, using the Black-Scholes formula.
    The vomma is calculated as the second derivative of the option price with respect to the implied volatility.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating vomma")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vomma = S * np.sqrt(T) * norm.pdf(d1) * (d1 * d2) / sigma
    return vomma
  
  def calculate_charm(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the charm of the option fed into the class, using the Black-Scholes formula.
    The charm is calculated as the first derivative of the delta with respect to time.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating charm")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    charm = -norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    return charm
  
  def calculate_speed(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the speed of the option fed into the class, using the Black-Scholes formula.
    The speed is calculated as the third derivative of the option price with respect to the price of the underlying asset.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating speed")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    speed = -norm.pdf(d1) / (S**2 * sigma * np.sqrt(T)) * (d1 + sigma * np.sqrt(T))
    return speed
  
  def calculate_zomma(self, risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the zomma of the option fed into the class, using the Black-Scholes formula.
    The zomma is calculated as the derivative of the gamma with respect to the price of the underlying asset.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating zomma")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    zomma = norm.pdf(d1) * (d1 * d2 - 1) / (S * sigma**2 * np.sqrt(T))
    return zomma
  
  def calculate_color(self, risk_free_rate_horizon: 'str' = '13 weeks') -> float:
    """
    Method to calculate the color of the option fed into the class, using the Black-Scholes formula.
    The color is calculated as the second derivative of the charm with respect to the price of the underlying asset.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating color")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    color = -norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma**2 * np.sqrt(T))
    return color

  def calculate_ultima(self, risk_free_rate_horizon: 'str' = '13 weeks') -> float:
    """
    Method to calculate the ultima of the option fed into the class, using the Black-Scholes formula.
    The ultima is calculated as the second derivative of the vomma with respect to the price of the underlying asset.
    Inputs:
    - risk_free_rate_horizon: str, the horizon of the risk free rate
    """
    if self.verbose:
      print("Calculating ultima")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]
    K = option['strike']
    T = (self.date - pd.to_datetime('today')).days / 365
    r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
    sigma = option['impliedVolatility']
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    ultima = S * np.sqrt(T) * norm.pdf(d1) * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2) / sigma**2
    return ultima
  
  def calculate_lambda(self,risk_free_rate_horizon: str = '13 weeks') -> float:
    """
    Method to calculate the lambda (leverage) of the option fed into the class.
    Lambda is calculated as the product of Delta and the ratio of the underlying asset's price to the option's price.
    """
    if self.verbose:
        print("Calculating lambda")
    option = self.option
    dataframe = self.df
    S = dataframe['Close'].iloc[-1]  
    V = option['lastPrice'] 
    delta = self.calculate_delta(risk_free_rate_horizon)  

    lambda_ = delta * (S / V)
    return lambda_


class OptionPricing:
  def __init__(self, 
               ticker: str,
               call: bool,
               identification: Union[str, int],
               verbose: bool = False):
    self.fetcher = Fetcher(ticker)
    self.df, self.yf_ticker = self.fetcher.df, self.fetcher.yf_stock
    if type(identification) == str:
     self.option_chain = self.yf_ticker.option_chain().calls if call else self.yf_ticker.option_chain().puts
     self.option = self.option_chain[self.option_chain['contractSymbol'] == identification]
    elif type(identification) == int:
      self.option = self.yf_ticker.option_chain().calls.iloc[identification] if call else self.yf_ticker.option_chain().puts.iloc[identification]
    else:
      raise ValueError('identification must be either a string or an integer')
    self.verbose = verbose
    self.date = self.calculate_expiration()
    self.call = call

  def from_name_to_datestr(self, s: str) -> str:
    """Helper function to convert the contract symbol to a date string"""
    if self.verbose:
      print(f"Processing {s}")
    match = re.search(r'[A-Za-z](\d{2})(\d{2})(\d{2})[CP]', s)
    if match:
        year, month, day = match.groups()
        return f"20{year}-{month}-{day}"
    else:
        return "No date found"
  
  def calculate_expiration(self) -> pd.Timestamp:
    """Function to calculate the expiration date of the option"""
    if self.verbose:
      print("Calculating expiration date")
    to_process = self.option['contractSymbol']
    date_str = self.from_name_to_datestr(to_process)
    date = pd.to_datetime(date_str)
    return date  
  
  def pricing_formula(self, price_of_underlying_asset:float = None, risk_free_rate: Union[str, float] = '13 weeks', use_black_scholes: bool = True, **kwargs):
        """
        Calculate the price of an option using user-defined pricing components passed as keyword arguments.
        Inputs:
        - price_of_underlying_asset: float, the price of the underlying asset. If None, the last price in the dataframe will be used.
        - risk_free_rate: Union[str, float], the risk free rate to use in the calculations, either a string representing the horizon or a float.
        - use_black_scholes: bool, whether to use the Black-Scholes formula to calculate the price of the option.
        - **kwargs: dict, the keyword arguments containing the custom pricing components. Note: they will not be used if use_black_scholes is True.
    

        Each keyword argument should be a function that only accept as argument the price of the underlying asset and the risk free rate, a float. They will not be validated, so make sure they are correctly implemented.
        Each function will be simply summed, so each transformation should be done within the function itself.

        Example:
        def my_custom_pricing_component(S, r):
          return S * (-np.log(r * 0.5))
        
        option = OptionPricing('AAPL', True, 'AAPL220121C00100000')
        option.pricing_formula(price_of_underlying_asset = 100, risk_free_rate = 0.12, custom_pricing_component = my_custom_pricing_component)
        """
        if price_of_underlying_asset is None:
          S = self.df['Close'].iloc[-1]
        else:
          S = price_of_underlying_asset
        if self.verbose:
            print("Calculating option price")
        price_components = []
        if type(risk_free_rate) == str:
          r = self.fetcher.get_risk_free_rate(risk_free_rate_horizon)['Close'].iloc[-1]
        else:
          r = risk_free_rate
        if use_black_scholes:
          K = self.option['strike']
          T = (self.date - pd.to_datetime('today')).days / 365
          sigma = self.option['impliedVolatility']
          r = r
          d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
          d2 = d1 - sigma * np.sqrt(T)
          if self.call:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
          else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
          return price
        else:
          for _, func in kwargs.items():
              try:
                price_component = func(r)
                price_components.append(price_component)
              except Exception as e:
                print(f"Failed to calculate {func} due to error: {e}")

          final_price = sum(price_components)
          return final_price
  
  def calculate_transition_matrix(self, num_states = 20):
    daily_returns = self.df['Close'].pct_change().dropna()
    bounds = np.linspace(daily_returns.min(), daily_returns.max(), num_states + 1)
    states = np.digitize(daily_returns, bounds) - 1
    transition_matrix = np.zeros((num_states, num_states))
    for (i, j) in zip(states[:-1], states[1:]):
        transition_matrix[i, j] += 1
    transition_matrix /= transition_matrix.sum(axis = 1)[:, None]
    return transition_matrix
  
  def simulate_price_paths(self, start_price, days_to_expiration, transition_matrix, num_simulations = 1000):
    state_space = np.arange(len(transition_matrix))        
    price_paths = np.zeros((num_simulations, days_to_expiration))
    for sim in range(num_simulations):
      current_state = np.searchsorted(state_space, start_price)
      price_path = [start_price]
      for _ in range(1, days_to_expiration):
        current_state = np.random.choice(state_space, p=transition_matrix[current_state])
        price_path.append(state_space[current_state])
      price_paths[sim] = price_path
    return price_paths

  def calculate_markov_chain(self, r: float = None, plot: bool = False, num_simulations: int = 1000, transition_matrix: np.ndarray = None,  num_states: int = 20, **kwargs) -> float:
    """
    Calculate the expected value of an option using a Markov chain model. 
    Inputs:
    - r: float, the risk free rate to use in the calculations. If None, it will be automatically calculated.
    - plot: bool, whether to plot the simulated price paths.
    - num_simulations: int, the number of simulations to run.
    - transition_matrix: np.ndarray, the transition matrix to use in the simulations. If None, it will be calculated as a discretized version of the daily returns.
    - num_states: int, the number of states to use in the transition matrix.
    - **kwargs: dict, the keyword arguments containing the custom pricing components. Note: if none are passed, the Black-Scholes formula will be used.

    
    """
    if r is None:
      r = self.fetcher.get_risk_free_rate('13 weeks')['Close'].iloc[-1]
    else:
      r = r
    days_to_expiration = (self.date - pd.to_datetime('today')).days
    if transition_matrix is None:
      transition_matrix = self.calculate_transition_matrix(num_states)
    start_price = self.df['Close'].iloc[-1]
    price_paths = self.simulate_price_paths(start_price, days_to_expiration, transition_matrix, num_simulations)
    if plot:
        plt.figure(figsize = (12, 10))
        for path in price_paths:
            sns.lineplot(x = np.arange(len(path)), y = path, alpha = 0.1)
        plt.title('Simulated price paths')
        plt.xlabel('Days to expiration')
        plt.ylabel('Price')
        plt.show()
    option_values = [self.pricing_formula(price_of_underlying_asset = path[-1], risk_free_rate = r, **kwargs) for path in price_paths]    
    return np.mean(option_values)
    





## Yield curve/convexity bonds
## Futures pricing
## Commodities pricing
## CDS pricing
## credit spread of a company with risk free rate
