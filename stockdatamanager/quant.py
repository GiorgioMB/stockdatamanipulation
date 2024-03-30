from .datafetcher import Fetcher
import pandas as pd
import numpy as np
import yfinance as yf
import re
from typing import Union
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import arch
import optuna
from stockdatamanager.customerrors import MethodError

class Greeks:
  """
  Class to calculate the greeks for an option
  Inputs:
  - ticker: str, the ticker of the stock
  - call: bool, whether the option is a call or a put
  - identification: Union[str, int], the contract symbol of the option or the index of the option in the option chain
                                     Note: to find the contract identifier, it is written as ticker-YY-MM-DD-C(or P)-strike price (omit the - and the strike price is in a thousands format, so 1000 is $1.00)
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
  
  def pricing_formula(self, price_of_underlying_asset:float = None, risk_free_rate: Union[str, float] = '13 weeks', sigma: float = None, use_black_scholes: bool = True, **kwargs):
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
        if self.verbose:
            print("Calculating option price, using custom pricing components") if not use_black_scholes else print("Calculating option price using Black-Scholes formula")
        if price_of_underlying_asset is None:
          S = self.df['Close'].iloc[-1]
        else:
          S = price_of_underlying_asset
        if self.verbose:
            print("Calculating option price")
        price_components = []
        if type(risk_free_rate) == str:
          r = self.fetcher.get_risk_free_rate(risk_free_rate)['Close'].iloc[-1]
        else:
          r = risk_free_rate
        if use_black_scholes:
          K = self.option['strike']
          T = (self.date - pd.to_datetime('today')).days / 365
          if sigma is None:
            sigma = self.option['impliedVolatility']
          else:
            sigma = sigma
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
  
  def calculate_transition_matrix(self, num_states=20):
    if self.verbose:
      print("Calculating transition matrix")
    daily_returns = self.df['Close'].pct_change().dropna()
    bounds = np.linspace(daily_returns.min(), daily_returns.max(), num_states + 1)
    states = np.digitize(daily_returns, bounds) - 1
    states = np.where(states >= num_states, num_states - 1, states)
    transition_matrix = np.zeros((num_states, num_states))
    for (i, j) in zip(states[:-1], states[1:]):
        transition_matrix[i, j] += 1    
    row_sums = transition_matrix.sum(axis=1)[:, None]
    uniform_prob = np.ones(num_states) / num_states
    for i, row_sum in enumerate(row_sums):
        if row_sum == 0:
            transition_matrix[i] = uniform_prob
        else:
            transition_matrix[i] /= row_sum
    if self.verbose:
      print("Transition matrix calculated")
    return transition_matrix
 
  def simulate_price_paths(self, start_price, days_to_expiration, transition_matrix, num_simulations=1000):
    if self.verbose:
      print("Simulating price paths")
    num_states = len(transition_matrix)
    price_paths = np.zeros((num_simulations, days_to_expiration))
    daily_returns = self.df['Close'].pct_change().dropna()
    returns_bounds = np.linspace(daily_returns.min(), daily_returns.max(), num_states + 1)
    returns = (returns_bounds[:-1] + returns_bounds[1:]) / 2 
    for sim in range(num_simulations):
        if self.verbose:
          print(f"Simulating path {sim + 1}/{num_simulations}")
        current_price = start_price
        price_paths[sim, 0] = current_price
        current_state_index = np.digitize(current_price, returns_bounds) - 1
        current_state_index = min(max(current_state_index, 0), num_states - 2)  # Ensure within bounds

        for day in range(1, days_to_expiration):
            if self.verbose:
              print(f"Day {day}/{days_to_expiration}")
            transition_probs = transition_matrix[current_state_index]
            next_state_index = np.random.choice(np.arange(num_states), p=transition_probs)
            chosen_return = returns[next_state_index]
            current_price *= (1 + chosen_return)
            price_paths[sim, day] = current_price            
            current_state_index = next_state_index

    return price_paths

  def calculate_option_value_with_markov_chain(self, r: float = None, plot: bool = False, num_simulations: int = 1000, transition_matrix: np.ndarray = None,  num_states: int = 20, use_garch: bool = False, optimize_garch: bool = False, **kwargs) -> float:
    """
    Calculate the expected value of an option using a Markov chain model. 
    Inputs:
    - r: float, the risk free rate to use in the calculations. If None, it will be automatically calculated.
    - plot: bool, whether to plot the simulated price paths.
    - num_simulations: int, the number of simulations to run.
    - transition_matrix: np.ndarray, the transition matrix to use in the simulations. If None, it will be calculated as a discretized version of the daily returns.
    - num_states: int, the number of states to use in the transition matrix.
    - use_garch: bool, whether to use a GARCH model to simulate the volatility.
    - optimize_garch: bool, whether to optimize the GARCH model. 
    - **kwargs: dict, the keyword arguments containing the custom pricing components. Note: if none are passed, the Black-Scholes formula will be used.
    """
    if self.verbose:
      print("Calculating option value with Markov chain model")
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
        if self.verbose:
          print("Plotting simulated price paths")
        plt.figure(figsize = (12, 10))
        for path in price_paths:
            sns.lineplot(x = np.arange(len(path)), y = path, alpha = 0.1)
        plt.title('Simulated price paths')
        plt.xlabel('Days to expiration')
        plt.ylabel('Price')
        plt.show()
    option_values = []
    if self.verbose:
      print("Calculating option values")
    if use_garch:
      if optimize_garch:
        garch_params = self.optimize_garch_model()
        self.build_garch_model(**garch_params)
      else:
        self.build_garch_model('GARCH', 'Constant', 1, 1, 'normal')
    for path in price_paths:
      if use_garch:
        res = self.garch_model.fit(disp='off')
        sigma = res.conditional_volatility
        sigma = sigma[-1]
        single_value = self.pricing_formula(price_of_underlying_asset = path[-1], risk_free_rate = r, sigma = sigma, **kwargs)
      else:
        single_value = self.pricing_formula(price_of_underlying_asset = path[-1], risk_free_rate = r, **kwargs)
      option_values.append(single_value)
    return np.mean(option_values)
  
  def build_garch_model(self, vol, mean, p, q, dist):
    if self.verbose: print("Building GARCH model")
    target_val = self.df['Close'].pct_change().dropna() * 100
    self.garch_model = arch.arch_model(target_val, vol = vol, mean = mean, p=p, q=q, dist=dist)
    if self.verbose: print("GARCH model built")

  def optimize_garch_model(self) -> dict:
    if self.verbose: print("Optimizing GARCH model")
    def objective(trial):
        p = trial.suggest_int('p', 1, 10)
        q = trial.suggest_int('q', 1, 10)
        vol = trial.suggest_categorical('vol', ['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'HARCH'])
        mean = trial.suggest_categorical('mean', ['Constant', 'Zero', 'LS', 'AR', 'ARX', 'HAR', 'HARX', 'constant', 'zero'])
        dist = trial.suggest_categorical('dist', ['normal', 'gaussian', 't', 'studentst', 'skewstudent', 'skewt', 'ged', 'generalized error'])
        if self.verbose: print(f"Starting trial with parameters {p}, {q}, {vol}, {mean}, {dist}")
        self.build_garch_model(vol, mean, p, q, dist)
        res = self.garch_model.fit(disp='off')
        return res.aic
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    if self.verbose: print("Optimization complete")
    return study.best_params
  
  def calculate_american_style_option_prices(self, method: str = 'binomial', num_steps: int = 100, use_garch: bool = False, optimize_garch: bool = False, M:int = 200, N:int = 200, tol:int = 1e-7, omega:int = 1.1) -> float:
    """
    Calculate the price of an American style option using a binomial tree or a Monte Carlo simulation.
    Inputs:
    - method: str, the method to use for the calculation. Either 'binomial' or 'finite_difference'
    - num_steps: int, the number of steps to use in the binomial tree or discretized partial differential equation.
    - use_garch: bool, whether to use a GARCH model to simulate the volatility.
    - optimize_garch: bool, whether to optimize the GARCH model.
    - M, N: Grid size for the discretized partial differential equation.
    - tol: Tolerance for convergence in the discretized partial differential equation.
    - omega: Relaxation factor
    """
    if method == 'binomial':
      S = self.df['Close'].iloc[-1]
      K = self.option['strike']
      T = (self.date - pd.to_datetime('today')).days / 365.0
      r = self.fetcher.get_risk_free_rate('13 weeks')['Close'].iloc[-1]
      if use_garch:
        if optimize_garch:
          garch_params = self.optimize_garch_model()
          self.build_garch_model(**garch_params)
        res = self.garch_model.fit(disp='off')
        sigma = res.conditional_volatility
        sigma = sigma[-1]
      else:
        sigma = self.option['impliedVolatility']
      dt = T / num_steps 
      u = np.exp(sigma * np.sqrt(dt))
      d = 1 / u 
      q = (np.exp(r * dt) - d) / (u - d)  
      prices = np.zeros((num_steps + 1, num_steps + 1))
      option = np.zeros((num_steps + 1, num_steps + 1))
      for i in range(num_steps + 1):
        for j in range(i + 1):
          prices[j, i] = S * (u ** j) * (d ** (i - j))
      for i in range(num_steps + 1):
        option[i, num_steps] = max(0, prices[i, num_steps] - K if self.call else K - prices[i, num_steps])
        
        # Calculate option value at each node, considering early exercise
        for i in range(num_steps - 1, -1, -1):
            for j in range(i + 1):
              option[j, i] = max((q * option[j, i + 1] + (1 - q) * option[j + 1, i + 1]) * np.exp(-r * dt), 
                                 prices[j, i] - K if self.call else K - prices[j, i])
        return option[0, 0]
    elif 'finite_difference':
      raise MethodError("Method not implemented yet, as it requires a more complex implementation than what I am capable of.")
    else:
      raise MethodError("Method not recognized. Please use either 'binomial' or 'finite_difference'.")
## Yield curve/convexity bonds
## Futures pricing
## Commodities pricing
## CDS pricing
## credit spread of a company with risk free rate
