from stockdatamanager.customerrors import MethodError
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
import math
from abc import ABC, abstractmethod

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

class Option(object):
    def __init__(self, 
                 S, 
                 K, 
                 r, 
                 T, 
                 N, 
                 pu=0, 
                 pd_=0, 
                 div=0, 
                 sigma=0, 
                 call=True, 
                 american=False):
        """
        Initialize the stock option base class. Defaults to European call unless specified.
        Inputs: 
          - S: initial stock price
          - K: strike price
          - r: risk-free interest rate
          - T: time to maturity (in years)
          - N: number of time steps
          - pu: probability at up state
          - pd: probability at down state
          - div: Dividend yield
          - call: True for a call option, False for a put option
          - american: True for an American option,
                False for a European option
        """
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1, N)
        self.STs = [] 
        self.pu, self.pd_= pu, pd_
        self.div = div
        self.sigma = sigma
        self.is_call = call
        self.is_european = american

    @property
    def dt(self):
        """ Single time step, in years """
        return self.T/float(self.N)

    @property
    def df(self):
        """ The discount factor """
        return math.exp(-(self.r-self.div)*self.dt)  

class FiniteDifferences(object):
    def __init__(self, 
                 S, 
                 K, 
                 r, 
                 T, 
                 sigma, 
                 Smax = 1, 
                 M = 1, 
                 N = 1, 
                 call = True):
        """This class is the base class for the finite difference methods, for you it's useless"""
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.Smax = Smax
        self.M, self.N = M, N
        self.is_call = not is_put
        self.i_values = np.arange(self.M)
        self.j_values = np.arange(self.N)
        self.grid = np.zeros(shape=(self.M+1, self.N+1))
        self.boundary_conds = np.linspace(0, Smax, self.M+1)

    @property
    def dS(self):
        return self.Smax/float(self.M)

    @property
    def dt(self):
        return self.T/float(self.N)

    @abstractmethod
    def setup_boundary_conditions(self):
        raise MethodError('What did I tell you? Useless')

    @abstractmethod
    def setup_coefficients(self):
        raise MethodError("Womp womp, you're not supposed to use this")

    @abstractmethod
    def traverse_grid(self):
        """  
        Iterate the grid backwards in time 
        """
        raise MethodError("This thing is useless, why are you even trying to use it?")

    @abstractmethod
    def interpolate(self):
        """
        Use piecewise linear interpolation on the initial
        grid column to get the closest price at S0.
        """
        return np.interp(
            self.S, self.boundary_conds, self.grid[:,0])

    def price(self):
        self.setup_boundary_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()

class OptionPricing(object):
  def __init__(self, 
               ticker: str,
               call: bool,
               american: bool,
               risk_free_rate: Union[str, float],
               identification: Union[str, int],
               use_yfinance_volatility: bool = True,
               optimize_garch: bool = False,
               verbose: bool = False):
    self.fetcher = Fetcher(ticker)
    self.dataframe, self.yf_ticker = self.fetcher.df, self.fetcher.yf_stock
    if type(identification) == str:
      self.option_chain = self.yf_ticker.option_chain().calls if call else self.yf_ticker.option_chain().puts
      self.option = self.option_chain[self.option_chain['contractSymbol'] == identification]
    elif type(identification) == int:
      self.option = self.yf_ticker.option_chain().calls.iloc[identification] if call else self.yf_ticker.option_chain().puts.iloc[identification]
    else:
      raise ValueError('identification must be either a string or an integer')
    self.verbose = verbose
    self.date = self.calculate_expiration()
    self.is_call = call
    self.dividend_yield = self.fetcher.get_dividend_yield()
    if type(risk_free_rate) == str:
      self.risk_free_rate = self.fetcher.get_risk_free_rate(risk_free_rate)['Close'].iloc[-1]
    elif type(risk_free_rate) == float or type(risk_free_rate) == int:
      self.risk_free_rate = risk_free_rate
    else:
      raise ValueError('risk_free_rate must be either a string representing the horizon or a value')
    self.S = self.dataframe['Close'].iloc[-1]
    self.T = (self.date - pd.to_datetime('today')).days / 365
    self.K = self.option['strike']
    self.american = american
    if use_yfinance_volatility:
      self.sigma = self.option['impliedVolatility']
    else:
      if optimize_garch:
        garch_params = self.optimize_garch_model()
        self.build_garch_model(**garch_params)
      else:
        self.build_garch_model('GARCH', 'Constant', 1, 1, 'normal')
      res = self.garch_model.fit(disp='off')
      sigma = res.conditional_volatility
      self.sigma = sigma[-1]

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
  
  def calculate_option_price(self, method: str, describe: bool, **kwargs) -> float:
    """
    Main wrapper function to calculate the price of an option using the specified method.
    Inputs:
      - method: str, the method to use for pricing
      - describe: bool, whether to describe the method
      - kwargs: additional arguments to be passed to the method
    Returns:
      - price: float, the price of the option
    """
    ##Remove spaces and upper cases from the method
    method = method.replace(' ', '').lower()
    if method == 'binomialtree' or 'binomial':
      if 'N' not in kwargs:
        print("N not passed, using default value of 1000")
      if 'pd_' not in kwargs:
        print("pd_ not passed, using default value of 0")
      if 'pu' not in kwargs:
        print("pu not passed, using default value of 0")
      if describe:
        print("The Binomial Tree Method requires three optional values,")
        print("N, which is the number of time steps")
        print("pd_, which is the probability at the down state")
        print("pu, which is the probability at the up state")
      if self.american:
        price = self.binom_american(N, pd_, pu)
      else:
        price = self.binom_european(N, pd_, pu)
      return price
    if method == 'cox-ross-rubinstein' or 'coxrossrubinstein' or 'crr':
      if 'N' not in kwargs:
        print("N not passed, using default value of 1000")
      if 'pd_' not in kwargs:
        print("pd_ not passed, using default value of 0")
      if 'pu' not in kwargs:
        print("pu not passed, using default value of 0")
      if describe:
        print("The Cox-Ross-Rubinstein Method requires three optional values,")
        print("N, which is the number of time steps")
        print("pd_, which is the probability at the down state")
        print("pu, which is the probability at the up state")
      if self.american:
        price = self.us_cox_ross_rubinstein(N, pd_, pu)
      else:
        price = self.eu_cox_ross_rubinstein(N, pd_, pu)
      return price
    if method == 'leisen-reimertree' or 'lrtee':
      if 'N' not in kwargs:
        print("N not passed, using default value of 1000")
      if 'pd_' not in kwargs:
        print("pd_ not passed, using default value of 0")
      if 'pu' not in kwargs:
        print("pu not passed, using default value of 0")
      if describe:
        print("The Leisen-Reimer Tree Method requires three optional values,")
        print("N, which is the number of time steps")
        print("pd_, which is the probability at the down state")
        print("pu, which is the probability at the up state")
      if self.american:
        price = self.us_lr_tree(N, pd_, pu)
      else:
        price = self.european_lr_tree(N, pd_, pu)
      return price
    if method == 'trinomialtree' or 'trinomial':
      if 'N' not in kwargs:
        print("N not passed, using default value of 1000")
      if 'pd_' not in kwargs:
        print("pd_ not passed, using default value of 0")
      if 'pu' not in kwargs:
        print("pu not passed, using default value of 0")
      if describe:
        print("The Trinomial Tree Method requires three optional values,")
        print("N, which is the number of time steps")
        print("pd_, which is the probability at the down state")
        print("pu, which is the probability at the up state")
      if self.american:
        price = self.trinom_american(N, pd_, pu)
      else:
        price = self.trinom_european(N, pd_, pu)
      return price
    if method == 'finitedifference' or 'finite_difference':
      raise MethodError("Please specify between Explicit Finite Difference, Implicit Finite Difference and Crank-Nicolson Finite Difference")
    if method == 'explicitfinitedifference' or 'explicit' or 'explicitdifference' or 'explicitfinite':
      if self.american:
        pass
      else:
        pass
      return price
    if method == 'implicitfinitedifference' or 'implicit' or 'implicitdifference' or 'implicitfinite':
      if self.american:
        pass
      else:
        pass
      return price
    if method == 'crank-nicolsonfinitedifference' or 'crank-nicolson' or 'cranknicolson' or 'crank-nicolsonfinite' or 'crank-nicolsondifference' or 'cranknicolsonfinite' or 'cranknicolsondifference':
      if self.american:
        pass
      else:
        pass
      return price
    raise MethodError(f"Method {method} not found")

  def trinom_american(self, N: int = 100, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    u = np.exp(Option.sigma * np.sqrt(2 * Option.dt))
    d = 1 / u
    m = 1
    qu = ((np.exp((Option.r - Option.div) * Option.dt / 2) - np.exp(-Option.sigma * np.sqrt(Option.dt / 2))) / (np.exp(Option.sigma * np.sqrt(Option.dt / 2)) - np.exp(-Option.sigma * np.sqrt(Option.dt / 2)))) ** 2
    qd = ((np.exp(Option.sigma * np.sqrt(Option.dt / 2)) - np.exp((Option.r - Option.div) * Option.dt / 2)) / (np.exp(Option.sigma * np.sqrt(Option.dt / 2)) - np.exp(-Option.sigma * np.sqrt(Option.dt / 2)))) ** 2
    qm = 1 - qu - qd
    def init_stock_price_tree():
      Option.STs = [np.array([Option.S])]
      for i in range(N):
        prev_branches = Option.STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1]* m, prev_branches[-1]* d]))
        Option.STs.append(st)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def check_early(payoffs, node):
      if self.is_call:
        return np.maximum(payoffs, self.STs[node] - self.K)
      else:
        return np.maximum(payoffs, self.K - self.STs[node])
    def traverse_tree(payoffs):
      for i in reversed(range(N)):
        payoffs = check_early((payoffs[:-2] * qu + payoffs[1:-1] * qm + payoffs[2:] * qd) * Option.df, i)
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]

  def trinom_european(self, N: int = 100, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    u = np.exp(Option.sigma * np.sqrt(2 * Option.dt))
    d = 1 / u
    m = 1
    qu = ((np.exp((Option.r - Option.div) * Option.dt / 2) - np.exp(-Option.sigma * np.sqrt(Option.dt / 2))) / (np.exp(Option.sigma * np.sqrt(Option.dt / 2)) - np.exp(-Option.sigma * np.sqrt(Option.dt / 2)))) ** 2
    qd = ((np.exp(Option.sigma * np.sqrt(Option.dt / 2)) - np.exp((Option.r - Option.div) * Option.dt / 2)) / (np.exp(Option.sigma * np.sqrt(Option.dt / 2)) - np.exp(-Option.sigma * np.sqrt(Option.dt / 2)))) ** 2
    qm = 1 - qu - qd
    def init_stock_price_tree():
      Option.STs = [np.array([Option.S])]
      for i in range(N):
        prev_branches = Option.STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1]* m, prev_branches[-1]* d]))
        Option.STs.append(st)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def traverse_tree(payoffs):
      for i in reversed(range(N)):
        payoffs = (payoffs[:-2] * qu + payoffs[1:-1] * qm + payoffs[2:] * qd) * Option.df
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]
    
  def us_lr_tree(self, N: int = 1000, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    odd_N = N if N % 2 == 1 else N + 1
    def pp_2_inversion(z, n):
        return .5 + math.copysign(1, z) * math.sqrt(.25 - .25 * math.exp( -((z / (n + 1 / 3+ .1 / (n + 1))) ** 2.) * (n + 1/6)))
    d1 = (np.log(Option.S / Option.K) + ((Option.r - Option.div) + (Option.sigma ** 2) / 2) * Option.T) / (Option.sigma * np.sqrt(Option.T))
    d2 = (np.log(Option.S / Option.K) + ((Option.r - Option.div) - (Option.sigma ** 2) / 2) * Option.T) / (Option.sigma * np.sqrt(Option.T))
    pbar = pp_2_inversion(d1, odd_N)
    p = pp_2_inversion(d2, odd_N)
    u = 1 / Option.df * pbar / p
    d = (1/Option.df - p * u) / (1 - p)
    qu = p
    qd = 1 - p
    def init_stock_price_tree():
      Option.STs = [np.array([Option.S])]
      for i in range(N):
        prev_branches = Option.STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1]* d]))
        Option.STs.append(st)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def check_early(payoffs, node):
      if self.is_call:
        return np.maximum(payoffs, self.STs[node] - self.K)
      else:
        return np.maximum(payoffs, self.K - self.STs[node])
    def traverse_tree(payoffs):
      for i in reversed(range(N)):
        payoffs = check_early((payoffs[:-1] * qu + payoffs[1:] * qd) * Option.df, i)
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]

  def europen_lr_tree(self, N: int = 1000, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    odd_N = N if N % 2 == 1 else N + 1
    def pp_2_inversion(z, n):
        return .5 + math.copysign(1, z) * math.sqrt(.25 - .25 * math.exp( -((z / (n + 1 / 3+ .1 / (n + 1))) ** 2.) * (n + 1/6)))
    d1 = (np.log(Option.S / Option.K) + ((Option.r - Option.div) + (Option.sigma ** 2) / 2) * Option.T) / (Option.sigma * np.sqrt(Option.T))
    d2 = (np.log(Option.S / Option.K) + ((Option.r - Option.div) - (Option.sigma ** 2) / 2) * Option.T) / (Option.sigma * np.sqrt(Option.T))
    pbar = pp_2_inversion(d1, odd_N)
    p = pp_2_inversion(d2, odd_N)
    u = 1 / Option.df * pbar / p
    d = (1/Option.df - p * u) / (1 - p)
    qu = p
    qd = 1 - p
    def init_stock_price_tree():
      Option.STs = np.zeros(M)
      for i in range(M):
        Option.STs[i] = self.S * (u ** (N - i)) * (d ** i)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def traverse_tree(payoffs):
      for i in range(N):
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * Option.df
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]

  def eu_cox_ross_rubinstein(self, N: int = 1000, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    u = math.exp(Option.sigma * math.sqrt(Option.dt))
    d = 1 / u
    qu = (math.exp(Option.r - Option.div) - d) / (u - d)
    qd = 1 - qu
    def init_stock_price_tree():
      Option.STs = np.zeros(M)
      for i in range(M):
        Option.STs[i] = self.S * (u ** (N - i)) * (d ** i)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def traverse_tree(payoffs):
      for i in range(N):
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * Option.df
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]

  def us_cox_ross_rubinstein(self, N: int = 1000, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    u = math.exp(Option.sigma * math.sqrt(Option.dt))
    d = 1 / u
    qu = (math.exp(Option.r - Option.div) - d) / (u - d)
    qd = 1 - qu
    def init_stock_price_tree():
      Option.STs = [np.array([Option.S])]
      for i in range(N):
        prev_branches = Option.STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1]* d]))
        Option.STs.append(st)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def check_early(payoffs, node):
      if self.is_call:
        return np.maximum(payoffs, self.STs[node] - self.K)
      else:
        return np.maximum(payoffs, self.K - self.STs[node])
    def traverse_tree(payoffs):
      for i in reversed(range(N)):
        payoffs = check_early((payoffs[:-1] * qu + payoffs[1:] * qd) * Option.df, i)
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]

  def binom_american(self, N: int = 1000, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S,
                    self.K,
                    self.risk_free_rate,
                    self.T,
                    N,
                    pu,
                    pd_,
                    self.dividend_yield,
                    self.sigma,
                    self.is_call,
                    self.american)
    u = 1 + pu
    d = 1 - pd_
    qu = (math.exp(Option.r - Option.div) * Option.dt) - d / (u - d)
    qd = 1 - qu
    def init_stock_price_tree():
      Option.STs = [np.array([Option.S])]
      for i in range(N):
        prev_branches = Option.STs[-1]
        st = np.concatenate((prev_branches * u, [prev_branches[-1]* d]))
        Option.STs.append(st)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def check_early(payoffs, node):
      if self.is_call:
        return np.maximum(payoffs, self.STs[node] - self.K)
      else:
        return np.maximum(payoffs, self.K - self.STs[node])
    def traverse_tree(payoffs):
      for i in reversed(range(N)):
        payoffs = check_early((payoffs[:-1] * qu + payoffs[1:] * qd) * Option.df, i)
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]
  
  def binom_european(self, N: int = 1000, pd_: float = 0, pu: float = 0) -> float:
    Option = Option(self.S, 
                    self.K, 
                    self.risk_free_rate, 
                    self.T, 
                    N, 
                    pu, 
                    pd_, 
                    self.dividend_yield, 
                    self.sigma, 
                    self.is_call, 
                    self.american)
    M = N + 1
    u = 1 + pu
    d = 1 - pd_
    qu =(math.exp(Option.r - Option.div) * Option.dt) - d / (u - d)
    qd = 1 - qu
    def init_stock_price_tree():
      Option.STs = np.zeros(M)
      for i in range(M):
        Option.STs[i] = self.S * (u ** (N - i)) * (d ** i)
    def init_payoffs_tree():
      if self.is_call:
        return np.maximum(0, Option.STs - Option.K)
      else:
        return np.maximum(0, Option.K - Option.STs)
    def traverse_tree(payoffs):
      for i in range(N):
        payoffs = (payoffs[:-1] * qu + payoffs[1:] * qd) * Option.df
      return payoffs
    def begin_tree_traversal():
      payoffs = init_payoffs_tree()
      return traverse_tree(payoffs)
    init_stock_price_tree()
    payoff = begin_tree_traversal()
    return payoff[0]
  
  def build_garch_model(self, vol, mean, p, q, dist):
    if self.verbose: print("Building GARCH model")
    target_val = self.df['Close'].pct_change().dropna() * 100
    self.garch_model = arch.arch_model(target_val, 
                                       vol = vol, 
                                       mean = mean,
                                       p=p, 
                                       q=q, 
                                       dist=dist)
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
  

## Yield curve/convexity bonds
## Futures pricing
## Commodities pricing
## CDS pricing
## credit spread of a company with risk free rate
