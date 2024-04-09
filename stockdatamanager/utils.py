import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class _SMA(object):
    def __init__(self, target, window = 5):
        self.target = target
        self.window = window
        if not isinstance(self.target, pd.Series):
            self.target = pd.Series(self.target)
        if len(self.target) < self.window:
            raise ValueError("The window size is larger than the target size")
        if self.window < 1:
            raise ValueError("The window size must be greater than 0")
        if self.target.isna().sum() > 0:
            raise ValueError("The target contains missing values, please fill them before using this class")
    
    def predict_val(self, num_steps = 1):
        values = self.target.rolling(window = self.window).mean().shift(-1)
        return np.mean(values[-num_steps:])

class _EMA(object):
    def __init__(self, target, window = 5):
        self.target = target
        self.window = window
        if not isinstance(self.target, pd.Series):
            self.target = pd.Series(self.target)
        if len(self.target) < self.window:
            raise ValueError("The window size is larger than the target size")
        if self.window < 1:
            raise ValueError("The window size must be greater than 0")
        if self.target.isna().sum() > 0:
            raise ValueError("The target contains missing values, please fill them before using this class")
    
    def predict_val(self, num_steps = 1):
        values = self.target.ewm(span = self.window).mean().shift(-1).fillna(0)
        return np.mean(values[-num_steps:])

class _SARIMAX_model(object):
    def __init__(self, target: pd.Series, exog: pd.Series = None, optimize: bool = False, 
                 optimization_rounds: int = 100, test_size: float = 0.2, random_state: int = 42, verbose: bool = False,
                 p: int = 1, d: int = 1, q: int = 1, P: int = 1, D: int = 1, Q: int = 1, s: int = 1, trend: str = 'n',
                 measure_error: bool = False, time_varying_regression: bool = False, mle_regression: bool = False,
                 simple_differencing: bool = False, enforce_stationarity: bool = False, enforce_invertibility: bool = False,
                 hamilton_representation: bool = False, concentrate_scale: bool = False, trend_offset: int = 1,
                 use_exact_diffuse: bool = False):
        """
        Utility class to fit SARIMAX models with exogenous variables, optimize the hyperparameters and predict values
        Inputs:
            - target: pd.Series, target variable to predict
            - exog: pd.Series, exogenous variable to use in the model
            - optimize: bool, whether to optimize the hyperparameters
            - optimization_rounds: int, number of rounds to optimize the hyperparameters
            - test_size: float, size of the test set
            - random_state: int, random state for reproducibility
            - p: int, AR order
            - d: int, differencing order
            - q: int, MA order
            - P: int, seasonal AR order
            - D: int, seasonal differencing order
            - Q: int, seasonal MA order
            - s: int, seasonal period
            - trend: str, trend parameter
            - measure_error: bool, whether to measure the error
            - time_varying_regression: bool, whether to use time varying regression
            - mle_regression: bool, whether to use MLE regression
            - simple_differencing: bool, whether to use simple differencing
            - enforce_stationarity: bool, whether to enforce stationarity
            - enforce_invertibility: bool, whether to enforce invertibility
            - hamilton_representation: bool, whether to use Hamilton representation
            - concentrate_scale: bool, whether to concentrate scale
            - trend_offset: int, trend offset
            - use_exact_diffuse: bool, whether to use exact diffuse
        """
        self.assert_types(target, exog, optimize, optimization_rounds, test_size, random_state, p, d, q, P, D, Q, s, trend,
                            measure_error, time_varying_regression, mle_regression, simple_differencing, enforce_stationarity,
                            enforce_invertibility, hamilton_representation, concentrate_scale, trend_offset, use_exact_diffuse)
        self.target = target
        self.verbose = verbose
        self.exog = exog
        self.model = None
        if exog is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.exog, self.target, test_size = test_size, random_state = random_state)
        else:
            _, _, self.y_train, self.y_test = train_test_split(np.zeros(len(self.target)), self.target, test_size = test_size, random_state = random_state)
        if optimize:
            self.best_params = self.optimize(optimization_rounds)
            self.init_model(self.best_params['p'], self.best_params['q'], self.best_params['d'], self.best_params['D'],
                            self.best_params['P'], self.best_params['Q'], self.best_params['s'], self.best_params['trend'],
                            self.best_params['measure_error'], self.best_params['time_varying_regression'], self.best_params['mle_regression'],
                            self.best_params['simple_differencing'], self.best_params['enforce_stationarity'], self.best_params['enforce_invertibility'],
                            self.best_params['hamilton_representation'], self.best_params['concentrate_scale'], self.best_params['trend_offset'],
                            self.best_params['use_exact_diffuse'])
        else:
            self.init_model(p, q, d, D, P, Q, s, trend, measure_error, time_varying_regression, mle_regression,
                            simple_differencing, enforce_stationarity, enforce_invertibility, hamilton_representation,
                            concentrate_scale, trend_offset, use_exact_diffuse)
    
    def assert_types(self, target: pd.Series, exog: pd.Series = None, optimize: bool = False, 
                 optimization_rounds: int = 100, test_size: float = 0.2, random_state: int = 42,
                 p: int = 1, d: int = 1, q: int = 1, P: int = 1, D: int = 1, Q: int = 1, s: int = 1, trend: str = 'n',
                 measure_error = False, time_varying_regression = False, mle_regression = False,
                 simple_differencing = False, enforce_stationarity = False, enforce_invertibility = False,
                 hamilton_representation = False, concentrate_scale = False, trend_offset = 1,
                 use_exact_diffuse = False):
        """
        Utility method to assert the types of the inputs
        """
        if not isinstance(target, pd.Series):
            raise ValueError("The target variable must be a pandas Series")
        if exog is not None:
            if not isinstance(exog, pd.Series):
                raise ValueError("The exogenous variable must be a pandas Series")
        if len(target) != len(exog):
            raise ValueError("The target and exogenous variables must have the same length")
        if not isinstance(optimize, bool):
            raise ValueError("The optimize parameter must be a boolean")
        if not isinstance(optimization_rounds, int):
            raise ValueError("The optimization_rounds parameter must be an integer")
        if not isinstance(test_size, float) or test_size > 1 or test_size < 0:
            raise ValueError("The test_size parameter must be a float between 0 and 1")
        if not isinstance(random_state, int):
            raise ValueError("The random_state parameter must be an integer")
        if not isinstance(p, int):
            raise ValueError("The p parameter must be an integer")
        if not isinstance(d, int):
            raise ValueError("The d parameter must be an integer")
        if not isinstance(q, int):
            raise ValueError("The q parameter must be an integer")
        if not isinstance(P, int):
            raise ValueError("The P parameter must be an integer")
        if not isinstance(D, int):
            raise ValueError("The D parameter must be an integer")
        if not isinstance(Q, int):
            raise ValueError("The Q parameter must be an integer")
        if not isinstance(s, int):
            raise ValueError("The s parameter must be an integer")
        if trend not in ['n', 'c', 't', 'ct']:
            raise ValueError("The trend parameter must be one of 'n', 'c', 't', 'ct'")
        if not isinstance(measure_error, bool):
            raise ValueError("The measure_error parameter must be a boolean")
        if not isinstance(time_varying_regression, bool):
            raise ValueError("The time_varying_regression parameter must be a boolean")
        if not isinstance(mle_regression, bool):
            raise ValueError("The mle_regression parameter must be a boolean")
        if not isinstance(simple_differencing, bool):
            raise ValueError("The simple_differencing parameter must be a boolean")
        if not isinstance(enforce_stationarity, bool):
            raise ValueError("The enforce_stationarity parameter must be a boolean")
        if not isinstance(enforce_invertibility, bool):
            raise ValueError("The enforce_invertibility parameter must be a boolean")
        if not isinstance(hamilton_representation, bool):
            raise ValueError("The hamilton_representation parameter must be a boolean")
        if not isinstance(concentrate_scale, bool):
            raise ValueError("The concentrate_scale parameter must be a boolean")
        if not isinstance(trend_offset, int):
            raise ValueError("The trend_offset parameter must be an integer")
        if not isinstance(use_exact_diffuse, bool):
            raise ValueError("The use_exact_diffuse parameter must be a boolean")
        
    def init_model(self, p, q, d, D, P, Q, s, 
                   trend, measurement_error, 
                   time_varying_regression, 
                   mle_regression, simple_differencing, 
                   enforce_stationarity, enforce_invertibility, 
                   hamilton_representation, concentrate_scale, 
                   trend_offset, use_exact_diffuse):
        """
        Method to initialize the SARIMAX model and fit it to the data
        """
        if self.verbose:
            print("Initializing model")
        if self.X_train is not None:
            self.model = SARIMAX(self.y_train, self.X_train, order = (p, d, q), seasonal_order = (P, D, Q, s), trend = trend, 
                                 measurement_error = measurement_error, time_varying_regression = time_varying_regression,
                                 mle_regression = mle_regression, simple_differencing = simple_differencing,
                                 enforce_stationarity = enforce_stationarity, enforce_invertibility = enforce_invertibility,
                                 hamilton_representation = hamilton_representation, concentrate_scale = concentrate_scale,
                                 trend_offset = trend_offset, use_exact_diffuse = use_exact_diffuse)
        else:
            self.model = SARIMAX(self.y_train, order = (p, d, q), seasonal_order = (P, D, Q, s), trend = trend, 
                                 measurement_error = measurement_error, time_varying_regression = time_varying_regression,
                                 mle_regression = mle_regression, simple_differencing = simple_differencing,
                                 enforce_stationarity = enforce_stationarity, enforce_invertibility = enforce_invertibility,
                                 hamilton_representation = hamilton_representation, concentrate_scale = concentrate_scale,
                                 trend_offset = trend_offset, use_exact_diffuse = use_exact_diffuse)
        self.model.fit()    
    
    def optimize(self, rounds):
        print("Warning: This feature is experimental, it may take a long time to run, as some models may be invalid or take a long time to fit")
        def objective(trial):
            p = trial.suggest_int('p', 0, 5)
            q = trial.suggest_int('q', 0, 5)
            d = trial.suggest_int('d', 0, 5)
            P = trial.suggest_int('P', 0, 5)
            Q = trial.suggest_int('Q', 0, 5)
            D = trial.suggest_int('D', 0, 5)
            s = trial.suggest_int('s', 0, 5)
            trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])
            measure_error = trial.suggest_categorical('measure_error', [True, False])
            time_varying_regression = trial.suggest_categorical('time_varying_regression', [True, False])
            mle_regression = trial.suggest_categorical('mle_regression', [True, False])
            simple_differencing = trial.suggest_categorical('simple_differencing', [True, False])
            enforce_stationarity = trial.suggest_categorical('enforce_stationarity', [True, False])
            enforce_invertibility = trial.suggest_categorical('enforce_invertibility', [True, False])
            hamiltion_representation = trial.suggest_categorical('hamilton_representation', [True, False])
            concentrate_scale = trial.suggest_categorical('concentrate_scale', [True, False])
            trend_offset = trial.suggest_int('trend_offset', 0, 5)
            use_exact_diffuse = trial.suggest_categorical('use_exact_diffuse', [True, False])
            if self.verbose:
                print("Trying parameters: ", p, q, d, D, P, Q, s, trend, measure_error, time_varying_regression)
                print(mle_regression, simple_differencing, enforce_stationarity, enforce_invertibility, hamiltion_representation, concentrate_scale, trend_offset, use_exact_diffuse)
            try:
                self.init_model(p, q, d, D, P, Q, s, trend, measure_error, time_varying_regression, mle_regression,
                                simple_differencing, enforce_stationarity, enforce_invertibility, hamiltion_representation,
                                concentrate_scale, trend_offset, use_exact_diffuse)
                if self.X_test is not None:
                    prediction = self.model.forecast(steps = len(self.X_test), exog = self.X_test)
                else:
                    prediction = self.model.forecast(steps = len(self.y_test))
                if self.verbose:
                    print("The model was fitted successfully, the ROC AUC score is:", roc_auc_score(self.y_test, prediction))
                return roc_auc_score(self.y_test, prediction)
            except Exception as e:
                print("The model failed to fit, error: ", e)
                return 0
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objective, n_trials = rounds)
        return study.best_params
    
    def forecast(self, steps: int = 1):
        """
        Method to forecast values
        Inputs:
            - steps: int, number of steps to forecast
        """
        return self.model.forecast(steps = steps)
