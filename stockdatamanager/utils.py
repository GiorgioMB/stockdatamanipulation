"""
This utility module provides foundational classes and methods for financial time series analysis and prediction within the OptionPricing class. 
It encompasses a range of statistical models and machine learning techniques to assist in tasks such as risk-free rate approximation and volatility estimation.

Components:
- _SMA: Implements a Simple Moving Average model for short-term forecasting based on a sliding window approach.
- _EMA: Utilizes an Exponential Moving Average for weighted forecasting that prioritizes more recent observations.
- _SARIMAX_model: An advanced model for time series forecasting that includes seasonal adjustments and exogenous variables. This class also supports hyperparameter optimization using Optuna.
- _AutoEncoder: A neural network based on PyTorch that learns efficient representations for time series data and can be applied to volatility estimation.
"""
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from typing import Union


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
    
    def predict_val(self, num_steps = 1) -> float:
        rolling_mean = self.target.rolling(window=self.window).mean()
        final_result = rolling_mean.tail(num_steps)
        return np.mean(final_result)


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
    
    def predict_val(self, num_steps = 1) -> float:
        values = self.target.ewm(span = self.window).mean().tail(num_steps)
        return np.mean(values)

class _SARIMAX_model(object):
    def __init__(self, target: pd.Series, exog: pd.Series = None, optimize: bool = False, 
                 optimization_rounds: int = 100, test_size: float = 0.2, random_state: int = 42, verbose: bool = False,
                 p: int = 1, d: int = 1, q: int = 1, P: int = 1, D: int = 1, Q: int = 1, s: int = 2, trend: str = 'n',
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
        self.verbose = verbose
        self.assert_types(target, exog, optimize, optimization_rounds, test_size, random_state, p, d, q, P, D, Q, s, trend,
                            measure_error, time_varying_regression, mle_regression, simple_differencing, enforce_stationarity,
                            enforce_invertibility, hamilton_representation, concentrate_scale, trend_offset, use_exact_diffuse)
        self.target = target
        self.exog = exog
        self.model = None
        if exog is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.exog, self.target, test_size = test_size, random_state = random_state)
            if self.verbose:
                print("Data was split successfully into train and test sets")
                print("Samples:")
                print("X_train:", self.X_train.head())
                print("X_test:", self.X_test.head())
                print("y_train:", self.y_train.head())
                print("y_test:", self.y_test.head())
        else:
            _, _, self.y_train, self.y_test = train_test_split(np.zeros(len(self.target)), self.target, test_size = test_size, random_state = random_state)
            if self.verbose:
                print("Data was split successfully into train and test sets")
                print("Samples:")
                print("y_train:", self.y_train.head())
                print("y_test:", self.y_test.head())

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
                 p: int = 1, d: int = 1, q: int = 1, P: int = 1, D: int = 1, Q: int = 1, s: int = 2, trend: str = 'n',
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
        if self.verbose:
            print("All types were asserted successfully")
            print("Parameters:")
            print("target:", target)
            print("exog:", exog)
            print("optimize:", optimize)
            print("optimization_rounds:", optimization_rounds)
            print("test_size:", test_size)
            print("random_state:", random_state)
            print("p:", p)
            print("d:", d)
            print("q:", q)
            print("P:", P)
            print("D:", D)
            print("Q:", Q)
            print("s:", s)
            print("trend:", trend)
            print("measure_error:", measure_error)
            print("time_varying_regression:", time_varying_regression)
            print("mle_regression:", mle_regression)
            print("simple_differencing:", simple_differencing)
            print("enforce_stationarity:", enforce_stationarity)
            print("enforce_invertibility:", enforce_invertibility)
            print("hamilton_representation:", hamilton_representation)
            print("concentrate_scale:", concentrate_scale)
            print("trend_offset:", trend_offset)
            print("use_exact_diffuse:", use_exact_diffuse)



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
            print("Initializing model with parameters: ")
            print("p:", p)
            print("q:", q)
            print("d:", d)
            print("D:", D)
            print("P:", P)
            print("Q:", Q)
            print("s:", s)
            print("trend:", trend)
            print("measurement_error:", measurement_error)
            print("time_varying_regression:", time_varying_regression)
            print("mle_regression:", mle_regression)
            print("simple_differencing:", simple_differencing)
            print("enforce_stationarity:", enforce_stationarity)
            print("enforce_invertibility:", enforce_invertibility)
            print("hamilton_representation:", hamilton_representation)
            print("concentrate_scale:", concentrate_scale)
            print("trend_offset:", trend_offset)
            print("use_exact_diffuse:", use_exact_diffuse)
        if hasattr(self, 'X_train') and self.X_train is not None:
            self.model = SARIMAX(endog = self.y_train, exog = self.X_train, order = (p, d, q), seasonal_order = (P, D, Q, s), trend = trend, 
                                 measurement_error = measurement_error, time_varying_regression = time_varying_regression,
                                 mle_regression = mle_regression, simple_differencing = simple_differencing,
                                 enforce_stationarity = enforce_stationarity, enforce_invertibility = enforce_invertibility,
                                 hamilton_representation = hamilton_representation, concentrate_scale = concentrate_scale,
                                 trend_offset = trend_offset, use_exact_diffuse = use_exact_diffuse)
        else:
            self.model = SARIMAX(endog = self.y_train, order = (p, d, q), seasonal_order = (P, D, Q, s), trend = trend, 
                                 measurement_error = measurement_error, time_varying_regression = time_varying_regression,
                                 mle_regression = mle_regression, simple_differencing = simple_differencing,
                                 enforce_stationarity = enforce_stationarity, enforce_invertibility = enforce_invertibility,
                                 hamilton_representation = hamilton_representation, concentrate_scale = concentrate_scale,
                                 trend_offset = trend_offset, use_exact_diffuse = use_exact_diffuse)
        self.model = self.model.fit()
        if self.verbose:
            print("The model was fitted successfully")
    
    def optimize(self, rounds) -> dict:
        """
        Method to optimize the hyperparameters of the model using Optuna
        """
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
                if hasattr(self, 'X_test') and self.X_test is not None:
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
    
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Method to forecast values
        Inputs:
            - steps: int, number of steps to forecast
        """
        return self.model.forecast(steps = steps)

class _AutoEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, device: torch.device, verbose: bool, test_size = 0.2, random_state = 42):
        super(_AutoEncoder, self).__init__()
        self.init_params(input_size, hidden_size, num_layers, dropout, device, verbose, test_size, random_state)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Dropout(dropout))
        self.decoder = nn.Linear(hidden_size, input_size)  # output_size matches input_size for reconstruction
        self.volatility_predictor = nn.Linear(hidden_size, 1)  # Output size is 1 for volatility prediction
        self.device = device
        self.verbose = verbose
        self.to(device)

    def init_params(self, input_size, hidden_size, num_layers, dropout, device, verbose, test_size, random_state):
        if not isinstance(input_size, int):
            raise ValueError("The input_size parameter must be an integer")
        if not isinstance(hidden_size, int):
            raise ValueError("The hidden_size parameter must be an integer")
        if not isinstance(device, torch.device):
            raise ValueError("The device parameter must be a torch device")
        if not isinstance(verbose, bool):
            raise ValueError("The verbose parameter must be a boolean")
        if not isinstance(test_size, float) or test_size > 1 or test_size < 0:
            raise ValueError("The test_size parameter must be a float between 0 and 1")
        if not isinstance(random_state, int):
            raise ValueError("The random_state parameter must be an integer")
        if not isinstance(num_layers, int):
            raise ValueError("The num_layers parameter must be an integer")
        if not isinstance(dropout, float) or dropout > 1 or dropout < 0:
            raise ValueError("The dropout parameter must be a float between 0 and 1")
        self.test_size = test_size
        self.random_state = random_state
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers:
            x = torch.relu(layer(x))
        decoded = torch.relu(self.decoder(x))
        volatility = self.volatility_predictor(x)  # Linear output for volatility
        return decoded, volatility
    
    def predict(self, x):
        """
        Method to predict values
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def validate_data(self, X, y):
        """
        Checks if the input data is a torch tensor, if not converts it to a torch tensor and moves it to the device
        """
        if self.verbose:
            print("Validating data")
        if not isinstance(X, torch.Tensor):
            try:
                if isinstance(X, np.ndarray):
                    X = torch.from_numpy(X).float()
                elif isinstance(X, (pd.DataFrame, pd.Series)):
                    X = torch.from_numpy(X.values).float()
                else:
                    raise ValueError("The input data type is not supported")
            except Exception as e:
                raise ValueError("The input data type is not supported, error: ", e)
        if not isinstance(y, torch.Tensor):
            try:
                if isinstance(y, np.ndarray):
                    y = torch.from_numpy(y).float()
                elif isinstance(y, (pd.DataFrame, pd.Series)):
                    y = torch.from_numpy(y.values).float()
                else:
                    raise ValueError("The input data type is not supported")
            except Exception as e:
                raise ValueError("The input data type is not supported, error: ", e)
        return X.to(self.device), y.to(self.device)
    
    def window_data(self, X, window_size):
        """
        Window the data size to be [batch_size, window_size, 1]
        """
        if X.shape[0] < window_size:
            raise ValueError("The window size is larger than the input data size")
        if self.verbose:
            print("Windowing data")
        X = X.unfold(0, window_size, 1)
        return X
    
    def balanced_criterion(self, output, target, alpha = 0.5):
        """
        Method to calculate the balanced criterion
        formula: alpha * MSE(reconstruction) + (1 - alpha) * MSE(volatility)
        lower alpha means higher importance to the volatility prediction loss
        """
        output = list(output)
        target = list(target)
        output[0], target[0] = output[0].squeeze(-1), target[0]
        output[1], target[1] = output[1].squeeze(-1), target[1]

        criterion = nn.MSELoss()
        loss1 = criterion(output[0], target[0])
        loss2 = criterion(output[1], target[1])
        return alpha * loss1 + (1 - alpha) * loss2
    
    def fit(self, X: Union[torch.Tensor, pd.Series, pd.DataFrame, np.ndarray], 
            y: Union[torch.Tensor, pd.Series, pd.DataFrame, np.ndarray],
            learning_rate: float = 0.001, 
            epochs: int = 10, optimizer: torch.optim.Optimizer = None, 
            criterion: callable = None, batch_size: int = 32, 
            metrics: callable = mean_absolute_percentage_error, alpha: float = 0.5):
        """
        Method to train the model
        Inputs:
            - X: Union[torch.Tensor, pd.Series, pd.DataFrame, np.ndarray], input data
            - y: Union[torch.Tensor, pd.Series, pd.DataFrame, np.ndarray], target data
            - epochs: int, number of epochs, default is 10
            - optimizer: torch.nn.optim.Optimizer, optimizer to use, default is Adam
            - criterion: callable, loss function, default is balanced_criterion (alpha * MSE(reconstruction) + (1 - alpha) * MSE(volatility))
            - batch_size: int, batch size
            - metrics: callable, metrics function
            - alpha: float, alpha parameter for the balanced criterion
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        if criterion is None:
            criterion = self.balanced_criterion
        X, y = self.validate_data(X, y)
        X = self.window_data(X, self.input_size)
        y = y[self.input_size - 1:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state= self.random_state)
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = True)
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for i, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output, volatility = self.forward(data)
                loss = criterion((output, volatility), (data, target), alpha)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch + 1}, loss: {total_loss / len(train_loader)}")
            predicted = []
            true = []
            val_loss = 0
            for i, (data, target) in enumerate(test_loader):
                output, volatility = self.predict(data)
                loss = criterion((output, volatility), (data, target), alpha)
                val_loss += loss.item()
                predicted.append(volatility)
                true.append(target)
        
            predicted = torch.cat(predicted).squeeze(-1)
            true = torch.cat(true)
            if self.verbose:
                print(f"Validation loss: {val_loss / len(test_loader)}")
                print(f"Metrics: {metrics(true, predicted)}")
        return self
