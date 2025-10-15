# Lab Guidelines

This file contains all of the technical specifications you need for the lab case. This includes the data dictionary and the expected output format from your model. Kindly read this thoroughly.

## Submission & Reproducibility Requirements

For your submission, ensure that you have the following files:

| File                          | Description                                                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `report.ipynb`                | The main technical report                                                                                                       |
| `report.pdf`                  | Optional. The main technical report in PDF. You may need to install `pandoc` for this                                           |
| `<stock_code>_model.py`       | Your stock model's source code file. Note you can have multiple model.py files for each stock, or a single model for all.       |
| `<stock_code>_historical.csv` | The historical stock data used for the report. Note that for each stock chosen, you save the historical as a CSV.               |
| `<stock_code>_forecasts.csv`  | The forecasts for each chosen stock as a CSV. Ensure that you include the date and the column corresponds to the `Close` price. |
| `contribution sheet`          | The accomplished contribution sheet of the group as a PDF                                                                       |
| `utils.py`                    | The (un)modified utils.py provided                                                                                              |
| `backtest.py`                 | The (un)modified backtest.py provided                                                                                           |

You may include figures or a directory for figures within your submission. You may also include other files that are needed to reproduce your report properly.

For your submission, kindly compress all of these files in a single zip file. The file name would be the following: `03-forecast-<insert lt #>.zip`. An example submission would look like `03-forecast-lt1.zip`. A submission bin in ALICE will be available on the week of the lab.

We are going to assume you are using Jojie. Therefore, the Python version should be >=3.12.

### Packages

If you know how to, create a `requirements.txt` file that would include all of the package requirements within your virtual environment. If not, kindly add any non-Jojie-native package installations within your report as a separate cell. For example

```
!pip install imblearn langchain --quiet
```

You can use the `--quiet` argument to avoid displaying the installation prints or use `%%capture` within the cell. Example:

```
%%capture

!pip install imblearn langchain
```

### Creating your `<stock_code>_model.py`

Create a `<stock_code>_model.py` file containing a class called `<Stock_Code>TradingModel` that would output the price forecasts. This model class should inherit to an appropriate parent class when applicable. Moreover, it should contain **at least** the `fit`, `predict`, and `fit_predict` methods. These methods should have the parameters `[X,y]`, `[X]`, and `[X, y]`, respectively. If your method is ARMA, ARIMA, SARIMA, or a statistical model, then you may replace the `predict` and `fit_predict` methods with a `forecast` method that takes in `n_steps` as its parameter.

A sample implementation is shown below for an example only:

```python
# Instructor's notes for students:
# Include any imports you may need

...
from sklearn.base import BaseEstimator, RegressorMixin
...

class AAPLTradingModel(BaseEstimator, RegressorMixin):
    def __init__(...):
        ...

    def fit(X, y):
        ...

    def predict(X):
        ...

    def fit_predict(X, y):
        ...

    # Instructor's notes for students:
    # Include and create any methods you may need
    # Make sure that these are prefixed by an underscore (_)
    # especially if they are methods used within the class only

    # You may also include the forecasting strategy, e.g., recursive forecasts,
    # rolling window, expanded window, etc., within this class if applicable.
    ...

```

Of course, feel free to remove the comments shown above.

### Importing the custom model from `<stock_code>_model.py`

To use the model within `<stock_code>_model.py` simply import it in a cell, such as

```
from aapl_model import AAPLTradingModel
```

This should allow you to use the model within the script. To help streamline your development process, you can include the `autoreload` extension within your jupyter notebook. To use it, simply run or include this within a cell and run it.

```
%load_ext autoreload
%autoreload 2
```

Using this extension would allow your notebook to automatically update the `FraudDetector` class within your noteboook without having to manually re-run the `from model import FraudDetector` cell.

## Sample Usage of Backtesting and created metrics

To further guide the group, here is a simple implementation to show how to use the two scripts provided.

```python
import numpy as np
import pandas as pd
import yfinance as yf

from utils import ForecastingMetrics, TradingMetrics
from backtest import Backtest

data = yf.download('MSFT', start='2022-01-01', end='2025-01-01')

train = data["Close"][:-30].values.flatten()
test = data["Close"][-30:].values.flatten()

# conduct naive forecast
forecasts = [train[-1]] * len(test)

# model metrics
print(ForecastingMetrics.compute_all_metrics(np.array(test), np.array(forecasts)))

# backtesting sample
backtest_framework = Backtest()
trade_hist = backtest_framework.run_strategy(np.array(test), np.array(forecasts))

print(backtest_framework.compute_trade_metrics())

```

For the `PortfolioEvaluator` class, you would need to run the backtests first before placing it in the class. Ensure that you have added your stocks to the portfolio with the correct `allocation`. Below is a sample usage of this evaluator.

```python
import numpy as np
import pandas as pd
import yfinance as yf

from utils import PortfolioEvaluator
from backtest import Backtest

aapl = yf.download('AAPL', start='2022-01-01', end='2025-01-01')
msft = yf.download('MSFT', start='2022-01-01', end='2025-01-01')

# Define test and training splits

train_msft = msft["Close"][:-30].values.flatten()
test_msft = msft["Close"][-30:].values.flatten()

train_aapl = aapl["Close"][:-30].values.flatten()
test_aapl = aapl["Close"][-30:].values.flatten()

# Conduct a naive forecast

msft_forecasts = np.array([train_msft[-1]] * len(test_msft))
aaple_forecasts = np.array([train_aapl[-1]] * len(test_aapl))

# Define starting capital

capital = 10000
msft_backtest = Backtest(starting_capital=capital)
aapl_backtest = Backtest(starting_capital=capital)

msft_backtest.run_strategy(np.array(test_msft), msft_forecasts)
aapl_backtest.run_strategy(np.array(test_aapl), aaple_forecasts)

# Define Evaluator

allocation = "equal"
# allocation = {"AAPL": 0.7, "MSFT": 0.3} # you can use this if you have a different stock allocation
portfolio = PortfolioEvaluator(starting_capital=capital, allocation=allocation)

portfolio.add_stock('AAPL', aapl_backtest, model_name='Naive')
portfolio.add_stock('MSFT', msft_backtest, model_name='Naive')

# Display report
# Note that this will run the metrics already
portfolio.show_report()

```

## Code documentation instructions

Keep in mind that docstrings and type annotations are part of the grading criteria. Docstrings are simply strings that describe the method and provides additional context on the parameters and output. A type annotation is a tool that would inform other developers what should be the expected data type of the parameters and the expected output type. The syntax of a type annotation is `parameter: dtype`. Type annotations come from the `typing` modules.

Here are two examples of functions with type annotations

```
def sum(a: int, b: int) -> int:
    pass
```

```
from typing import Union
def sum(a: Union[int, float], b: Union[int, float] = 1) -> Union[int, float]:
    pass
```

Notice that the `Union` operator provides an `or` statement to the developer. The arrow (`->`) after the function parenthesis and before the colon (`:`) is the type annotation for the output. Also notice for the second example that we set the defaults after the type annotation, i.e., `b: Union[int, float] = 1`. For our lab, the type annotation for the output will be optional.

### Additional notes

You may use the scikit-learn pipeline object. However, it is not required. You are also given the freedom to create your own trading strategy and pick a stock
outside of the provided list of stocks.

### Important Reminder

Ensure that your report provides the requirements from the case, as well as, a comparison and/or explanations of your proposed model/method within the main report. The main goal of this laboratory is to solidify your understanding of time series forecasting through a practical use case of trading.

Make sure to properly attribute the data in `yfinance`.
