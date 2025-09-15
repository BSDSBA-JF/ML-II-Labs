# Lab Guidelines

This file contains all of the technical specifications you need for the lab case. This includes the data dictionary and the expected output format from your model. Kindly read this thoroughly. For Prof. Ethan, the project environment is set up with Python version 3.12.8.

## Data Dictionary

| Column    | Description                                                 |
| --------- | ----------------------------------------------------------- |
| `tid`     | Unique transaction ID                                       |
| `V(1-28)` | Principal components from a PCA operation done to mask PIIs |
| `Amount`  | The transaction amount in USD                               |
| `outcome` | Indicates if the transaction is fraudulent (1) or not (0)   |

## Model Output Requirements

Once you have finalized the `model.py` file, create a CSV file that contains the model's predictions on `test.csv`. It should have two columns, namely, `tid` and `outcome`. Note that you would not be able to evaluate on the test set. The test set performance does not have any bearing to your lab grade, but rather it would be used to inform the group of their approach's feasibility. The test set evaluation would be performed by the instructor, once everyone has submitted their final submission.

For more clarity, here is an example output from your model:

| `tid` | `outcome` |
| ----- | --------- |
| 1234  | 0         |
| 5678  | 1         |

You can retrieve the `tid` values for your model predictions within the `test.csv` file.

Note: when exporting to CSV from a pandas DataFrame, always set the argument `index=False` to avoid saving the indices within the file.

## Submission & Reproducibility Requirements

For your submission, ensure that you have the following files:

| File                 | Description                                                                           |
| -------------------- | ------------------------------------------------------------------------------------- |
| `report.ipynb`       | The main technical report                                                             |
| `report.pdf`         | Optional. The main technical report in PDF. You may need to install `pandoc` for this |
| `model_output.csv`   | The model's prediction on the test set as a CSV file                                  |
| `model.py`           | The updated model file                                                                |
| `historical.csv`     | The historical data provided                                                          |
| `test.csv`           | The test data provided                                                                |
| `contribution sheet` | The accomplished contribution sheet of the group as a PDF                             |

You may include figures or a directory for figures within your submission. You may also include other files that are needed to reproduce your report properly.

For your submission, kindly compress all of these files in a single zip file. The file name would be the following: `01-imba-lt<insert lt #>.zip`. An example submission would look like `01-imba-lt1.zip`. A submission bin in Jojie will be available on the week of the lab.

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

### Importing the custom model from `model.py`

To use the model within `model.py` simply import it in a cell, such as

```
from model import FraudDetector
```

This should allow you to use the model within the script. To help streamline your development process, you can include the `autoreload` extension within your jupyter notebook. To use it, simply run or include this within a cell and run it.

```
%load_ext autoreload
%autoreload 2
```

Using this extension would allow your notebook to automatically update the `FraudDetector` class within your noteboook without having to manually re-run the `from model import FraudDetector` cell.

## `model.py` instructions

Only modify the `_handle_imbalance` method within the `FraudDetector` class. Should you need to include other parameters, make sure that this is reflected within the `fit` method of the class. You can also add your own class methods within `FraudDetector`. Just ensure that they have the underscore (\_) prefix. For example,

```
def _smote(...):
    ...
```

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

Treat the `model.py` file as a scikit-learn model. You can therefore use your existing modules and functions in scikit-learn with this model. An example usage is

```
from sklearn.pipe import Pipeline
from model import FraudDetector

pipe = Pipeline(
    [("clf", FraudDetector)]
)

pipe.fit(...)

```

## Attribution

The dataset is a modified version of this dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
