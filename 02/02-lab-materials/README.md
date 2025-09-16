# Lab Guidelines

This file contains all of the technical specifications you need for the lab case. This includes the data dictionary and the expected output format from your model. Kindly read this thoroughly.

## Data Dictionary

The historical dataset contains a total of 70 unique babies each observed for a period of 30 days.

| Column                        | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `baby_id`                     | Unique identifier for a baby                       |
| `name`                        | Name of the baby                                   |
| `gender`                      | Male or Female                                     |
| `gestational_age_weeks`       | Gestational age at birth (normal: 37-42 weeks)     |
| `birth_weight_kg`             | Birth weight (normal: 2.5-4.5 kg)                  |
| `birth_length_cm`             | Length at birth (avg. 48-52 cm)                    |
| `birth_head_circumference_cm` | Head circumference at birth (avg. 33-35cm)         |
| `date`                        | Monitoring date                                    |
| `age_days`                    | Age of the baby in days since birth                |
| `weight_kg`                   | Daily updated weight (growth trend ~25–30g/day)    |
| `length_cm`                   | Daily updated body length                          |
| `head_circumference_cm`       | Daily updated head circumference                   |
| `temperature_c`               | Body temperature in °C (normal: 36.5–37.5°C)       |
| `heart_rate_bpm`              | Heart rate (normal: 120–160 bpm)                   |
| `respiratory_rate_bpm`        | Breathing rate (normal: 30–60 breaths/min)         |
| `oxygen_saturation`           | SpO₂ level (normal >95%)                           |
| `feeding_type`                | Breastfeeding / Formula / Mixed                    |
| `feeding_frequency_per_day`   | Number of feeds per day (normal: 8–12)             |
| `urine_output_count`          | Wet diapers/day (normal: 6–8+)                     |
| `stool_count`                 | Bowel movements per day (0–5 is common)            |
| `jaundice_level_mg_dl`        | Bilirubin level (normal <5, mild 5–12, severe >15) |
| `apgar_score`                 | 0–10 score at birth (only day 1)                   |
| `immunizations_done`          | Yes/No (BCG, HepB, OPV on Day 1 & 30)              |
| `reflexes_normal`             | Newborn reflex check (Yes/No)                      |
| `risk_level`                  | Healthy (0) or At Risk (1)                         |

## Model Output Requirements

Once you have finalized the `model.py` file, create a CSV file that contains the model's predictions on `test.csv`. It should have two columns, namely, `baby_id`, `age_days` and `risk_level`. Note that you would not be able to evaluate on the test set. The test set performance does not have any bearing to your lab grade, but rather it would be used to inform the group of their approach's feasibility. The test set evaluation would be performed by the instructor, once everyone has submitted their final submission.

For more clarity, here is an example output from your model:

| `baby_id` | `age_days` | `risk_level` |
| --------- | ---------- | ------------ |
| 1234      | 2          | 0            |
| 5678      | 5          | 1            |

You can retrieve the `baby_id` and `age_days` values for your model predictions within the `test.csv` file.

Note: when exporting to CSV from a pandas DataFrame, always set the argument `index=False` to avoid saving the indices within the file.

## Submission & Reproducibility Requirements

For your submission, ensure that you have the following files:

| File                 | Description                                                                           |
| -------------------- | ------------------------------------------------------------------------------------- |
| `report.ipynb`       | The main technical report                                                             |
| `report.pdf`         | Optional. The main technical report in PDF. You may need to install `pandoc` for this |
| `model_output.csv`   | The model's prediction on the test set as a CSV file                                  |
| `model.py`           | Your model's source code file                                                         |
| `historical.csv`     | The historical data provided                                                          |
| `test.csv`           | The test data provided                                                                |
| `contribution sheet` | The accomplished contribution sheet of the group as a PDF                             |
| `old_model.joblib`   | The renamed `model.joblib` model file                                                 |
| `new_model.joblib`   | Your proposed trained model                                                           |

You may include figures or a directory for figures within your submission. You may also include other files that are needed to reproduce your report properly.

For your submission, kindly compress all of these files in a single zip file. The file name would be the following: `02-eval-lt<insert lt #>.zip`. An example submission would look like `02-eval-lt1.zip`. A submission bin in ALICE will be available on the week of the lab.

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

### Creating your `model.py`

Create a `model.py` file containing a class called `WhiteBox` that could output probabilities or class labels. If you are to propose a framework that outputs probabilities, the threshold is a standard `0.5`. You have the option to output probabilities for your `model_output.csv` file but know that the instructor would be thresholding during their testing.

Your `WhiteBox` class should inherit from `scikit-learn`'s `estimator` objects. Moreover, it should contain **at least** the `fit`, `predict`, and `fit_predict` methods. These methods should have the parameters `[X,y]`, `[X]`, and `[X, y]`, respectively. A sample implementation could look something like the code below.

```python
# Instructor's notes for student:
# Include any imports you may need

...
from sklearn.base import BaseEstimator, ClassifierMixin
...

class WhiteBox(BaseEstimator, ClassifierMixin):
    def __init__(...):
        ...

    def fit(X, y):
        ...

    def predict(X):
        ...

    def fit_predict(X, y):
        ...

    # Instructor's notes for student:
    # Include and create any methods you may need
    # Make sure that these are prefixed by an underscore (_)
    # especially if they are methods used within the class only
    ...

```

Of course, feel free to remove the comments shown above.

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

Treat the `model.py` file as a scikit-learn model. You can therefore use your existing modules and functions in scikit-learn with this model. An example usage is

```
from sklearn.pipe import Pipeline
from model import WhiteBox

pipe = Pipeline(
    [("clf", WhiteBox)]
)

pipe.fit(...)
```

Also, we are expecting that on your submission you would rename the `model.joblib` file provided into `old_model.joblib`. Part of your submission requirements is to include your own trained model called `new_model.joblib`.

### Important Reminder

Ensure that your report provides the requirements from the case, as well as, a comparison and/or explanations of your proposed model/method within the main report. The main goal of this laboratory is to solidify explainability methods while improving the learner's programming competency.

## Attribution

The dataset is a modified version of this dataset: https://www.kaggle.com/datasets/miadul/newborn-health-monitoring-dataset
