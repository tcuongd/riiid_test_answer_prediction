## Riiid Education Test Answer Prediction

[Competition Link](https://www.kaggle.com/c/riiid-test-answer-prediction/overview)

[Final Submission Link](https://www.kaggle.com/abcdqd/riiid-submission-2-extra-features)

### Installation

`pipenv` is used to manage the Python environment for this collection of scripts. To install:

* For MACOSX, Install `pipenv` with `brew install pipenv`.
* In this directory, `pipenv install --dev`.

### Data

Download and unzip data from the [Competition Link](https://www.kaggle.com/c/riiid-test-answer-prediction/overview)
into the current directory.

### Scripts

To train the main models used in the competition:

```
pipenv run pretrain_models
```

To build the user summary database that will be used to fetch and update historical
user stats at predicton time:

```
pipenv run prebuild_user_summary
```

To test the prediction pipeline

```
pipenv run generate_predictions
```

### Exploration

`pipenv run jupyter lab` to run a Juypter notebook with the correct Python environment.

Take a sample of users to iterate on feature engineering and validating models.
See `prototype.ipynb` for an example.

### Insights

Exploratory summaries of feature distributions and feature importance as determined
by the models: see `model_insights.ipynb`.
