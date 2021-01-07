import os
from datetime import datetime
from typing import Union, Optional, Tuple

import pandas as pd
import numpy as np
import xgboost
import lightgbm

from . import constants
from .utils import get_logger, bool_to_float

log = get_logger()


def load_question_features(filepath) -> pd.DataFrame:
    df = pd.read_csv(filepath)[["content_id"] + constants.QUESTION_FEATURE_COLS]
    df.set_index("content_id", inplace=True)

    return df


def load_xgboost(filepath) -> xgboost.XGBClassifier:
    m = xgboost.XGBClassifier()
    m.load_model(filepath)

    return m


def load_lgbm(filepath) -> lightgbm.Booster:
    m = lightgbm.Booster(model_file=filepath)

    return m


def process_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a stream of user observations and returns an aggregated dataframe
    with one row per user, summarising their total and recent question performance.

    Returns
    -------
    pd.DataFrame indexed by user_id.
    """
    required_columns = [
        "user_id",
        "question_accuracy",
        "answered_correctly",
        "prior_question_elapsed_time",
        "prior_question_had_explanation",
    ]
    missing_columns = set(required_columns).difference(df.columns.tolist())
    assert not missing_columns, f"Missing columns: {missing_columns}"
    for col in ["prior_question_elapsed_time", "prior_question_had_explanation"]:
        df[col].replace(
            np.nan, 0.0, inplace=True
        )  # Can't parse np.nan from list-string
    user_g = df.groupby("user_id")
    summ = user_g.agg(
        {
            "answered_correctly": [
                "count",
                "sum",
                lambda x: list(x)[-constants.RECENT_WINDOW :],
            ],
            "question_accuracy": ["sum", lambda x: list(x)[-constants.RECENT_WINDOW :]],
            "prior_question_elapsed_time": [
                "sum",
                lambda x: list(x)[-constants.RECENT_WINDOW :],
            ],
            "prior_question_had_explanation": [
                "sum",
                lambda x: list(x)[-constants.RECENT_WINDOW :],
            ],
        }
    )
    summ.columns = [
        "answered",
        "correct",
        "recent_history_answered_correctly",
        "correct_baseline",
        "recent_history_question_accuracy",
        "cum_prior_question_elapsed_time",
        "recent_history_prior_question_elapsed_time",
        "cum_prior_question_had_explanation",
        "recent_history_prior_question_had_explanation",
    ]

    return summ


def process_train_observations(
    skiprows: int, nrows: int, question_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Process a subset of the train.csv data. We assume that the data is ordered by user_id,
    so that if the user_id changes from one row to the next, we have the complete
    history for the user. This also means we cut out the last user_id from any batch
    of rows we read in (except for when there's only one user_id).

    Parameters
    ----------
    skiprows: skip this number of observations. Note we automatically skip the
        header row of the csv.
    nrows: read this many observations after `skiprows`.
    question_features: pd.DataFrame indexed by content_id that contains QUESTION_FEATURE_COLS.

    Returns
    -------
    One row per user, for every user in the training dataset, with summary stats
    as calculated by `process_observations`.
    """
    df = pd.read_csv(
        "train.csv",
        names=constants.TRAIN_SCHEMA.keys(),
        dtype=constants.TRAIN_SCHEMA,
        usecols=constants.DATAPREP_COLS,
        skiprows=skiprows + 1,
        nrows=nrows,
    )
    if df.empty:
        return None, 0
    else:
        last_user_id = df["user_id"].iloc[-1]
        if df["user_id"].unique().shape[0] > 1:
            df = df.loc[~(df["user_id"] == last_user_id)].copy()
        rows_processed = df.shape[0]
        df.query("content_type_id == 0", inplace=True)
        df.drop(columns="content_type_id", inplace=True)
        df = pd.merge(
            df,
            question_features,
            left_on="content_id",
            right_index=True,
            how="left",
            copy=False,
        )
        summ = process_observations(df)
        return summ, rows_processed


def process_test_observations(
    curr_obs: pd.DataFrame, prev_obs: pd.DataFrame, question_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Process test observations that come in groups. At the start of each new group
    of observations, we are given the ground truth values for whether the user answered
    the questions in the previous group correctly. We join these two datasets together
    to build a user-level summary of performance since the start of the testing period.

    Parameters
    ----------
    curr_obs: The dataframe for the current test data group.
    prev_obs: The dataframe for the previous test data group.
    question_features: Question features to join on content_id.

    Returns
    -------
    One row per user, with stats calculated as per `process_observations`.
    """
    answer_history = eval(curr_obs["prior_group_answers_correct"].iloc[0])
    if not answer_history:
        return None
    else:
        prev_obs["answered_correctly"] = answer_history
        prev_obs = prev_obs.loc[prev_obs["content_type_id"] == 0].drop(
            columns="content_type_id"
        )
        prev_obs = pd.merge(
            prev_obs,
            question_features,
            left_on="content_id",
            right_index=True,
            how="left",
            copy=False,
        )
        summ = process_observations(prev_obs)

        return summ


def calculate_user_features(df: pd.DataFrame, inplace=True) -> Optional[pd.DataFrame]:
    """
    Calculate the user features to be used in scoring, modifying the source dataframe
    inplace by default.
    """
    required_columns = [
        k for k in constants.USER_SUMMARY_SCHEMA.keys() if k != "user_id"
    ]
    missing_columns = set(required_columns).difference(df.columns.tolist())
    assert not missing_columns, f"Missing columns: {missing_columns}"

    if not inplace:
        df = df.copy()
    df["prior_question_had_explanation"] = df["prior_question_had_explanation"].apply(
        bool_to_float
    )

    df["recent_answered"] = df["recent_history_answered_correctly"].apply(len)
    df["recent_correct"] = df["recent_history_answered_correctly"].apply(sum)
    df["recent_correct_baseline"] = df["recent_history_question_accuracy"].apply(sum)

    df["cum_accuracy"] = df["correct"] / df["answered"]
    df["recent_accuracy"] = df["recent_correct"] / df["recent_answered"]
    df["trend_accuracy"] = constants.DEFAULT_NA_VALUE
    df.loc[df["recent_answered"] < df["answered"], "trend_accuracy"] = (
        df.loc[df["recent_answered"] < df["answered"], "recent_accuracy"]
        - df.loc[df["recent_answered"] < df["answered"], "cum_accuracy"]
    )

    df["cum_avg_score"] = (
        df["correct"] - df["correct_baseline"] + df["answered"]
    ) / df["answered"]
    df["recent_avg_score"] = (
        df["recent_correct"] - df["recent_correct_baseline"] + df["recent_answered"]
    ) / df["recent_answered"]
    df["trend_avg_score"] = constants.DEFAULT_NA_VALUE
    df.loc[df["recent_answered"] < df["answered"], "trend_avg_score"] = (
        df.loc[df["recent_answered"] < df["answered"], "recent_avg_score"]
        - df.loc[df["recent_answered"] < df["answered"], "cum_avg_score"]
    )

    df["cum_prior_question_elapsed_time"] = (
        df["cum_prior_question_elapsed_time"] + df["prior_question_elapsed_time"]
    )
    df["cum_prior_question_had_explanation"] = (
        df["cum_prior_question_had_explanation"] + df["prior_question_had_explanation"]
    )
    df["recent_prior_question_elapsed_time"] = (
        df["recent_history_prior_question_elapsed_time"].apply(lambda x: sum(x[1:]))
        + df["prior_question_elapsed_time"]
    )
    df["recent_prior_question_had_explanation"] = (
        df["recent_history_prior_question_had_explanation"].apply(lambda x: sum(x[1:]))
        + df["prior_question_had_explanation"]
    )

    df["cum_avg_prior_question_elapsed_time"] = (
        df["cum_prior_question_elapsed_time"] / df["answered"]
    )
    df["cum_avg_prior_question_had_explanation"] = (
        df["cum_prior_question_had_explanation"] / df["answered"]
    )
    df["recent_avg_prior_question_elapsed_time"] = (
        df["recent_prior_question_elapsed_time"] / df["recent_answered"]
    )
    df["recent_avg_prior_question_had_explanation"] = (
        df["recent_prior_question_had_explanation"] / df["recent_answered"]
    )

    df["frequency_of_use"] = (
        (df["answered"] + 1) / (df["timestamp"] / (1000 * 60 * 60 * 24))
    ).replace([-np.inf, np.inf], np.nan)

    for col in constants.TRAIN_COLS:
        if col in ["answered", "timestamp"]:
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(constants.DEFAULT_NA_VALUE, inplace=True)

    if not inplace:
        return df
    else:
        return None


class UserSummary:
    def __init__(self):
        """
        An object that acts as a database for user features. Used for fetching the
        latest user stats for scoring at prediction time, and updating user stats
        as we get correct / incorrect answer feedback from the user.
        """
        self.data = None

    def load(self, filepath) -> None:
        """
        Load UserSummary.data from a csv file.
        """
        log.info(f"Loading user summary from {filepath}.")
        tic = datetime.utcnow()
        df = pd.read_csv(
            filepath,
            dtype=constants.USER_SUMMARY_SCHEMA,
        ).set_index("user_id")
        for col in constants.USER_SUMMARY_SCHEMA.keys():
            if col.startswith("recent_history_"):
                df[col] = df[col].apply(eval)
        self.data = df.to_dict(orient="index")
        toc = datetime.utcnow()
        log.info(f"Loaded user summary. Time taken: {toc - tic}")

    def build(
        self, question_features: pd.DataFrame, rows_per_read: int = int(10e6)
    ) -> None:
        """
        Parameters
        ----------
        question_features: pd.DataFrame indexed by content_id, with columns QUESTION_FEATURE_COLS.
        """
        rows_processed = 0
        dfs = []
        tic = datetime.utcnow()
        while True:
            df, processed_this_batch = process_train_observations(
                skiprows=int(rows_processed),
                nrows=int(rows_per_read),
                question_features=question_features,
            )
            if processed_this_batch == 0:
                log.info("Reached end of file.")
                break
            else:
                rows_processed += processed_this_batch
                dfs.append(df)
                toc = datetime.utcnow()
                log.info(
                    f"Processed {rows_processed:,.0f} rows. Total time taken: {toc - tic}."
                )
        data = pd.concat(dfs)
        self.data = data.to_dict(orient="index")

    def export(self, filepath) -> None:
        tic = datetime.utcnow()
        log.info(f"Exporting UserSummary data in csv format.")
        df = pd.DataFrame.from_dict(self.data, orient="index")
        df.index.name = "user_id"
        df.to_csv(filepath, index=True)
        toc = datetime.utcnow()
        log.info(f"Exported UserSummary data to {filepath}. Time taken: {toc - tic}.")

    def update(self, newdata: pd.DataFrame) -> None:
        """
        newdata: pd.DataFrame indexed by user_id that contains all the columns in
        constants.USER_SUMMARY_SCHEMA.

        This function updates self.data by adding onto existing user counts and
        inserting data for new users.
        """
        required_columns = [
            k for k in constants.USER_SUMMARY_SCHEMA.keys() if k != "user_id"
        ]
        existing_users = newdata.index.intersection(self.data.keys())
        new_users = newdata.index.difference(self.data.keys())

        for idx in existing_users:
            for col in required_columns:
                self.data[idx][col] = self.data[idx][col] + newdata.at[idx, col]
                if col.startswith("recent_history_"):
                    self.data[idx][col] = self.data[idx][col][
                        -constants.RECENT_WINDOW :
                    ]
        for idx in new_users:
            self.data[idx] = {}
            for col in required_columns:
                self.data[idx][col] = newdata.at[idx, col]

    def get_feature(self, user_id: int, feature: str) -> Union[int, float, list]:
        """
        Return a single value for the specified feature for the specified user.
        """
        defaults = {
            "answered": 0,
            "correct": 0,
            "correct_baseline": 0.0,
            "cum_prior_question_elapsed_time": 0.0,
            "cum_prior_question_had_explanation": 0.0,
            "recent_history_question_accuracy": [],
            "recent_history_answered_correctly": [],
            "recent_history_prior_question_elapsed_time": [],
            "recent_history_prior_question_had_explanation": [],
        }
        if not self.data.get(user_id):
            return defaults[feature]
        else:
            return self.data[user_id].get(feature, defaults[feature])


def predict(
    m_xgb: xgboost.XGBClassifier,
    m_lgbm: lightgbm.Booster,
    test: pd.DataFrame,
    test_previous: pd.DataFrame,
    user_summary: "UserSummary",
    question_features: pd.DataFrame,
) -> Tuple[pd.DataFrame]:
    """
    Predict the probability that the user will answer the current question correctly.

    Parameters
    ----------
    m: The model object, an xgboost classifier.
    test: The test data for which to generate predictions.
    test_previous: The previous group of test data observations, used to update
        user summary statistics.
    user_summary: A UserSummary object containing user features, that can be updated
        with incoming data.
    question_features: Question features to join on content_id.

    Returns
    -------
    A tuple of (prediction dataframe, timer dataframe). The timer dataframe is produced
    to help identify bottlenecks in the prediction pipeline that may cause a timeout
    on Kaggle.
    """
    timer = {}
    if test_previous is not None:
        tic = datetime.utcnow()
        newdata = process_test_observations(test, test_previous, question_features)
        toc = datetime.utcnow()
        timer["process_test_observations"] = (toc - tic).total_seconds()

        tic = datetime.utcnow()
        user_summary.update(newdata)
        toc = datetime.utcnow()
        timer["update_user_summary"] = (toc - tic).total_seconds()

    test = test.loc[test["content_type_id"] == 0].drop(columns="content_type_id")
    tic = datetime.utcnow()
    test = pd.merge(
        test,
        question_features,
        how="left",
        left_on="content_id",
        right_index=True,
        copy=False,
    )
    toc = datetime.utcnow()
    timer["merge_question_features"] = (toc - tic).total_seconds()

    tic = datetime.utcnow()
    required_columns = [
        k for k in constants.USER_SUMMARY_SCHEMA.keys() if k != "user_id"
    ]
    for col in required_columns:
        test[col] = [
            user_summary.get_feature(user_id, col) for user_id in test["user_id"]
        ]
    calculate_user_features(test, inplace=True)
    toc = datetime.utcnow()
    timer["merge_user_features"] = (toc - tic).total_seconds()

    tic = datetime.utcnow()
    # test["answered_correctly"] = m_xgb.predict_proba(test[constants.TRAIN_COLS])[:, 1]
    test["answered_correctly"] = m_lgbm.predict(test[constants.TRAIN_COLS])
    toc = datetime.utcnow()
    timer["prediction"] = (toc - tic).total_seconds()

    return test, pd.DataFrame(timer, index=[0])


def prebuild_user_summary():
    """
    Prebuild the user summary database to speed up Kaggle submission times.
    """
    os.makedirs("submission", exist_ok=True)
    question_features = load_question_features()
    user_summary = UserSummary()
    log.info("Building user summary database.")
    user_summary.build(question_features=question_features)
    user_summary.export(constants.USER_SUMMARY)


def generate_predictions():
    """
    Generate predictions on the example test data.
    """
    question_features = load_question_features(constants.FINAL_QUESTION_FEATURES)
    user_summary = UserSummary()
    user_summary.load(constants.USER_SUMMARY)
    m_xgb = load_xgboost(constants.FINAL_MAIN_XGB)
    m_lgbm = load_lgbm(constants.FINAL_MAIN_LGBM)
    test_data = pd.read_csv("example_test.csv")
    all_preds = []
    test_previous = None
    for name, group in test_data.groupby("group_num"):
        log.info(f"Generating predictions for group {name}")
        preds, _ = predict(
            m_xgb, m_lgbm, group, test_previous, user_summary, question_features
        )
        test_previous = group.copy()
        all_preds.append(preds)
    all_preds = pd.concat(all_preds)
    all_preds.to_csv("submission/sample_predictions_df.csv")
    all_preds[["row_id", "answered_correctly", "group_num"]].to_csv(
        "submission/sample_predictions.csv", index=False
    )
    log.info(
        "Predictions on sample test set saved to submission/sample_predictions.csv"
    )


def generate_predictions_large(actual_test_num_rows: int = 2500000):
    """
    Generate predictions for a larger test data set to estimate the runtime for
    Kaggle submissions.
    """
    os.makedirs("submission", exist_ok=True)
    question_features = load_question_features(constants.FINAL_QUESTION_FEATURES)
    user_summary = UserSummary()
    user_summary.load(constants.USER_SUMMARY)
    m_xgb = load_xgboost(constants.FINAL_MAIN_XGB)
    m_lgbm = load_lgbm(constants.FINAL_MAIN_LGBM)
    test_data = pd.read_csv("generated_test.csv")
    all_preds = []
    timers = []
    test_previous = None
    log.info(f"Generating predictions by group.")
    tic = datetime.utcnow()
    for _, group in test_data.groupby("group_num"):
        preds, timer = predict(
            m_xgb, m_lgbm, group, test_previous, user_summary, question_features
        )
        test_previous = group.copy()
        all_preds.append(preds)
        timers.append(timer)
    all_preds = pd.concat(all_preds)
    all_preds[["row_id", "answered_correctly", "group_num"]].to_csv(
        "submission/generated_test_predictions.csv", index=False
    )
    all_preds.to_csv("submission/generated_test_predictions_df.csv", index=False)
    timers = (
        pd.concat(timers, ignore_index=True)
        .reset_index()
        .rename(columns={"index": "group_num"})
    )
    timers.to_csv("submission/generated_test_predictions_timer.csv", index=False)
    toc = datetime.utcnow()
    time_taken = toc - tic
    extrapolated_time_taken = (actual_test_num_rows / all_preds.shape[0]) * time_taken
    log.info(f"Prediction time taken for {all_preds.shape[0]:,.0f} rows: {time_taken}.")
    log.info(
        f"Extrapolated prediction time taken for {actual_test_num_rows:,.0f} rows: {extrapolated_time_taken}."
    )
