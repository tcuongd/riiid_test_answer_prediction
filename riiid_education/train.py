import os
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np
import xgboost
import lightgbm
import seaborn as sns
import matplotlib.pyplot as plt

from . import constants
from .utils import cumsum_to_previous, get_logger

plt.rcParams["figure.figsize"] = [8, 6]
sns.set_style("whitegrid")
log = get_logger()


def read_train_sample(n_users: int, rows_per_read: int = 200000) -> pd.DataFrame:
    """
    Read a sample of train.csv, making sure to read the entire set of observations
    for any given user_id. Relies on train.csv being ordered by user_id and timestamp.

    n_users: Number of users to read in.
    rows_per_read: Number of rows to read per iteration.
    """
    rows_read = 0
    user_count = 0
    dfs = []
    last_user_id = -1
    tic = datetime.utcnow()
    while user_count < n_users:
        df = pd.read_csv(
            "train.csv",
            names=constants.TRAIN_SCHEMA.keys(),
            dtype=constants.TRAIN_SCHEMA,
            usecols=constants.DATAPREP_COLS,
            skiprows=rows_read + 1,
            nrows=rows_per_read,
        )
        if df.empty:
            break
        first_user_id = df["user_id"].iloc[0]
        users_in_batch = df["user_id"].unique().shape[0]
        if last_user_id == first_user_id:
            users_in_batch -= 1
        user_count += users_in_batch
        last_user_id = df["user_id"].iloc[-1]
        rows_read += df.shape[0]
        if user_count >= n_users:
            df = df.loc[df["user_id"] != last_user_id]
        dfs.append(df)
    toc = datetime.utcnow()
    log.info(f"Users processed: {user_count}. Time taken: {toc - tic}.")

    return pd.concat(dfs)


def read_questions() -> pd.DataFrame:
    return pd.read_csv("questions.csv", usecols=constants.QUESTION_COLS).rename(
        columns={"question_id": "content_id"}
    )


def split_train_data(
    df: pd.DataFrame, question_features_perc: float
) -> Dict[str, pd.DataFrame]:
    """
    - Exclude lecture observations
    - Randomly exclude a subset of users (question_features_perc) for training content_id success rates (question difficulty)
    """
    df = df.loc[df["content_type_id"] == 0].drop(columns="content_type_id")
    users = df["user_id"].unique()
    np.random.shuffle(users)
    users_question_features = users[: int(question_features_perc * users.shape[0])]

    return {
        "user_model": df.loc[~df["user_id"].isin(users_question_features)].reset_index(
            drop=True
        ),
        "question_features": df.loc[
            df["user_id"].isin(users_question_features)
        ].reset_index(drop=True),
    }


def explode_tags(questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode questions data from question : [tag1, tag2, ...] format to one row per
    question-tag combination.
    """
    df = questions_df.copy()
    df["tags"] = questions_df["tags"].apply(lambda x: str(x).split(" "))
    df = df.explode("tags").rename(columns={"tags": "tag"}).reset_index(drop=True)

    return df[["content_id", "tag"]]


def train_question_features(
    obs_df: pd.DataFrame, questions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simple Beta-Binomial model for question difficulty at the question, part, and tag level.
    """
    # Calculate accuracy at the question level
    obs_g = (
        obs_df.groupby("content_id")
        .agg({"content_id": "count", "answered_correctly": "sum"})
        .rename(columns={"content_id": "answered"})
        .reset_index()
    )
    questions_df = pd.merge(
        questions_df,
        obs_g[["content_id", "answered", "answered_correctly"]],
        how="left",
        on="content_id",
    ).fillna(0)
    # Calculate accuracy at the part level
    part_df = (
        questions_df.groupby("part")
        .agg({"answered": "sum", "answered_correctly": "sum"})
        .reset_index()
    )
    # Calculate accuracy at the tag level
    all_tags = explode_tags(questions_df)
    all_tags = pd.merge(
        all_tags,
        obs_g[["content_id", "answered", "answered_correctly"]],
        how="left",
        on="content_id",
    ).fillna(0)
    tags_df = (
        all_tags.groupby("tag")
        .agg({"answered": "sum", "answered_correctly": "sum"})
        .reset_index()
    )

    prior_alpha = constants.QUESTION_FEATURES_PRIOR[0]
    prior_beta = constants.QUESTION_FEATURES_PRIOR[1]
    questions_df["question_accuracy"] = (
        questions_df["answered_correctly"] + prior_alpha
    ) / (questions_df["answered"] + prior_alpha + prior_beta)
    part_df["part_accuracy"] = (part_df["answered_correctly"] + prior_alpha) / (
        part_df["answered"] + prior_alpha + prior_beta
    )
    tags_df["tag_accuracy"] = (tags_df["answered_correctly"] + prior_alpha) / (
        tags_df["answered"] + prior_alpha + prior_beta
    )

    questions_tags = pd.merge(
        all_tags[["content_id", "tag"]], tags_df[["tag", "tag_accuracy"]], on="tag"
    )
    tags_summ = (
        questions_tags.groupby("content_id")
        .agg({"tag_accuracy": ["min", "mean", "max", "count"]})
        .reset_index()
    )
    tags_summ.columns = [
        "content_id",
        "min_tag_accuracy",
        "avg_tag_accuracy",
        "max_tag_accuracy",
        "tag_count",
    ]

    features = pd.merge(
        questions_df, part_df[["part", "part_accuracy"]], on="part", how="left"
    )
    features = pd.merge(features, tags_summ, on="content_id", how="left")
    for agg_type in ["part", "min_tag", "avg_tag", "max_tag"]:
        features[f"diff_question_{agg_type}"] = (
            features["question_accuracy"] - features[f"{agg_type}_accuracy"]
        )
    features = features.fillna(constants.DEFAULT_NA_VALUE)

    return features


def build_features(
    obs_df: pd.DataFrame, question_features: pd.DataFrame, recent_obs_number: int
) -> pd.DataFrame:
    """
    Feature engineering for one observation per row. We are assuming that at any
    given observation, we will have the fully history of the user's previous question
    attempts. This is violated in practice, as sometimes users complete a "task container",
    which is a collection of questions, and we don't know whether they got the previous
    question(s) in the container correct until they complete the whole container.
    However, because each "container" has at most 3-4 questions, we think the impact
    is minimal and not worth the effort of fixing.

    Parameters
    ----------
    obs_df: Dataframe of observations for each user. Each user must have their full
    history of observations.
    question_features: Embeddings or other features calculated at the question level.
    recent_obs_number: The number of observations we determine to be "recent". For
        example, if recent_obs_number = 15, then any features calculated using
        the last 15 observations only are considered to capture "recent" behaviour.

    Returns
    -------
    One row per observations, with user_id, the feature columns, and the target.
    """
    obs_df = pd.merge(
        obs_df,
        question_features[["content_id"] + constants.QUESTION_FEATURE_COLS],
        how="left",
        on="content_id",
    )
    # Users' total previous accuracy
    user_g = obs_df.groupby("user_id")
    obs_df["answered"] = user_g.cumcount()
    obs_df["correct"] = user_g["answered_correctly"].apply(cumsum_to_previous).fillna(0)
    obs_df["cum_accuracy"] = obs_df["correct"] / obs_df["answered"]
    # Users' recent previous accuracy
    obs_df["answered_recent"] = obs_df["answered"].clip(upper=recent_obs_number)
    obs_df["correct_recent"] = obs_df["correct"] - user_g["correct"].shift(
        periods=recent_obs_number
    )
    obs_df.loc[obs_df["correct_recent"].isna(), "correct_recent"] = obs_df.loc[
        obs_df["correct_recent"].isna(), "correct"
    ]
    obs_df["recent_accuracy"] = obs_df["correct_recent"] / obs_df["answered_recent"]
    # Stability in users' accuracy metric (i.e. how consistent is their track record)
    obs_df["trend_accuracy"] = constants.DEFAULT_NA_VALUE
    obs_df.loc[obs_df["answered_recent"] < obs_df["answered"], "trend_accuracy"] = (
        obs_df.loc[obs_df["answered_recent"] < obs_df["answered"], "recent_accuracy"]
        - obs_df.loc[obs_df["answered_recent"] < obs_df["answered"], "cum_accuracy"]
    )
    # Previous question elapsed time and explanation flag
    for col in ["prior_question_elapsed_time", "prior_question_had_explanation"]:
        obs_df[f"cum_{col}"] = user_g[col].cumsum()
        obs_df[f"cum_avg_{col}"] = obs_df[f"cum_{col}"] / obs_df["answered"]
        obs_df[f"recent_{col}"] = obs_df[f"cum_{col}"] - user_g[f"cum_{col}"].shift(
            periods=recent_obs_number
        )
        obs_df.loc[obs_df[f"recent_{col}"].isna(), f"recent_{col}"] = obs_df.loc[
            obs_df[f"recent_{col}"].isna(), f"cum_{col}"
        ]
        obs_df[f"recent_avg_{col}"] = (
            obs_df[f"recent_{col}"] / obs_df["answered_recent"]
        )
    # Users' previous performance taking into account the difficulty of the question.
    # Score is between 0 and 2 (the higher the better)
    obs_df["answered_score"] = (
        obs_df["answered_correctly"] - obs_df["question_accuracy"] + 1
    )
    obs_df["cum_score"] = user_g["answered_score"].apply(cumsum_to_previous).fillna(0)
    obs_df["cum_avg_score"] = obs_df["cum_score"] / obs_df["answered"]
    obs_df["recent_score"] = obs_df["cum_score"] - user_g["cum_score"].shift(
        periods=recent_obs_number
    )
    obs_df.loc[obs_df["recent_score"].isna(), "recent_score"] = obs_df.loc[
        obs_df["recent_score"].isna(), "cum_score"
    ]
    obs_df["recent_avg_score"] = obs_df["recent_score"] / obs_df["answered_recent"]
    obs_df["trend_avg_score"] = constants.DEFAULT_NA_VALUE
    obs_df.loc[obs_df["answered_recent"] < obs_df["answered"], "trend_avg_score"] = (
        obs_df.loc[obs_df["answered_recent"] < obs_df["answered"], "recent_avg_score"]
        - obs_df.loc[obs_df["answered_recent"] < obs_df["answered"], "cum_avg_score"]
    )
    # User 'frequency of use' proxied by questions answered per day
    obs_df["frequency_of_use"] = (
        (obs_df["answered"] + 1) / (obs_df["timestamp"] / (1000 * 60 * 60 * 24))
    ).replace([-np.inf, np.inf], np.nan)
    obs_df = obs_df.fillna(constants.DEFAULT_NA_VALUE)

    return obs_df[constants.TRAIN_COLS + ["user_id", constants.TARGET_COL]]


def prepare_validation_data(
    df: pd.DataFrame,
    val_perc: float,
    train_obs_per_user: int = 200,
    val_obs_per_user: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    During training we may to reduce the bias towards power users by sampling
    a set number of observations per user. This will be the last train_obs_per_user
    observations for each user. If train_obs_per_user=None, we take all observations for
    all users.

    We want to make the distribution of validation questions resemble the test set
    questions, but that is not possible given we are not given actual timestamp info
    for each observations. We set the # obs per user in the val set equal to a slightly
    lower number than for the test set, since we are predicting into the future
    for many existing users (rather than predicting their earlier questions), which
    is a proxy. We can only check whether this resembles the test set by submitting
    to the leaderboard.
    """
    summ = df.groupby("user_id")["answered_correctly"].count().reset_index()
    val_users = summ["user_id"].sample(frac=val_perc, random_state=1234)

    train = df.loc[~df["user_id"].isin(val_users)].copy()
    if train_obs_per_user:
        train["reverse_obs_num"] = (
            train.groupby("user_id").cumcount(ascending=False) + 1
        )
        train = train.loc[train["reverse_obs_num"] <= train_obs_per_user]

    val = df.loc[df["user_id"].isin(val_users)].copy()
    val["reverse_obs_num"] = val.groupby("user_id").cumcount(ascending=False) + 1
    val = val.loc[val["reverse_obs_num"] <= val_obs_per_user]

    return {"train": train, "val": val}


def train_lgbm(model_data: dict, params: dict) -> lightgbm.LGBMClassifier:
    m = lightgbm.LGBMClassifier(**params)
    X_train = model_data["train"][constants.TRAIN_COLS]
    y_train = np.array(model_data["train"][constants.TARGET_COL]).astype(int)
    X_val = model_data["val"][constants.TRAIN_COLS]
    y_val = np.array(model_data["val"][constants.TARGET_COL]).astype(int)

    tic = datetime.utcnow()
    m.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        eval_metric=["logloss", "auc"],
        verbose=False,
        early_stopping_rounds=50,
    )
    toc = datetime.utcnow()
    log.info(f"lgbm training time: {toc - tic}")

    return m


def train_xgboost(model_data: dict, params: dict) -> xgboost.XGBClassifier:
    m = xgboost.XGBClassifier(**params)
    X_train = model_data["train"][constants.TRAIN_COLS]
    y_train = np.array(model_data["train"][constants.TARGET_COL]).astype(int)
    X_val = model_data["val"][constants.TRAIN_COLS]
    y_val = np.array(model_data["val"][constants.TARGET_COL]).astype(int)

    tic = datetime.utcnow()
    m.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["logloss", "auc"],
        verbose=False,
        early_stopping_rounds=50,
    )
    toc = datetime.utcnow()
    log.info(f"xgboost training time: {toc - tic}")

    return m


def extract_lgbm_features(model: lightgbm.LGBMClassifier) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "feature_name": model.feature_name_,
            "feature_importance": model.feature_importances_,
        }
    )
    return df


def extract_xgboost_features(model: xgboost.XGBClassifier) -> pd.DataFrame:
    bst = model.get_booster()
    df = pd.DataFrame(
        {
            "feature_name": bst.feature_names,
            "feature_importance": model.feature_importances_,
        }
    )
    return df


def extract_lightgbm_eval(model: lightgbm.LGBMClassifier) -> pd.DataFrame:
    df = pd.DataFrame(model.evals_result_["val"]).rename(
        columns={"binary_logloss": "logloss"}
    )
    df["iteration"] = [i + 1 for i in range(df.shape[0])]
    df["eval_set"] = "val"
    df["model"] = "lgbm"
    return df


def extract_xgboost_eval(model: xgboost.XGBClassifier) -> pd.DataFrame:
    df = pd.DataFrame(model.evals_result()["validation_0"])
    df["iteration"] = [i + 1 for i in range(df.shape[0])]
    df["eval_set"] = "val"
    df["model"] = "xgboost"
    return df


def extract_eval(
    lgbm: lightgbm.LGBMClassifier, xgb: xgboost.XGBClassifier
) -> pd.DataFrame:
    df = pd.concat(
        [extract_lightgbm_eval(lgbm), extract_xgboost_eval(xgb)], ignore_index=True
    )
    return df


def plot_eval(eval_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    sns.lineplot(
        data=eval_df,
        x="iteration",
        y="auc",
        hue="model",
    )
    plt.title("AUC")
    plt.savefig("models/auc_eval.png")

    fig, ax = plt.subplots()
    sns.lineplot(
        data=eval_df,
        x="iteration",
        y="logloss",
        hue="model",
    )
    plt.title("Logistic Loss")
    plt.savefig("models/logloss_eval.png")


def train_models() -> None:
    """
    Train and save models.
    """
    os.makedirs("models", exist_ok=True)
    log.info("Reading training data")
    tr = read_train_sample(n_users=constants.NUM_USERS, rows_per_read=int(10e6))
    questions = read_questions()
    splits = split_train_data(
        tr, question_features_perc=constants.QUESTION_FEATURES_PERC
    )
    for k, df in splits.items():
        log.info(f"{k}: {df.shape[0]:,.0f} observations")

    log.info("Building question features")
    question_features = train_question_features(splits["question_features"], questions)
    question_features.to_csv("models/question_features.csv", index=False)
    log.info("Question features saved to models/question_features.csv")

    log.info("Building all features")
    df = build_features(
        splits["user_model"],
        question_features,
        recent_obs_number=constants.RECENT_WINDOW,
    )
    model_data = prepare_validation_data(
        df,
        val_perc=constants.VALIDATION_PERC,
        train_obs_per_user=200,
        val_obs_per_user=100,
    )
    for k, df in model_data.items():
        log.info(f"Main model {k}: {df.shape[0]:,.0f} observations.")
        # Save train and validation data for later debugging
        # df.to_csv(f"models/main_model_{k}.csv", index=False)
        # log.info(f"{k} data saved to models/main_model_{k}.csv")

    log.info("Training main models")
    lgbm = train_lgbm(model_data, constants.LIGHTGBM_PARAMS)
    xgb = train_xgboost(model_data, constants.XGBOOST_PARAMS)
    lgbm.booster_.save_model("models/main_model_lgbm.b")
    xgb.save_model("models/main_model_xgb.b")
    log.info("Main models saved to models/")

    lgbm_features = extract_lgbm_features(lgbm)
    xgb_features = extract_xgboost_features(xgb)
    lgbm_features.to_csv("models/main_model_lgbm_features.csv", index=False)
    xgb_features.to_csv("models/main_model_xgb_features.csv", index=False)
    eval_results = extract_eval(lgbm=lgbm, xgb=xgb)
    plot_eval(eval_results)
    log.info("Eval plots model features saved to models/")
