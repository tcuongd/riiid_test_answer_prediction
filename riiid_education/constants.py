import numpy as np

USER_SUMMARY = "submission/user_summary_extra_features.csv"
FINAL_MAIN_XGB = "models/main_model_xgb.b"
FINAL_MAIN_LGBM = "models/main_model_lgbm.b"
FINAL_QUESTION_FEATURES = "models/question_features.csv"
NUM_USERS = 393655  # ALL USERS
DEFAULT_NA_VALUE = -99
TRAIN_SCHEMA = {
    "row_id": np.int64,
    "timestamp": np.int64,
    "user_id": np.int32,
    "content_id": np.int16,
    "content_type_id": np.int8,
    "task_container_id": np.int16,
    "user_answer": np.int8,
    "answered_correctly": np.int8,
    "prior_question_elapsed_time": np.float32,
    "prior_question_had_explanation": np.float32,
}
DATAPREP_COLS = [
    "timestamp",
    "user_id",
    "content_id",
    "content_type_id",
    "answered_correctly",
    "prior_question_elapsed_time",
    "prior_question_had_explanation",
]
QUESTION_COLS = ["question_id", "part", "tags"]
QUESTION_FEATURES_PERC = 0.5
QUESTION_FEATURES_PRIOR = (10, 10)
RECENT_WINDOW = 15
QUESTION_FEATURE_COLS = [
    "question_accuracy",
    "part_accuracy",
    "min_tag_accuracy",
    "avg_tag_accuracy",
    "max_tag_accuracy",
    "diff_question_part",
    "diff_question_avg_tag",
    "diff_question_min_tag",
    "diff_question_max_tag",
    "tag_count",
]
TRAIN_COLS = QUESTION_FEATURE_COLS + [
    "answered",
    "prior_question_elapsed_time",
    "cum_avg_prior_question_elapsed_time",
    "recent_avg_prior_question_elapsed_time",
    "prior_question_had_explanation",
    "cum_avg_prior_question_had_explanation",
    "recent_avg_prior_question_had_explanation",
    "timestamp",
    "frequency_of_use",
    "cum_accuracy",
    "recent_accuracy",
    "trend_accuracy",
    "cum_avg_score",
    "recent_avg_score",
    "trend_avg_score",
]
TARGET_COL = "answered_correctly"
VALIDATION_PERC = 0.2
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 10,
    "learning_rate": 0.05,
    "objective": "binary:logistic",
    "booster": "gbtree",
    "tree_method": "hist",
    "subsample": 0.5,
    "missing": DEFAULT_NA_VALUE,
    "use_label_encoder": False,
}
LIGHTGBM_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 20,
    "num_leaves": 600,
    "learning_rate": 0.05,
    "objective": "binary",
    "boosting_type": "gbdt",
    "subsample": 0.5,
}
MODEL_DATA_SCHEMA = {
    "question_accuracy": np.float,
    "part_accuracy": np.float,
    "avg_tag_accuracy": np.float,
    "tag_count": np.int,
    "answered": np.int,
    "prior_question_elapsed_time": np.float,
    "prior_question_had_explanation": np.float,
    "timestamp": np.int,
    "frequency_of_use": np.float,
    "cum_accuracy": np.float,
    "recent_accuracy": np.float,
    "trend_accuracy": np.float,
    "cum_avg_score": np.float,
    "recent_avg_score": np.float,
    "trend_avg_score": np.float,
}
USER_SUMMARY_SCHEMA = {
    "user_id": np.int,
    "answered": np.int,
    "correct": np.int,
    "recent_history_answered_correctly": str,
    "correct_baseline": np.float,
    "recent_history_question_accuracy": str,
    "cum_prior_question_elapsed_time": np.float,
    "recent_history_prior_question_elapsed_time": str,
    "cum_prior_question_had_explanation": np.float,
    "recent_history_prior_question_had_explanation": str,
}
