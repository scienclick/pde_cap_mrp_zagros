#region imports

import pandas as pd
import random
from sklearn.neighbors import KNeighborsRegressor
from sklearn import set_config
set_config(display="diagram");
RANDOMSEED = 100
DISPLAY_WIDTH = 400
DISPLAYMAX_COLUMNS = 25
random.seed(RANDOMSEED)
pd.set_option('display.width', DISPLAY_WIDTH)
pd.set_option('display.max_columns', DISPLAYMAX_COLUMNS)
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='once')
import numpy as np
import shap
import pickle
from mrputils.loaders import DataLoader
from mrputils.processors import tweak_data,tweak_data4_prediction
#endregion
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import mlflow
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold,cross_val_score

import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


if __name__ == "__main__":
    dp=DataLoader("./data")
    dp.get_data() #loading all the data

    df_awards,df_all_casting,df_all_details=dp.df_awards,dp.df_all_casting,dp.df_all_details

    __,df_nulls=tweak_data(df_all_casting,df_all_details,df_awards)

    # Split the data into training and test sets. (0.75, 0.25) split.
    X=df_nulls.drop(columns=["id","revenue","popularity"])
    X_with_id=df_nulls.drop(columns=["revenue","popularity"])
    y=df_nulls[["revenue","popularity"]]
    target_1=y.iloc[:,0]
    target_2=y.iloc[:,1]

    mlflow.sklearn.autolog()
    mlflow.xgboost.autolog()

    # with mlflow.start_run():
    cat_transformer = Pipeline([
        ("ohc",OneHotEncoder(handle_unknown='ignore'))
        ])
    preprocessor = ColumnTransformer([
        ('cat_tr', cat_transformer, ['day_of_week']),
        ('cat_imputer', SimpleImputer(strategy="most_frequent"), ['day_of_week','actor1_name', 'actor2_name',
                                                                'actor3_name', 'actor4_name', 'actor5_name',
                                                                'director_freq', 'fame_nominated', 'fame_won']),
        ('imputer',SimpleImputer(strategy="median"),["runtime",'age', 'sin_month', 'cos_month'])

        ],remainder="passthrough")

    estimator=Pipeline(
        [
            ("estimator",XGBRegressor(
                max_depth=6,           # maximum depth of each tree - try 2 to 10
                learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
                n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
                min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
                colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
                subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
                reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
                reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
                num_parallel_tree=1,   # set > 1 for boosted random forests
            ))
            # ("estimator",lgbm.LGBMRegressor())
            # ("estimator",random_search.best_estimator_)
        ]
    )

    m=Pipeline([
            ('preprocessor',preprocessor),
            # ('imputer',KNNImputer()),
            ('scaler',StandardScaler()),
            ("estimator",estimator)

        ])

    kf=KFold(n_splits=5,random_state=1,shuffle=True)

    scores=cross_val_score(m, X, y.iloc[:,0], cv=kf)
        # scores

        # # mlflow.log_param("alpha", alpha)
        # # mlflow.log_param("l1_ratio", l1_ratio)
        # mlflow.log_metric("score1", scores[0])
        # mlflow.log_metric("score2", scores[1])
        # mlflow.log_metric("score3", scores[2])
        # mlflow.log_metric("score4", scores[3])
        # mlflow.log_metric("score5", scores[4])
        # mlflow.log_metric("score", np.mean(scores))

        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # # Model registry does not work with file store
        # if tracking_url_type_store != "file":

        #     # Register the model
        #     # There are other ways to use the Model Registry, which depends on the use case,
        #     # please refer to the doc for more information:
        #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        #     mlflow.sklearn.log_model(m, "model_MRP", registered_model_name="XGBoost")
        # else:
        #     mlflow.sklearn.log_model(m, "model_MRP")
