import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_tansformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Correct column names based on the actual dataset
            numerical_columns = ['writing score', 'reading score']

            categorical_columns = [
                "gender",
                "race/ethnicity",   # FIXED COLUMN NAME
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            # Combine both pipelines
            preprocess = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocess

        except Exception as e:
            logging.error("Error occurred in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']

            # Splitting input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            # Fit on train, transform both
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Data transformation completed")

            # Save preprocessor object
            save_object(
                file_path=self.data_tansformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_tansformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Error occurred during initiate_data_transformation")
            raise CustomException(e, sys)
