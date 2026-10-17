from network_security.entity.artifact_entity import DataIngestionArtifact
from network_security.entity.config_entity import DataValidationConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from network_security.utils.main_utils.utils import read_yaml_file,write_yaml_file
from network_security.entity.artifact_entity import DataValidationArtifact



class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        """Validate column count, names, and order against the configured schema."""
        try:
            schema_columns = self._schema_config.get("columns")
            if not isinstance(schema_columns, list) or not schema_columns:
                raise ValueError("Schema must define a non-empty 'columns' list.")

            expected_columns = []
            for column_definition in schema_columns:
                if not isinstance(column_definition, dict) or len(column_definition) != 1:
                    raise ValueError(
                        "Each schema column entry must be a mapping with exactly one column name."
                    )
                expected_columns.append(next(iter(column_definition)))

            if len(expected_columns) != len(set(expected_columns)):
                raise ValueError("Schema contains duplicate column names.")

            actual_columns = list(dataframe.columns)
            logging.info(f"Required number of columns :{len(expected_columns)}")
            logging.info(f"Dataframe has columns :{len(actual_columns)}")
            if actual_columns != expected_columns:
                logging.error(
                    "Dataframe schema mismatch. Expected ordered columns %s; got %s",
                    expected_columns,
                    actual_columns,
                )
                return False
            return True
        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        """Write per-column KS diagnostics and return True only when no drift is detected."""
        try:
            if list(base_df.columns) != list(current_df.columns):
                raise ValueError(
                    "Base and current dataframes must have identical columns in identical order."
                )
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_sampe_dist=ks_2samp(d1,d2)
                if threshold<=is_sampe_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_sampe_dist.pvalue),
                    "drift_status":is_found
                }})
            drif_report_file_path=self.data_validation_config.drift_report_file_path

            dir_path=os.path.dirname(drif_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drif_report_file_path,content=report)
            return status

        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            if not self.validate_number_of_columns(dataframe=train_dataframe):
                raise ValueError("Training dataframe columns do not match the configured schema.")
            if not self.validate_number_of_columns(dataframe=test_dataframe):
                raise ValueError("Test dataframe columns do not match the configured schema.")

            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            valid_train_path = self.data_validation_config.valid_train_file_path
            valid_test_path = self.data_validation_config.valid_test_file_path
            for output_path in (valid_train_path, valid_test_path):
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

            train_dataframe.to_csv(valid_train_path,index=False,header=True)
            test_dataframe.to_csv(valid_test_path,index=False,header=True)
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
