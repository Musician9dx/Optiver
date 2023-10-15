from noops.utils.logger import get_logger
from noops.utils.yaml import handle_yaml
from dataclasses import dataclass
import os
from pymongo.mongo_client import MongoClient
import pandas as pd


logger=get_logger()
yaml_handler=handle_yaml()


@dataclass
class DataIngestionConfig():
        
    try:

        logger.info("Reading YAML")

        config=yaml_handler.read_data(
            os.path.join(
                os.getcwd(),
                "src","Optiver",
                "config","dataingestion.yaml"
                )
            )
        logger.info("YAML Read Successfull")
    
    except Exception as e:
        
        logger.error(str(e))


class DataIngestion():

    def __init__(self) -> None:

        try:

            logger.info("Reading Configuration")

            self.config=DataIngestionConfig().config

            logger.info("Configuration Loaded")
        
        except Exception as e:

            logger.error(str(e))
        



    def connect_to_Mongo(self):

        try:

            logger.info("Connecting to MongoDB")

            
            client=MongoClient(self.config.mongoURL)

            database=client["KaggleOptiver"]

            self.collection=database["Optiver"]
        
            logger.info("Cursor Returned Successfully")
        
        except Exception as e:

            logger.critical(str(e))


    def get_data(self):

        try:
        
            logger.info("Parsing Data")

            cursor=self.collection

            data=[]

            for record in cursor.find({}):

                data.append(record)

            self.data=data

            logger.info("Data Successfully Parsed")
        
        except Exception as e:

            logger.error(str(e))
    
    def save_data(self):

        try:

            logger.info("Saving Data")

            DataFrame=pd.DataFrame(self.data)

            DataFrame.to_csv(os.path.join( self.config.DataStoragePath , "data.csv" )  )
        
            logger.info("Data Saved Successfully")

        except Exception as e:

            logger.error(str(e))

