from Optiver.components.data_ingestion import DataIngestion


obj=DataIngestion()

obj.connect_to_Mongo()

obj.get_data()

obj.save_data()

