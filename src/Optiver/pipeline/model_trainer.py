from Optiver.components.model_trainer import ModelTrainer

obj=ModelTrainer()

obj.read_data()

obj.create_datasets()

obj.BuildModel()

obj.train_model()