from noops.utils.logger import get_logger
from noops.utils.yaml import handle_yaml
from noops.prisma.pickles import object_handler
from dataclasses import dataclass
import os


logger=get_logger()
yaml_handler=handle_yaml()
object_handler=object_handler()

@dataclass
class ModelPredictorConfig():

    config=yaml_handler.read_data(
        os.path.join(
            os.getcwd(),
            "src","Optiver",
            "config","modelprediction.yaml"
            )
        )

class ModelPredictor():

    def __init__(self) -> None:
        self.config=ModelPredictorConfig().config
    

    def load_model(self):

        self.ml=object_handler(self.config.ModelPath)

    def load_data(self):

        self.TestData=object_handler(self.config.TestDataPath)

    def predict_timeseries(self):
        
        predictions=self.ml.predict(self.TestData)

    def save_predictions(self):
        pass
