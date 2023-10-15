from noops.utils.logger import get_logger
from noops.utils.yaml import handle_yaml
from dataclasses import dataclass
import os


logger=get_logger()
yaml_handler=handle_yaml()


@dataclass
class DataIngestionConfig():

    config=yaml_handler.read_data(
        os.path.join(
            os.getcwd(),
            "src","Optiver",
            "config","dataingestion.yaml"
            )
        )


class DataIngestion():

    def __init__(self) -> None:
        self.config=DataIngestionConfig().config
