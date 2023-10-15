from noops.utils.logger import get_logger
from noops.utils.yaml import handle_yaml
from dataclasses import dataclass
import os


logger=get_logger()
yaml_handler=handle_yaml()


@dataclass
class ModelEvaluatorConfig():

    config=yaml_handler.read_data(
        os.path.join(
            os.getcwd(),
            "src","Optiver",
            "config","modelevaluation.yaml"
            )
        )


class ModelEvaluator():

    def __init__(self) -> None:
        self.config=ModelEvaluatorConfig().config
