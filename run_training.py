import logging

from ml_garden import Pipeline

logging.basicConfig(level=logging.DEBUG)

data = Pipeline.from_json("config.json").run(is_train=True)
