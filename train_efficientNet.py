from config import Config
from utils.trainHelper import TrainHelper

if __name__ == '__main__':
    config = Config.config()
    helper = TrainHelper(config=config)
    helper.run()
