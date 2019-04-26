import logging.handlers
import os

from util.path_control import LOG_DIR

file_name = os.path.basename(LOG_DIR)
file_path = os.path.dirname(LOG_DIR)
if not os.path.exists(file_path):
    os.makedirs(file_path)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler(LOG_DIR, when='H', interval=6, backupCount=40)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(funcName)s][line:%(lineno)d][%(levelname)s] %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)