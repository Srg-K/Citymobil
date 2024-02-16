import logging
import os

from dotenv import load_dotenv

base_path = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path="{dir}/.env".format(dir=base_path))  # custom settings
load_dotenv(dotenv_path="{dir}/.env.dist".format(dir=base_path))  # defaults


mysql_segmentation_conn = {
    'host': os.getenv("MYSQL_SEGMENTATION_HOST"),
    'port': int(os.getenv("MYSQL_SEGMENTATION_PORT")),
    'user': os.getenv("MYSQL_SEGMENTATION_USER"),
    'password': os.getenv("MYSQL_SEGMENTATION_PASSWORD"),
    'database': os.getenv("MYSQL_SEGMENTATION_DATABASE"),
}

ch_segmentation_conn = {
    'url': os.getenv("CH_SEGMENTATION_URL"),
    'user': os.getenv("CH_SEGMENTATION_USER"),
    'password': os.getenv("CH_SEGMENTATION_PASSWORD"),
    'database': os.getenv("CH_SEGMENTATION_DATABASE"),
}

_log_filename = os.getenv('LOG_FILENAME')
if len(_log_filename) > 0:
    if _log_filename[0] != '/' and _log_filename[0] != '~':
        _log_filename = base_path + '/' + _log_filename
    os.makedirs(os.path.dirname(_log_filename), exist_ok=True)
logging.basicConfig(
    filename=_log_filename,
    level=logging.getLevelName(os.getenv("LOG_LEVEL")),
)

version = os.getenv("SOURCE_CODE_VERSION")
