import logging
import os

from dotenv import load_dotenv

base_path = os.path.dirname(os.path.abspath(__file__))

load_dotenv(dotenv_path="{dir}/.env".format(dir=base_path))  # custom settings
load_dotenv(dotenv_path="{dir}/.env.dist".format(dir=base_path))  # defaults


slack_token = os.getenv("SLACK_TOKEN")
slack_channel = os.getenv("SLACK_CHANNEL")

exasol_conn = {
    'host': os.getenv("EXASOL_HOST"),
    'user': os.getenv("EXASOL_USER"),
    'password': os.getenv("EXASOL_PASSWORD"),
}

ch_segmentation_conn = {
    'url': os.getenv("CH_SEGMENTATION_URL"),
    'user': os.getenv("CH_SEGMENTATION_USER"),
    'password': os.getenv("CH_SEGMENTATION_PASSWORD"),
    'database': os.getenv("CH_SEGMENTATION_DATABASE"),
}

logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL")),
)

version = os.getenv("SOURCE_CODE_VERSION")
