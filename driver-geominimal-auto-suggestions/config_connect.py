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

ch_main_conn = {
    'url': os.getenv("CH_MAIN_URL"),
    'user': os.getenv("CH_MAIN_USER"),
    'password': os.getenv("CH_MAIN_PASSWORD"),
}
