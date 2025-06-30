import logging
from logging.config import dictConfig

import yaml

# Load logging configuration
with open("app/trace/logging.yml", "r") as f:
    config = yaml.safe_load(f.read())
    dictConfig(config)

# Initialize logger instance
logger = logging.getLogger("GNW")

