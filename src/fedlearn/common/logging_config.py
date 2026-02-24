import logging.config

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "fedlearn": {
            "()": "colorlog.ColoredFormatter",
            "format": (
                "%(blue)s%(asctime)s%(reset)s | %(levelname)s | %(name)s | %(message)s"
            ),
        },
        "plain": {
            "format": "%(levelname)s : %(message)s",
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stdout",
        },
        "console_fedlearn": {
            "class": "logging.StreamHandler",
            "formatter": "fedlearn",
            "stream": "ext://sys.stdout",
        },
    },

    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },

    "loggers": {
        "fedlearn": {
            "handlers": ["console_fedlearn"],
            "level": "INFO",
            "propagate": False,
        },

        # reduce noise
        "flwr": {
            "level": "INFO",  # WARNING
            "propagate": True,
        },
        "optuna": {
            "level": "WARNING",
            "propagate": True,
        },
    },
}


def setup_logging() -> None:
    logging.config.dictConfig(LOGGING)
