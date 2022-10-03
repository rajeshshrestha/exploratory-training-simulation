import logging
import os
import sys
from datetime import datetime


def setup_logging(log_file_folder: str,
                  log_level: str = "INFO",
                  filemode: str = "a",
                  format: str = '%(asctime)s [%(filename)s Line %(lineno)d]: '
                                '%(levelname)s - %(message)s') -> None:
    """
    Setup the configuration for logging

    Parameters
    ----------
    log_file_folder : str
        path of the folder where the log file needs to be stored
    log_level : str, optional
        the level of the logging, by default "INFO"
    filemode : str, optional
        the mode in which the logfile is to be opened, by default "a"
    format : str, optional
        the format of logging, by default '%(asctime)s [%(filename)s Line %(
        lineno)d]: %(levelname)s - %(message)s'
    """

    '''Prepare log file name and directory'''
    log_file_name = f"{datetime.now()}.log"
    log_file_path = os.path.join(log_file_folder,
                                 log_file_name)
    os.makedirs(log_file_folder,
                exist_ok=True)

    '''Basic configuration for logging'''
    logging.basicConfig(
        format=format,
        level=log_level,
        handlers=[logging.FileHandler(log_file_path,
                                      mode=filemode),
                  logging.StreamHandler(sys.stdout)]
    )

    logging.info("Logging has been configured.")


if __name__ == "__main__":
    setup_logging(log_file_folder="./logs/")
