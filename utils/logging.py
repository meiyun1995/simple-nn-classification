import logging
import os

def get_logger(name: str):
    """Get logger object.
    
    Args:
        name (str): name of the logger
    
    Returns:
        logging.Logger: logger object
    """
    if not os.path.exists("log/"):
        os.mkdir("log/")

    logging.basicConfig(
        filename="log/hotel_reservations.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %I:%M:%S%p",
    )

    # Module-level logger
    logger = logging.getLogger(name)

    return logger

LOG = get_logger("hotel_reservations")