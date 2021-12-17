import logging

from ml_engine import *



def init_logger():
    _logger = logging.getLogger('Main')
    _logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename='test.log')
    file_handler.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    return _logger



if __name__ == '__main__':

    logger = init_logger()

    me = MLEngine()
    me.danger()

