import logging

def build_logger(name):

    # 创建一个日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个文件处理器，并设置级别为DEBUG
    file_handler = logging.FileHandler('app.log')
    # file_handler.setLevel(logging.DEBUG)

    # 创建一个控制台处理器，并设置级别为WARNING
    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.WARNING)

    # 创建一个日志格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 为文件处理器和控制台处理器设置格式化器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 记录日志
    # logger.debug('This is a debug message')
    # logger.info('This is an info message')
    # logger.warning('This is a warning message')
    # logger.error('This is an error message')
    # logger.critical('This is a critical message')

    return logger

# logger = build_logger()