import time

startup_time = time.time() * 1000


def get_time():
    return time.time() * 1000 - startup_time
