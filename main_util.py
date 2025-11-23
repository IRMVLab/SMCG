import logging

logging.getLogger().setLevel(logging.INFO)


def make_print_to_file(path="./", fileName="Debug.log"):
    import os
    import sys

    if not os.path.exists(path):
        os.makedirs(path)

    class Logger(object):
        def __init__(self, filename="Debug.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(
                os.path.join(path, filename),
                "a",
                encoding="utf8",
            )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(fileName, path=path)
