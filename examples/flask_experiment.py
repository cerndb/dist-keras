from flask import Flask, request

from functools import wraps

from multiprocessing import Process

import time

class Test:

    def __init__(self):
        self.TEST = 10
        self.app = Flask(__name__)
        self.__name__ = __name__

    def start_service(self):
        self.service = Process(target=self.service)
        self.service.start()

    def service(self):
        test = self

        @self.app.route("/test", methods=['GET'])
        @wraps(self)
        def test():
            self.TEST = 5

            return "OK"

        self.app.run(host='0.0.0.0', threaded=True, use_reloader=False)

    def stop_service(self):
        self.service.terminate()
        self.service.join()

    def test(self):
        self.start_service()
        time.sleep(5)
        self.stop_service()

        return self.TEST

test = Test()
c = test
value = test.test()
print(value)
