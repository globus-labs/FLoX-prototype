import concurrent.futures

from flox.logic.base_executor import BaseExecutor


class LocalExecutor(BaseExecutor):
    def __init__(self):
        self.futures = []

    def submit(self, function, data):
        self.futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(function, data)
            self.futures.append(future)

    def get_results(self):
        return [f.result() for f in self.futures]
