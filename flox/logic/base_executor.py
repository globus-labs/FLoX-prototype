class BaseExecutor:
    def submit(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def get_results(self):
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")
