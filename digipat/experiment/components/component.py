class Component:
    def sanity_check(self):
        raise NotImplementedError("Please Implement this method")

    def __call__(self):
        raise NotImplementedError("Please Implement this method")

    def create_artifact(self):
        raise NotImplementedError("Please Implement this method")
