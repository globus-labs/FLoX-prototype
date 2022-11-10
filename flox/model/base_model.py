from abc import ABC


class BaseModel(ABC):

    arch: any

    def __init__(self, arch):
        self.arch = arch

    def fit():
        pass

    def eval():
        pass