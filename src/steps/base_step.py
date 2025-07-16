from abc import ABC, abstractmethod

class BaseStep(ABC):
    def __init__(self, params, data, general_config):
        self.params = params
        self.data = data
        self.general_config = general_config

    @abstractmethod
    def run(self):
        pass
