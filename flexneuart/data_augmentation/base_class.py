import spacy
import random


class DataAugment:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

        return

    @abstractmethod
    def augment(text, **kwargs):
        pass       
