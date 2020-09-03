from random import randint, random

from common import Settings


class Individual:
    """References Individual

    """

    def __init__(self, genes: list = None):
        self.__genes = genes
        self.__fitness = 0

        if self.__genes is None:
            self.__initialize()

        self.__evaluate_fitness()

    def __repr__(self) -> str:
        return f'< Individual: [{self.chromosome}] ({self.fitness}) >'

    def __str__(self) -> str:
        return f'[{self.chromosome}]: {self.fitness}'

    def __evaluate_fitness(self):
        """fitness evaluation

        the more an individual as ones, the fittest it is
        """
        self.__fitness = sum(self.__genes)

    def __initialize(self):
        """individual initialization

        initialize the individual genes with random 1's ans 0's
        """
        self.__genes = [randint(0, 1) for _ in range(Settings.chromosome_len)]

    def mutate(self):
        """perform a mutation on the individual

        select a random gene in the chromosome and toggle its value
        """
        if random() < Settings.mutation_probability:
            rdm_index = randint(0, Settings.chromosome_len - 1)
            self.__genes[rdm_index] = 0 if self.__genes[rdm_index] else 1

    @property
    def chromosome(self) -> str:
        return ''.join([str(e) for e in self.__genes])

    @property
    def fitness(self) -> int:
        return self.__fitness

    @property
    def genes(self):
        return self.__genes
