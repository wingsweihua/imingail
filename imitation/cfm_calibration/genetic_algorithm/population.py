from random import randint
from typing import List

from common import Settings
from individual import Individual


class Population:
    def __init__(self, density: int = Settings.population_size):
        self.__individuals = [
            Individual() for _ in range(density)
        ]

    def __str__(self) -> str:
        return str(self.__individuals[0])

    def evolve(self, generations: int = 1) -> None:
        """loop through the

        at each loop, creates two children from the best
        parents among the population and insert them
        """
        for _ in range(generations):
            self.perform_crossover()

    def insert_new(self, individuals: list):
        """insert new individuals in the population

        remove the worsts candidates to add the newest
        """
        self.__individuals = \
            individuals + sorted(
                self.__individuals,
                key=lambda individual: -individual.fitness
            )[:-2]
    
    def perform_crossover(self):
        """perform a crossover among the best candidate

        create new individuals based on the best ones
        each new individual may mutate given a certain probability
        """
        best, second = self.best_parents
        crossover_point = randint(1, Settings.chromosome_len - 1)

        children = [
            Individual(
                second.genes[:crossover_point] + best.genes[crossover_point:]
            ),
            Individual(
                second.genes[:crossover_point] + best.genes[crossover_point:]
            )
        ]
        [child.mutate() for child in children]

        self.insert_new(children)

    @property
    def population(self) -> List[Individual]:
        return self.__individuals
    
    @property
    def best_candidate(self) -> Individual:
        return sorted(
            self.__individuals,
            key=lambda individual: -individual.fitness
        )[0]
    
    @property
    def best_parents(self) -> List[Individual]:
        return sorted(
            self.__individuals,
            key=lambda individual: -individual.fitness
        )[:2]

