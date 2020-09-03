from population import Population

if __name__ == "__main__":
    population = Population(density=5)

    print(f'best: {population.best_candidate}')
    population.evolve(500)
    print(f'best: {population.best_candidate}')
