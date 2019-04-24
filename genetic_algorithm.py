import numpy as np
from skopt.space import Real, Integer
from copy import deepcopy
import logging

class GA(object):
    """
    A genetic algorithm optimizer that gives the steps of evolutionary optimization.
    
    Parameters
    
    dimensions [list, shape=(n_dims,)]
        List of search space dimensions. Each search dimension is an instance of a
        'Dimension' object ('Real' or 'Integer')
    
    the argument num_iterations in ask specifies the number of generations
    """
    
    nonuniformityMutationConstant = 3
    blank = -1
    
    def __init__(self, dimensions, populationSize):
        self.paramRanges = dimensions
        self.numParams = len(self.paramRanges)
        self.populationSize = populationSize
        self.numElite = populationSize // 5
        self.fitness = np.array([self.blank]*populationSize)
        
        population = []
        for i in range(0, populationSize):
            chromosome = []
            for j in range(0, self.numParams):
                gene = float(self.paramRanges[j].rvs()[0])
                chromosome.append(gene)
            population.append(chromosome)
        self.population = population
    
    def setGenerations(self, maxGenerations):
        self.maxGenerations = maxGenerations
    
    def ask(self):
        rounded = deepcopy(self.population)
        for i in range(len(self.population)):
            for j in range(len(self.paramRanges)):
                if isinstance(self.paramRanges[j], Integer):
                    rounded[i][j] = int(rounded[i][j])
        logging.debug('Asked for: {}'.format(rounded))
        return rounded
    
    # needs params/results to have length populationSize
    def tell(self, params, results, generation):
        searchIn = self.ask()
        for i in range(len(searchIn)):
            for j in range(len(params)):
                if params[j] == searchIn[i] and self.fitness[i] == self.blank:
                    logging.debug('Found told params')
                    self.fitness[i] = 1/results[j]
        logging.debug('Fitness: {}'.format(self.fitness))
        if self.blank not in self.fitness:
            logging.debug('Population before stepping: {}'.format(self.population))
            self.step(generation)
            logging.debug('Population after stepping: {}'.format(self.population))
        
        loss = 1
        
        bestInd = self.fitness.argsort()[-1:][::-1][0]
        if self.fitness[bestInd] is not None:
                loss = 1 / self.fitness[bestInd]
        return self.population[bestInd], loss
        
    
    def step(self, generation):
        logging.debug('Stepping in generation {} with fitness: {}'.format(generation, self.fitness))
        children = []
        newFitness = [self.blank]*self.populationSize
        
        # elitism: copy fittest organisms
        eliteIndices = self.fitness.argsort()[-self.numElite:][::-1]
        for i in range(len(eliteIndices)):
            children.append(self.population[eliteIndices[i]])
        
        i = self.numElite
        while i < self.populationSize:
            p1 = self.tournamentSelect()
            p2 = self.tournamentSelect()
            children.append(self.mutate(self.crossover(p1, p2), generation))
            i += 1
        
        self.population = children
        self.fitness = np.array(newFitness)
    
    def tournamentSelect(self):
        arr = list(range(self.populationSize))
        ind1 = np.random.randint(self.populationSize)
        arr.remove(ind1)
        ind2 = np.random.choice(arr)
        
        if self.fitness[ind1] > self.fitness[ind2]:
            return ind1
        else:
            return ind2
    
    def crossover(self, p1, p2):
        moreFit = p1
        lessFit = p2
        if self.fitness[p2] > self.fitness[p1]:
            moreFit = p2
            lessFit = p1
        
        child = []
        for i in range(self.numParams):
            gene = np.random.random() * (self.population[moreFit][i] - self.population[lessFit][i]) + self.population[moreFit][i]
            gene = self.capGene(gene, i)
            child.append(gene)
        
        return child
    
    def capGene(self, gene, i):
        if gene < self.paramRanges[i].low:
            gene = float(self.paramRanges[i].low)
        elif gene > self.paramRanges[i].high:
            gene = float(self.paramRanges[i].high)
        return gene
    
    def mutate(self, chromosomeOrig, generation):
        chromosome = list(chromosomeOrig)
        for i in range(self.numParams):
            mutation = 1 - np.power(np.random.random(), np.power(1 - generation/self.maxGenerations, self.nonuniformityMutationConstant))
            
            if np.random.random() > 0.5:
                mutation *= self.paramRanges[i].high - chromosome[i]
            else:
                mutation *= -(chromosome[i] - self.paramRanges[i].low)
            chromosome[i] += mutation
            chromosome[i] = self.capGene(chromosome[i], i)
        return chromosome
    