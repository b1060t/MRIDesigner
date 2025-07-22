from deap import base, creator, tools
import numpy as np
import pandas as pd
from multiprocessing import Pool
import random
import logging
logger = logging.getLogger(__name__)

class Algorithm:
    def __init__(self, records=[]):
        logger.info(f"Initializing Algorithm {type(self).__name__}")
        self.records = pd.DataFrame(columns=['iteration', 'cur_err', 'best_err']+records)

    def init(self):
        pass

    def loop(self, iteration):
        logger.debug(f"Running loop for iteration {iteration}")

    def run(self):
        raise NotImplementedError("The run method must be implemented by subclasses.")
 
class GeneticAlgorithm(Algorithm):
    def __init__(self, records=[], popSize=100, cxpb=0.55, mutpb=0.4, ngen=50, poolSize=1,
                 args=None, evals=None):
        # evals: internal - return fitness
        #        external - return metrics in dict
        super().__init__(records)
        self.popSize = popSize
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.poolSize = poolSize
        self.args = args if args is not None else {}
        self.params = self.args['params'] * self.args['repeat']
        self.evals = evals
        self.init()

    def Indices(self, params):
        indices = []
        for i in range(len(params)):
            param = params[i]
            index = random.randint(0, len(param)-1)
            indices.append(index)
        return indices

    def Ind2Param(self, ind):
        params = []
        for i in range(len(ind)):
            param = self.params[i]
            params.append(param[ind[i]])
        return params
    
    def Evaluate(self, ind, eval_tag='internal'):
        params = self.Ind2Param(ind)
        try:
            metric = self.evals[eval_tag](params)
        except Exception as e:
            logger.error(f"Error evaluating individual {ind}: {e}")
            return 0,
        return metric,

    def Crossover(self, ind1, ind2):
        size = len(ind1)
        cxpoint1 = random.randint(1, size-1)
        cxpoint2 = random.randint(1, size-1)
        if cxpoint1 > cxpoint2:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        for i in range(cxpoint1, cxpoint2):
            ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def Mutate(self, ind, indpb, params):
        for i in range(len(ind)):
            if random.random() < indpb:
                param = params[i]
                j = random.randint(0, len(param)-1)
                ind[i] = j
        return ind,

    def init(self):
        super().init()
        logger.info(f"Genetic Algorithm: {self.args['desc']}")
        logger.debug(f"Initializing Genetic Algorithm with population size {self.popSize}")
        logger.debug(f"crossover probability {self.cxpb}, mutation probability {self.mutpb}")
        logger.debug(f"number of generations {self.ngen}, and pool size {self.poolSize}")
        self.ga = base.Toolbox()
        creator.create('fitness_' + str(id(self)), base.Fitness, weights=(self.args['fitness'],))
        creator.create('individual_' + str(id(self)), list, fitness=getattr(creator, 'fitness_' + str(id(self))))
        self.ga.register("indices", self.Indices, params=self.params)
        self.ga.register("individual", tools.initIterate, getattr(creator, 'individual_' + str(id(self))), self.ga.indices)
        self.ga.register("population", tools.initRepeat, list, self.ga.individual)
        self.ga.register("evaluate", self.Evaluate, eval_tag='internal')
        self.ga.register("crossover", self.Crossover)
        self.ga.register("mutate", self.Mutate, indpb=0.05, params=self.params)
        self.ga.register("select", tools.selTournament, tournsize=3)
        self.pop = self.ga.population(n=self.popSize)

        with Pool(processes=self.poolSize) as pool:
            self.fitnesses = pool.map(self.ga.evaluate, self.pop)
        for ind, fit in zip(self.pop, self.fitnesses):
            ind.fitness.values = fit

        self.err = np.inf * self.args['fitness'] * -1
        self.best = None

    def loop(self, iteration):
        super().loop(iteration)
        offspring = self.ga.select(self.pop, len(self.pop))
        offspring = list(map(self.ga.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cxpb:
                self.ga.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < self.mutpb:
                self.ga.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        with Pool(processes=self.poolSize) as pool:
            fitnesses = pool.map(self.ga.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in self.pop]
        if self.args['fitness'] < 0:
            best_idx = np.argmin(fits)
        else:
            best_idx = np.argmax(fits)
        best_fit = fits[best_idx]
        self.records.loc[len(self.records)] = {
            'iteration': iteration,
            'cur_err': best_fit,
        }
        if (best_fit - self.err) * self.args['fitness'] > 0:
            self.err = best_fit
            self.best = self.pop[best_idx]
            logger.info(f"Iteration {iteration}: Best fitness = {self.err}, Best individual = {self.best}")
            self.records.loc[len(self.records)-1, 'best_err'] = self.err
            metrics = self.Evaluate(self.best, eval_tag='external')[0]
            for key, value in metrics.items():
                logger.info(f"  Iteration {iteration}: {key} = {value}")
                self.records.loc[len(self.records)-1, key] = value
        else:
            logger.debug(f"Iteration {iteration}: No improvement, current error = {self.err}")
            self.records.loc[len(self.records)-1, 'best_err'] = self.err
            metrics = self.Evaluate(self.best, eval_tag='external')[0]
            for key, value in metrics.items():
                logger.debug(f"  Iteration {iteration}: {key} = {value}")
                self.records.loc[len(self.records)-1, key] = value
    
    def run(self):
        for iteration in range(self.ngen):
            self.loop(iteration)
        logger.info("------ Genetic Algorithm run completed ------")
        logger.info(f"  Final best fitness: {self.err}, Best individual: {self.best}")
        metrics = self.Evaluate(self.best, eval_tag='external')[0]
        for key, value in metrics.items():
            logger.info(f"  {key} = {value}")

Algo_LUT = {
    'ga': GeneticAlgorithm,
}