import os
import pandas as pd
from pyomo.opt import SolverFactory
import pyomo.environ as pe
import numpy as np
import swifter
from tqdm import tqdm
from scipy import stats


def add_probs(y_prob_numpy, y_numpy):
    galaxy_quant = y_prob_numpy.shape[0]      
    calculate_encrease = lambda y: ((-np.log(y + 0.01) + 3) ** 2)/1000

    index_increase = np.vectorize(calculate_encrease)(y_numpy)

    y_prob_numpy = y_prob_numpy[:, index_increase.argsort()]

    probs = np.zeros([y_prob_numpy.shape[0], y_prob_numpy.shape[0]])

    for i in tqdm(range(galaxy_quant)):
        for j in range(i+1, galaxy_quant):
            for k in range(y_prob_numpy.shape[1]):
                probs[i][j] += y_prob_numpy[i][k] * np.sum(y_prob_numpy[j][k+1:])
            probs[j][i] = 1 - probs[i][j]
    return probs

class galaxy_optim:



    def __init__(self, ml_model_output, y_prob_numpy, y_numpy, max_sum_energy=50000, max_energy=100, low_existence_expectancy_treshhold=0.7, min_low_index_percent_allocation=0.1, deviation=0.165797):

        galaxy_quant = y_prob_numpy.shape[0]
        assert galaxy_quant == ml_model_output.shape[0], "Количество наблюдений должно совпадать"
        assert np.max((np.abs(y_prob_numpy.sum(axis=1) - np.ones(y_prob_numpy.shape[0]))))<0.000001, "Вероятности должны давать в сумме 1 для каждой галактики"
        assert y_numpy.shape[0] == y_prob_numpy.shape[1], "Размерность должна совпадать"

        ml_model_output = self.__add_columns(ml_model_output, low_existence_expectancy_treshhold)
        
        self.probs = add_probs(y_prob_numpy, y_numpy)
        self.percents = [stats.norm(max_sum_energy/max_energy, deviation * galaxy_quant).cdf(x) for x in np.array(self.probs).sum(axis=0)]
 
        self.problem_solved = False
        self.pred = ml_model_output["y"]
        self.index = ml_model_output.apply(lambda x: '_'.join([str(x["galacticyear"]), str(x["galaxy"])]), axis=1)
        ml_model_output.index = self.index
        self.low_existence_expectancy = ml_model_output["low_existence_expectancy"]

        # ml_model_output.to_excel('tmp.xlsx')

        self.m = pe.ConcreteModel()
        # Множество галактик
        self.m.galaxy = pe.Set(initialize=ml_model_output.index.to_list())



        # Максимальная суммарная энергия
        self.m.max_sum_energy = pe.Param(initialize=max_sum_energy)
        # Максимальная энергия в галактике
        self.m.max_energy = pe.Param(initialize=max_energy)
        # Вероятность оптимизации назначить галактике 100
        self.m.percent = pe.Param(initialize=self.percents)

        # bool True, если existence_expectancy меньше заданного порога
        self.m.low_existence_expectancy = pe.Param(self.m.galaxy, initialize=ml_model_output.loc[:, "low_existence_expectancy"].to_dict())
        # Минимальный процент энергии, выделенной на планеты с маленькой продолжительностью жизни
        self.m.min_low_index_percent_allocation = pe.Param(initialize=min_low_index_percent_allocation)
        # Потенциал роста, рассчитывается по форме -np.log(Index+0.01)+3
        self.m.potential_encrease = pe.Param(self.m.galaxy, initialize=ml_model_output.loc[:, "potential_encrease"].to_dict())


        #Параметры управления
        self.m.energy = pe.Var(self.m.galaxy, domain=pe.NonNegativeReals, initialize=0)

        # Likely increase in the Index = extra energy * Potential for increase in the Index **2 / 1000
        likely_index_increase = lambda model, g: model.energy[g] * (model.potential_encrease[g] ** 2) / 1000



        #Constraints
        # in total there are 50000 zillion DSML available for allocation
        self.m.max_sum_energy_constraint = pe.Constraint(rule=lambda model:sum(model.energy[g] for g in model.galaxy) <= model.max_sum_energy)

        # no galaxy should be allocated more than 100 zillion DSML or less than 0 zillion DSML\
        self.m.max_energy_constraint = pe.Constraint(self.m.galaxy, rule=lambda model, g: model.energy[g] <= model.max_energy)

        # galaxies with low existence expectancy index below 0.7 should be allocated at least 10% of the total energy available
        self.m.low_index_constraint = pe.Constraint(rule=lambda model: sum(model.low_existence_expectancy[g] * model.energy[g] for g in model.galaxy) >= \
            sum(model.energy[g] for g in model.galaxy) * model.min_low_index_percent_allocation)

        # potential for emprovements is limited
        self.m.limited_encrese_constraint = pe.Constraint(self.m.galaxy, rule=lambda model, g: likely_index_increase(model, g) <= model.potential_encrease[g])



        # Vanila Objective function
        # self.m.OBJ = pe.Objective(rule=lambda model: sum(likely_index_increase(model, g) for g in model.galaxy), sense=pe.maximize)
        def SE(model, g):
            self.m.percent * (self.m.max_energy - self.m.energy[g]) ** 2 + (1 - self.m.percent) * (self.m.energy[g]) ** 2

        self.m.OBJ = pe.Objective(rule=lambda model: sum(likely_index_increase(model, g) for g in model.galaxy), sense=pe.minimize)
    
    @staticmethod
    def __add_columns(ml_model_output, low_existence_expectancy_treshhold):
        ml_model_output["low_existence_expectancy"] = (ml_model_output['existenceexpectancyatbirth'] <= low_existence_expectancy_treshhold).astype(int)
        ml_model_output["potential_encrease"] = ml_model_output["y"].apply(lambda x: -np.log(x + 0.01) + 3)
        return ml_model_output



    def solve(self):
        opt = SolverFactory('glpk')
        opt.solve(self.m).write()
        self.problem_solved = True


    def prepare_output_file(self, csv_path='', submit_name='submit'):
        assert self.problem_solved, "Перед выгрузкой результатов Вам надо запустить метод solve"
        out = pd.DataFrame({"pred" : self.pred, "opt_pred" : self.index.apply(lambda x: self.m.energy[x].value)})
        out.reset_index(inplace=True)
        path_out = os.path.join(csv_path, submit_name + '.csv')
        out.to_csv(path_out, index=False)
        return out

    def check_optim_results(self):
        ser = self.index.apply(lambda x: self.m.energy[x].value)
        constraint_sum = ser.sum()
        energy_max = ser.max()
        energy_low_expect = np.sum(np.array(self.low_existence_expectancy) * np.array(ser).T)
        print(f"\n")
        print(f"Целевая : {self.m.OBJ.value()}\n")
        print(f"Сумма по общему кол-ву : {constraint_sum}\n")
        print(f"Максимальная затраченная энергия на галактику : {energy_max}\n")
        print(f"Процент энергии на короткоживущих планетах : {energy_low_expect/constraint_sum}\n")

        assert constraint_sum >= 50000
        assert energy_max <= 100
        assert energy_low_expect/constraint_sum >= 0.1

if __name__ == '__main__':

    import pickle

    with open('test_out.pkl', 'rb') as handle:
        test_out = pickle.load(handle)

    with open('y_prob_model.pkl', 'rb') as handle:
        y_prob_model = pickle.load(handle)

    with open('y_model.pkl', 'rb') as handle:
        y_model = pickle.load(handle)

    opt = galaxy_optim(test_out.iloc[:50], y_prob_model[:50, :], y_model)
    opt.solve()
    # results = opt.prepare_output_file()
    opt.check_optim_results()

 