import pandas as pd
from pyomo.opt import SolverFactory
import pyomo.environ as pe
import numpy as np
import os


class galaxy_optim:
    
    def __init__(self, ml_model_output, max_sum_energy=50000, max_energy=100, low_existence_expectancy_treshhold=0.7, min_low_index_percent_allocatio=0.1) :
        
        self.problem_solved = False
        self.pred = ml_model_output["y"]

        self.index = ml_model_output.apply(lambda x : '_'.join([str(x["galactic year"]), str(x["galaxy"])]), axis=1)
        ml_model_output.index =  self.index
        ml_model_output["low_existence_expectancy"] = (ml_model_output["existence expectancy index"] <= low_existence_expectancy_treshhold).astype(int)
        ml_model_output["potential_encrease"] = ml_model_output["y"].apply(lambda x: -np.log(x + 0.01) + 3)
        
        
        self.m = pe.ConcreteModel()
        # Множество галактик
        self.m.galaxy = pe.Set(initialize=ml_model_output.index.to_list())



        # Максимальная суммарная энергия
        self.m.max_sum_energy = pe.Param(initialize=max_sum_energy)
        # Максимальная энергия в галактике
        self.m.max_energy = pe.Param(initialize=max_energy)

        # bool True, если existence_expectancy меньше заданного порога
        self.m.low_existence_expectancy = pe.Param(self.m.galaxy, initialize=ml_model_output.loc[:, "low_existence_expectancy"].to_dict())
        # Минимальный процент энергии, выделенной на планеты с маленькой продолжительностью жизни
        self.m.min_low_index_percent_allocation = pe.Param(initialize=max_energy)
        # Потенциал роста, рассчитывается по форме -np.log(Index+0.01)+3
        tmp = ml_model_output.loc[:, "potential_encrease"].to_dict()
        self.m.potential_encrease = pe.Param(self.m.galaxy, initialize=ml_model_output.loc[:, "potential_encrease"].to_dict())


        #Параметры управления
        self.m.energy = pe.Var(self.m.galaxy, domain=pe.NonNegativeReals, initialize=0)

        # Likely increase in the Index = extra energy * Potential for increase in the Index **2 / 1000
        likely_index_increase = lambda model, g : model.energy[g] * (model.potential_encrease[g] ** 2) / 1000 


      
        #Constraints
        # in total there are 50000 zillion DSML available for allocation
        self.m.max_sum_energy_constraint = pe.Constraint(rule=lambda model:sum(model.energy[g] for g in model.galaxy) <= model.max_sum_energy)

        # no galaxy should be allocated more than 100 zillion DSML or less than 0 zillion DSML\
        self.m.max_energy_constraint = pe.Constraint(self.m.galaxy, rule = lambda model, g: model.energy[g] <= model.max_energy)

        # galaxies with low existence expectancy index below 0.7 should be allocated at least 10% of the total energy available
        self.m.low_index_constraint = pe.Constraint(rule = lambda model : sum( model.low_existence_expectancy[g] * model.energy[g] for g in model.galaxy) >= \
            sum(model.energy[g] for g in model.galaxy) * model.min_low_index_percent_allocation)

        # potential for emprovements is limited
        self.m.limited_encrese_constraint = pe.Constraint(self.m.galaxy, rule = lambda model, g : likely_index_increase(model, g) <= model.potential_encrease[g])



        #Objective function
        self.m.OBJ = pe.Objective(rule=lambda model : sum(likely_index_increase(model, g) for g in model.galaxy), sense=pe.maximize)


    def solve(self):
        opt = SolverFactory('glpk')
        opt.solve(self.m).write()
        self.problem_solved = True


    def prepare_output_file(self, csv_path='', submit_name='submit'):
        assert self.problem_solved, "Перед выгрузкой результатов Вам надо запустить метод solve"
        out = pd.DataFrame({"pred" : self.pred, "opt_pred" : self.index.apply(lambda x: self.m.energy[x].value)})
        path_out = os.path.join(csv_path, submit_name + '.csv')
        out.to_csv(path_out)
        return out


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv')
    opt = galaxy_optim(train)
    opt.solve()
    results = opt.prepare_output_file()
    print(results)
