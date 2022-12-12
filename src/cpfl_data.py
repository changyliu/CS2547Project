import numpy as np
import pandas as pd
import random

import os
from tqdm import tqdm

def read_beasley98(instance_path, train_or_test, instance_name,):
    file_path = f'{instance_path}/{train_or_test}/{instance_name}.txt'

    f = open(file_path, "r")

    instance = {}
    instance['instance_name'] = instance_name
    line = f.readline().split()
    instance['numLocation'] = int(line[0])
    instance['numCustomer'] = int(line[1])

    capacity = []
    fixed_cost = []
    for l in range(instance['numLocation']):
        line = f.readline().split()
        capacity.append(float(line[0]))
        fixed_cost.append(float(line[1]))
    
    instance['capacity'] = capacity
    instance['fixed_cost'] = fixed_cost

    # print(instance['capacity'])
    demand = []
    allocation_cost = []
    for c in range(instance['numCustomer']):
        d = f.readline()
        demand.append(int(d))
        allocation_cost_c = []
        counter = 0
        while counter < instance['numLocation']:
            line = f.readline().split()
            # print(line)
            for i in line:
                allocation_cost_c.append(float(i))
                counter += 1
            # print(counter)
        allocation_cost.append(allocation_cost_c)
    
    instance['demand'] = demand
    instance['allocation_cost'] = allocation_cost

    # generate penalty cost based on allocation costs
    instance['penalty_cost'] = np.percentile(instance['allocation_cost'], 60)

    return instance

def generate_scenarios(instance_path, train_or_test, instance_name, num_scenarios, rand_seed = 2022):
    # instance_path = f'{instance_path}/instances'
    instance = read_beasley98(instance_path, train_or_test, instance_name)
    file_name = instance_name + '_scenarios_' + str(num_scenarios)

    demand_mean = np.mean(instance['demand'])
    random.seed(rand_seed)
    std_factor = random.uniform(0.1, 0.2)
    demand_scenarios = []

    for _ in range(instance['numCustomer']):
        demand_scenarios.append(np.random.normal(demand_mean, demand_mean * std_factor, num_scenarios))
    
    demand_scenarios_df = pd.DataFrame(demand_scenarios)

    #####
    # assume penalty of 1 per unit
    #####

    demand_scenarios_df.to_csv(f'{instance_path}/scenarios/{file_name}.csv', index = False, header = False)

def read_scenarios(file_path):

    return pd.read_csv(file_path, header=None)


# instance = read_beasley98('/Users/chang/PhD_workplace/MIE1612/project/data/beasley98/cap41.txt')
# print(instance)
# data_path = '/Users/chang/PhD_workplace/MIE1612/project/data/beasley98'
# for file in tqdm(os.listdir(data_path+'/instances')):
#     file_name = file[:-4]
#     print(file_name)
#     generate_scenarios(data_path, file_name, 100)
        
