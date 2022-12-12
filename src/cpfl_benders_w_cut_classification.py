import gurobipy as gp
import numpy as np 
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import os

import argparse
from tqdm import tqdm

from cpfl_data import read_beasley98, read_scenarios


### Master Problem ###
def initialize_master_problem(instance):
    p = [1.0/instance['numScenario']] * instance['numScenario']

    MP = gp.Model('MP')
    MP.Params.outputFlag = 0
    MP.Params.method = 1 # dual simplex

    # Fist-stage variables
    x = MP.addVars(range(instance['numLocation']), obj = instance['fixed_cost'], vtype=gp.GRB.BINARY, name='x') 
    n = MP.addVars(range(instance['numScenario']), obj = p, vtype=gp.GRB.CONTINUOUS, name='n') 

    MP.modelSense = gp.GRB.MINIMIZE
    MP.update()

    return MP

### Sub Problem ###
def initialize_sub_problem(instance):
    SP = gp.Model('SP')
    SP.Params.outputFlag = 0


    # Secon-stage variables
    y = SP.addVars(range(instance['numLocation']), range(instance['numCustomer']), vtype = gp.GRB.CONTINUOUS, obj = instance['allocation_cost'], name = 'y')
    z = SP.addVars(range(instance['numCustomer']), vtype = gp.GRB.CONTINUOUS, obj = [instance['penalty_cost']] * instance['numCustomer'] , name = 'z')

    # linking demand
    demandConstr = []
    for j in range(instance['numCustomer']):
        demandConstr.append(SP.addConstr(gp.quicksum(y[i,j] for i in range(instance['numLocation'])) + z[j] >= instance['demand'][j], name = 'demandConstr'))
    # capacity constraints
    capacityConstr = []
    for i in range(instance['numLocation']):
        capacityConstr.append(SP.addConstr(gp.quicksum(y[i,j] for j in range(instance['numCustomer'])) <= instance['capacity'][i], name = 'capacityConstr'))

    SP.modelSense = gp.GRB.MINIMIZE
    SP.update()

    return SP


### Modify sub problem ###
def ModifyAndSolveSP(SP, instance, nsol_s, s, xsol, CutViolationTolerance = 0.0001):
    demandConstr = [c for c in SP.getConstrs() if "demandConstr" in c.ConstrName]
    capacityConstr = [c for c in SP.getConstrs() if "capacityConstr" in c.ConstrName]

    # print(SP.getConstrs())
    # print(len(demandConstr))
    # print(instance['numCustomer'])
    
    # print(instance['demand_scenarios'])
    for j in range(instance['numCustomer']):
        demandConstr[j].rhs = instance['demand_scenarios'][j][s]
    for i in range(instance['numLocation']):
        capacityConstr[i].rhs = instance['capacity'][i] * xsol[i]
    # demandConstr.rhs = instance['demand_scenarios'].iloc[:, s]
    # capacityConstr.rhs = [capacity[i] * xsol[i] for i in range(instance['numLocation'])]

    SP.update()
    SP.optimize()

    pi_sol = [demandConstr[j].Pi for j in range(instance['numCustomer'])]
    gamma_sol = [capacityConstr[i].Pi for i in range(instance['numLocation'])]
    
    SPobj = SP.objVal

    # Check whether a violated Benders cut is found
    CutFound_s = False
    if(nsol_s < SPobj - CutViolationTolerance): # Found Benders cut is violated at the current master solution
        CutFound_s = True

    return SPobj, CutFound_s, pi_sol, gamma_sol


### Benders Loop ###
def run_benders_loop(instance, iterLimit, CutViolationTolerance, prediction_threshold = 0.3, min_num_cut_pct = 0.5):
    start_time = time.time()

    xgb_model = pkl.load(open('/Users/chang/PhD_workplace/MIE1612/project/src/notebooks/xgb_model.pkl', "rb"))

    p = [1.0/instance['numScenario']] * instance['numScenario']

    CutFound = True
    NoIters = 0
    BestLB = 0

    UBs = []
    LBs = []
    NoCuts = []

    # Solve MP
    MP = initialize_master_problem(instance)

    cut_Matrix = np.zeros((iterLimit, instance['numScenario']))
    runtime_iter = []
    while(CutFound and NoIters < iterLimit):
        sub_start_time = time.time()
        print(f'**********Iteration {NoIters}***********')
        CutFound = False
        
        MP.update()
        MP.optimize()
        # print('sheeeep')
        # print(MP.display())

        x = [var for var in MP.getVars() if "x" in var.VarName]
        n = [var for var in MP.getVars() if "n" in var.VarName]
        
        # Get MP solution
        MPobj = MP.objVal
        # print('MPobj: ', MPobj)
        
        UBs.append(MPobj)
        
        xsol = [x[i].x for i in range(instance['numLocation'])]            
        nsol = [n[s].x for s in range(instance['numScenario'])]

        LB = 0
        CurCuts = 0
        
        SP = initialize_sub_problem(instance)

        cuts_info_list = pd.DataFrame()
        expr_list = []
        for s in range(instance['numScenario']):
            Qvalue, CutFound_s, pi_sol, gamma_sol = ModifyAndSolveSP(SP, instance, nsol[s], s, xsol, CutViolationTolerance)
            # print('Qvalue: ', Qvalue)
            # print('CutFound_s: ', CutFound_s)

            LB += p[s] * Qvalue

            if(CutFound_s):
                CutFound = True
                cut_Matrix[NoIters][s] = 1
                rhs = gp.quicksum(instance['demand_scenarios'][j][s] * pi_sol[j] for j in range(instance['numCustomer'])) + gp.quicksum(instance['capacity'][i] * x[i] * gamma_sol[i] for i in range(instance['numLocation']))

                expr = gp.LinExpr(n[s] - rhs)
                expr_list.append(expr)
                # MP.addConstr(expr >= 0, name = 'cuts ')
                # MP.update()

                # Get cut information
                past_num_cut = 0# how many cuts has this subproblem generated in all of previous iterations
                previous_cut_bool = False # has this subproblem generated a cut in the previous iteration
                previous_cut_all_bool = False# has this subproblem generated any cut in all of previous iterations

                if NoIters > 0:
                    past_num_cut = cut_Matrix[:,s].sum()
                    if past_num_cut > 0:
                        previous_cut_all_bool = True
                    if cut_Matrix[NoIters -1][s] > 0:
                        previous_cut_bool = True

                cut_info = {
                    # 'scenario_num'    : s,
                    'iteration'     : NoIters,
                    'numCustomer'   : instance['numCustomer'],
                    'numLocation'   : instance['numLocation'],
                    'tot_num_scenario'  : instance['numScenario'],
                    'past_num_cut'  : past_num_cut,
                    'previous_cut_all_bool' : previous_cut_all_bool,
                    'previous_cut_bool' : previous_cut_bool,
                    'sub_obj_gap'   : ((Qvalue - nsol[s])/Qvalue),
                    'pct_non_zero_in_gamma' : (gamma_sol.count(0)/len(gamma_sol)),
                    # 'master_nsol'   : nsol[s],
                    # 'sub_optimal_q_value'   : Qvalue,
                }
                cuts_info_list = cuts_info_list.append(cut_info, ignore_index=True)
                
        # predict on the validity of cuts
        if CutFound:
            cut_pred_proba = xgb_model.predict_proba(cuts_info_list)[:,1]
            # if NoIters == 0:
            #     min_num_cut = max(round(min_num_cut_pct * len(cuts_info_list)), round(0.05 * len(cuts_info_list)))
            # else:
            min_num_cut = max(round(min_num_cut_pct * len(cuts_info_list)), 1)
            # print(sum(prob > prediction_threshold for prob in cut_pred_proba))
            # print(min_num_cut)
            if sum(prob > prediction_threshold for prob in cut_pred_proba) < min_num_cut:
                indices = sorted(range(len(cut_pred_proba)), key=lambda i: cut_pred_proba[i], reverse=True)[:min_num_cut]
                # print(indices)
                for i in indices:
                    # print(expr_list[i])
                    MP.addConstr(expr_list[i] >= 0, name = 'cuts ')
                    MP.update()
                    CurCuts += 1
            else:
                for i,prob in enumerate(cut_pred_proba):
                    if prob >= prediction_threshold:
                        # add the cut
                        MP.addConstr(expr_list[i] >= 0, name = 'cuts ')
                        MP.update()
                        CurCuts += 1

        if (LB > BestLB):
            BestLB = LB
    
        LBs.append(BestLB)
        NoCuts.append(CurCuts)

        NoIters += 1
        sub_end_time = time.time()
        runtime_iter.append(sub_end_time - sub_start_time)


    end_time = time.time()
    runtime = end_time - start_time

    # print output to file
    # with open("output_singleCut.txt", "w") as f:
    #     print('***** SINGLECUT *****', file=f)
    #     print(f'Runtime: {runtime}', file=f)
    #     print(f'Total number of cuts added: {sum(NoCuts)}', file=f)
    #     print(f'Number of iterations: {NoIters}', file=f)
    #     print(f'Objective Value: {BestLB}', file=f)
    #     print('------------------')
    #     for i in range(NoIters):
    #         print(f'Iteration {i}: LB: {LBs[i]}, UB: {UBs[i]}', file=f)
        
        
    # out = pd.DataFrame({'LB': LBs, 'UB': UBs, 'NoCuts': NoCuts})
    # out.to_csv('output_singleCut.csv')

    print('\nOptimal Solution:')
    print(f'MPobj: {MPobj}')
    print(f'xsol: {str(xsol)}')
    # print(f'nsol: {str(nsol)}')
    print(f'NoIters: {str(NoIters)}')
    print(f'NoCuts: {sum(NoCuts)}')
    print(f'Runtime: {runtime}')
    # print('penalty_cost: ', str(instance['penalty_cost']))

    return MPobj, NoIters, sum(NoCuts), runtime_iter, runtime


def solve_cflp_benders(*args):
    args = args[0]

    data_path = args.data_path
    file_name = args.file_name
    iterLimit = args.iterLimit
    CutViolationTolerance = args.CutViolationTolerance
    numScenario = args.numScenario
    train_or_test = args.train_or_test
    min_num_cut_pct = args.min_num_cut_pct

    # scenario_path = (f'{data_path}/scenarios/{file_name}_scenarios_100.csv')

    # instance = read_beasley98(data_path, train_or_test, file_name)
    # instance['demand_scenarios'] = read_scenarios(scenario_path).to_numpy()
    # instance['numScenario'] = numScenario

    # print(instance['numScenario'])

    result_df = pd.DataFrame()
    for file in tqdm(os.listdir(data_path+'/test')):
        if not file.startswith('.'):
            file_name = file[:-4]
            scenario_path = (f'{data_path}/scenarios/{file_name}_scenarios_100.csv')

            instance = read_beasley98(data_path, train_or_test, file_name)
            instance['demand_scenarios'] = read_scenarios(scenario_path).to_numpy()
            instance['numScenario'] = numScenario

            print(file_name, ', ', instance['numCustomer'], ', ', instance['numLocation'])

            print('*****RUNNING BENDERS LOOP*****')
            MPobj, NoIters, NoCuts, runtime_iter, runtime = run_benders_loop(instance, iterLimit, CutViolationTolerance, min_num_cut_pct=min_num_cut_pct)

            result_df = result_df.append({'instance':file_name, 
                                        'MPobj': MPobj, 
                                        'NoIters': NoIters, 
                                        'NoCuts':NoCuts, 
                                        'Runtime':runtime, 
                                        'runtime_iter':runtime_iter}, ignore_index=True)

            with open(f'result_{file_name}_w_cut_clf_cutpct50.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow([MPobj, NoIters, NoCuts, runtime])
            
            with open(f'result_{file_name}_w_cut_clf_cutpct50_runtime_iter.csv', 'w') as f:
                write = csv.writer(f)
                write.writerow(runtime_iter)

    result_df.to_csv(f'project/results/results_all_w_cut_clf_cutpct{min_num_cut_pct}.csv', index = False)     

    # MPobj, NoIters, NoCuts, runtime_iter, runtime = run_benders_loop(instance, iterLimit, CutViolationTolerance)
    
    # with open(f'result_{file_name}_w_cut_clf.csv', 'w') as f:
    #     write = csv.writer(f)
    #     write.writerow([MPobj, NoIters, NoCuts, runtime])
    
    # with open(f'result_{file_name}_w_cut_clf_runtime_iter.csv', 'w') as f:
    #     write = csv.writer(f)
    #     write.writerow(runtime_iter)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='/Users/chang/PhD_workplace/MIE1612/project/data/beasley98') 
    parser.add_argument("--file_name", type=str, default='cap72')
    parser.add_argument("--iterLimit", type=int, default=500) 
    parser.add_argument("--CutViolationTolerance", type=float, default=0.0001)
    parser.add_argument("--numScenario", type=float, default=100)
    parser.add_argument("--train_or_test", type=str, default='test')
    parser.add_argument("--min_num_cut_pct", type=float, default='0.5')

    args, remaining = parser.parse_known_args()

    solve_cflp_benders(args)