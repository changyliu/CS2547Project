import gurobipy as gp
import numpy as np 
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
import argparse
import os

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
def run_benders_loop(instance, iterLimit, CutViolationTolerance):
    start_time = time.time()

    p = [1.0/instance['numScenario']] * instance['numScenario']

    CutFound = True
    NoIters = 0
    BestLB = 0

    UBs = []
    LBs = []
    NoCuts = []

    # Solve MP
    MP = initialize_master_problem(instance)

    # MP.update()
    # MP.optimize()
    
    cut_info_df = pd.DataFrame()
    scenario_list = []
    pi_sol_list = []
    gamma_sol_list = []
    cut_Matrix = np.zeros((iterLimit, instance['numScenario']))

    while(CutFound and NoIters < iterLimit):
        print(f'**********Iteration {NoIters}***********')
        CutFound = False
        
        MP.update()
        MP.optimize()
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

        for s in range(instance['numScenario']):
            Qvalue, CutFound_s, pi_sol, gamma_sol = ModifyAndSolveSP(SP, instance, nsol[s], s, xsol, CutViolationTolerance)
            # print('Qvalue: ', Qvalue)
            # print('CutFound_s: ', CutFound_s)

            LB += p[s] * Qvalue

            if(CutFound_s):
                CutFound = True
                cut_Matrix[NoIters][s] = 1
                # MP_obj_before = MP.objVal

                rhs = gp.quicksum(instance['demand_scenarios'][j][s] * pi_sol[j] for j in range(instance['numCustomer'])) + gp.quicksum(instance['capacity'][i] * x[i] * gamma_sol[i] for i in range(instance['numLocation']))

                expr = gp.LinExpr(n[s] - rhs)
                scenario_list.append(s)
                pi_sol_list.append(pi_sol)
                gamma_sol_list.append(gamma_sol)
                MP.addConstr(expr >= 0, name = 'cuts ')

                # MP.update()
                # MP.optimize()
                # MP_obj_after = MP.objVal

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
                    'scenario_num'    : s,
                    'iteration'     : NoIters,
                    'instance'      : instance['instance_name'],
                    'numCustomer'   : instance['numCustomer'],
                    'numLocation'   : instance['numLocation'],
                    'tot_num_scenario'  : instance['numScenario'],
                    'past_num_cut'  : past_num_cut,
                    'previous_cut_all_bool' : previous_cut_all_bool,
                    'previous_cut_bool' : previous_cut_bool,
                    'sub_obj_gap'   : ((Qvalue - nsol[s])/Qvalue),
                    'pct_non_zero_in_gamma' : (gamma_sol.count(0)/len(gamma_sol)),
                    'master_nsol'   : nsol[s],
                    'sub_optimal_q_value'   : Qvalue,
                    # 'MP_obj_before' : MP_obj_before,
                    # 'MP_obj_after'  : MP_obj_after
                }

                cut_info_df = cut_info_df.append(cut_info, ignore_index=True)

                CurCuts += 1
            
        if (LB > BestLB):
            BestLB = LB
    
        LBs.append(BestLB)
        NoCuts.append(CurCuts)

        NoIters += 1


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
    print(f'runtime: {runtime}')
    # print('penalty_cost: ', str(instance['penalty_cost']))

    return cut_info_df, scenario_list, pi_sol_list, gamma_sol_list

def get_MP_obj(instance, scenario_list, pi_sol_list, gamma_sol_list):
    MP = initialize_master_problem(instance)

    x = [var for var in MP.getVars() if "x" in var.VarName]
    n = [var for var in MP.getVars() if "n" in var.VarName]

    MP.update()
    MP.optimize()
    MP_obj = [MP.objVal]
    # print(MP_obj)

    for (s, pi_sol, gamma_sol) in tqdm(zip(scenario_list, pi_sol_list, gamma_sol_list), total=len(scenario_list)):
        # print(s)
        # print(pi_sol)
        # print(gamma_sol)
        rhs = gp.quicksum(instance['demand_scenarios'][j][s] * pi_sol[j] for j in range(instance['numCustomer'])) + gp.quicksum(instance['capacity'][i] * x[i] * gamma_sol[i] for i in range(instance['numLocation']))
        expr = gp.LinExpr(n[s] - rhs)
        MP.addConstr(expr >= 0, name = 'cuts ')
        MP.update()
        MP.optimize()
        MP_obj.append(MP.objVal)
    
    cut_label_df = pd.DataFrame({'MP_obj_before': MP_obj[:-1], 'MP_obj_after': MP_obj[1:]})

    return cut_label_df


def solve_cflp_benders(*args):
    args = args[0]

    data_path = args.data_path
    # file_name = args.file_name
    iterLimit = args.iterLimit
    CutViolationTolerance = args.CutViolationTolerance
    numScenario = args.numScenario

    # cut_info_df_all = pd.DataFrame()
    # print(os.listdir(data_path+'/train'))
    for file in tqdm(os.listdir(data_path+'/train')[27:28]):
        if not file.startswith('.'):
            file_name = file[:-4]
            scenario_path = (f'{data_path}/scenarios/{file_name}_scenarios_100.csv')

            instance = read_beasley98(data_path, file_name)
            instance['demand_scenarios'] = read_scenarios(scenario_path).to_numpy()
            instance['numScenario'] = numScenario

            print(file_name, ', ', instance['numCustomer'], ', ', instance['numLocation'])

            print('*****RUNNING BENDERS LOOP*****')
            cut_info_df, scenario_list, pi_sol_list, gamma_sol_list = run_benders_loop(instance, iterLimit, CutViolationTolerance)

            print('*****COMPUTING RMP OBJECTIVE CHANGE*****')
            cut_label_df = get_MP_obj(instance, scenario_list, pi_sol_list, gamma_sol_list)

            cut_info_df.to_csv(f'cpfl_cuts_info_{file_name}_features.csv', index = False)
            cut_label_df.to_csv(f'cpfl_cuts_info_{file_name}_labels.csv', index = False)

            # cut_info_df_all = cut_info_df_all.append(cut_info_df)
    
    # cut_info_df_all.to_csv('cpfl_cuts_info.csv', index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='/Users/chang/PhD_workplace/MIE1612/project/data/beasley98') 
    # parser.add_argument("--file_name", type=str, default='cap41')
    parser.add_argument("--iterLimit", type=int, default=50) 
    parser.add_argument("--CutViolationTolerance", type=float, default=0.0001)
    parser.add_argument("--numScenario", type=float, default=100)

    args, remaining = parser.parse_known_args()

    solve_cflp_benders(args)