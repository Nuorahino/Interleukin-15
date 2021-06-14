import sys
from tqdm import tqdm
from SALib.sample import saltelli
from SALib.sample import morris as morris_sample
from SALib.analyze import sobol
from SALib.analyze import morris
import numpy as np
from scipy.integrate import solve_ivp
import helpfunctions


def run_sensitivity_analysis(ode, y0, variables, parameters, objective_function, t=[0, 50], method="Sobol", n=4960, print_progress=True):
    """
    Run Sensitivity analysis on the given model

    return value:
    res -- dictionary with entries according to the output of the chosen method

    Keyword arguments:
    ode -- define the ode (function)
    y0 -- initial value (double array)
    variables -- variables to run sensitivity analysis on (dict)
    parameters -- fixed values required by ode (dict)
    objective_function -- mapping result of solve_ivp to real number (function)
    t -- time intervall for the ode (double array size 2)
    method -- method for Sensitivity Analysis (string)
    n -- number of Sampelingvektors (int)
    """

    # Define the problem required by SALib
    problem = {"num_vars": len(variables), "names": list(), "bounds": list()}
    for entry in variables.items():
        problem["names"].append(entry[0])
        problem["bounds"].append(entry[1])
    # Generate Sampeling Matrix
    if (method == "Sobol"):
        X = saltelli.sample(problem, n, calc_second_order=False)
    elif (method == "Morris"):
        X = morris_sample.sample(problem, n, num_levels=4)

    Y = np.empty((len(objective_function), len(X)))

    if(print_progress):
        loop_var = tqdm(range(len(X)))
    else:
        loop_var = tqdm(range(len(X)))

    for i in loop_var:
        j = 0
        for var in variables.keys():
            parameters[var] = X[i][j]
            j += 1
        erg = solve_ivp(ode, t, y0, args=([parameters]), method="BDF", rtol=1e-6, atol=1e-6)

        if i == 0:
            print("test")
            print( erg.y[5, :]+erg.y[9, :] + erg.y[0, :])
            print("test Ende")
        for eval in range(len(objective_function)):
            Y[eval, i] = objective_function[eval](erg)

    print(Y)
    res = list()
    for i in range(len(objective_function)):
        if (method == "Sobol"):
            res.append(sobol.analyze(problem, Y[i, :], calc_second_order=False, print_to_console=False, conf_level=0.95))
        elif (method == "Morris"):
            res.append(morris.analyze(problem, Y[i, :], print_to_console=False, conf_level=0.95))


    return(res)





#    if all_parameters:
#        directory += "all/"
#    else:
#        directory += "table2/"
#    filename = "untreated"
#    if primed_IL15:
#        filename += "+IL-15"
#    if O2:
#        filename += "+O2"
#    if D:
#        filename += "+DMOG"
#    if S_3:
#        filename += "+S31-201"
#    if R:
#        filename += "+Rapa"
#    if NFkBinhi:
#        filename += "+NF-kBi"
#
#    filename += "_interval_" + str(intervall_size)
#
#    helpfunctions.create_barplot_sobol(
#        directory + "HIF/" + filename + ".png",
#        "Sobol: " + filename,
#        problem["names"],
#        Si_Y,
#    )
#    helpfunctions.create_barplot_sobol(
#        directory + "time/" + filename + ".png",
#        "Sobol: " + filename,
#        problem["names"],
#        Si_T,
#    )
#    with open(directory + "data/" + filename + ".txt", "w") as f:
#        f.write("Result for Konzentration\n")
#        f.write(str(Si_Y))
#        f.write("\nResult for Time\n")
#        f.write(str(Si_Y))
