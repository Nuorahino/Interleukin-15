import matplotlib.pyplot as plt
from tqdm import tqdm
from SALib.sample import saltelli
from SALib.sample import morris as morris_sample
from SALib.analyze import sobol
from SALib.analyze import morris
import numpy as np
from scipy.integrate import solve_ivp


def run_sensitivity_analysis(ode, y0, variables, parameters, objective_function, t=[0, 50], method="Sobol", n=4960, use_log_scale=False, print_progress=True):
    """
    Run Sensitivity analysis on the given model

    return value:
    res(dict) -- Output the Result of SALib analyze

    Keyword arguments:
    ode(callable) -- define the ode
    y0(array double) -- initial value
    variables(dict) -- variables for the sensitivity analysis
    parameters(dict) -- fixed values required by ode
    objective_function(callable) -- maps result of solve_ivp to a real number
    t(double array size 2, default: [0, 50]) -- time intervall for the ode
    method(string, default: "Sobol") -- method for Sensitivity Analysis
    n(int) -- number of Sampelingvektors
    use_log_scale(bool) -- Variables should be analyzed based on log scale
    print_progress(bool) -- display a progress bar
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

    # Compute the function values for different inputs
    Y = np.empty((len(objective_function), len(X)))
    if(print_progress):
        loop_var = tqdm(range(len(X)))
    else:
        loop_var = range(len(X))

    for i in loop_var:
        j = 0
        for var in variables.keys():
            if use_log_scale:
                parameters[var] = 10 ** X[i][j]  # The base of the log scale could be changed here
            else:
                parameters[var] = X[i][j]
            j += 1
        erg = solve_ivp(ode, t, y0, args=([parameters]), method="BDF", rtol=1e-6, atol=1e-6)  # Change the IVP solver here

        for eval in range(len(objective_function)):
            Y[eval, i] = objective_function[eval](erg)

    # Run Sensitivity Analysis on the result
    res = list()
    for i in range(len(objective_function)):
        if (method == "Sobol"):
            res.append(sobol.analyze(problem, Y[i, :], calc_second_order=False, print_to_console=False, conf_level=0.95))
        elif (method == "Morris"):
            res.append(morris.analyze(problem, X, Y[i, :], print_to_console=False, conf_level=0.95))

    return(res)


def create_barplot_morris(filename, title, Si, names):
    """
    plots the result of the Morris Sensitivity Analysis

    Return value: None

    Keyword arguments:
    filename(string) -- filename of the destination file for the plot
    title(string) -- title of the plot
    Si(numpy.ndarray) -- result of the morris analysis
    """

    xpos = 2 * np.arange(len(names))
    plt.bar(xpos, Si["mu_star"], width=1)
    plt.xticks(xpos, names, fontsize=8, rotation=90)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def create_barplot_sobol(filename, title, Si, names):
    """
    plots the result of the Sobol Sensitivity Analysis

    Return value: None

    Keyword arguments:
    filename(string) -- filename of the destination file for the plot
    title(string) -- title of the plot
    Si(numpy.ndarray) -- result of the Sobol analysis
    """

    xpos = np.arange(len(names))
    plt.bar(xpos - 0.2, Si["S1"], width=0.4, label="First order Sensitivity")
    plt.bar(xpos + 0.2, Si["ST"], width=0.4, label="Total Sensitivity")
    plt.xticks(xpos, names, fontsize=8, rotation=90)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
