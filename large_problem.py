import numpy as np

used_datatype = np.float64


def network_model(t, y, parameters):
    """
    Return the value of the differential equation

    Return value:
    dgl(double array size n) -- value of the ode at the specified point

    Keyword arguments:
    t(double) -- time value of the point (unused)
    y(double array size n) -- function value at point t
    parameters(dict) -- further function parameters
    """

    dgl = np.empty(shape=10, dtype=used_datatype)
    dgl[0] = parameters["a_1"] - parameters["d_1"] * y[0]  # IL-15
    dgl[1] = (parameters["a_2"] + parameters["k_1"] * y[0] + parameters["k_s"] * (y[7] ** parameters["n_2"]) / (parameters["xi_28"] ** parameters["n_2"] + y[7] ** parameters["n_2"]) - parameters["d_2"] * y[1])  # AKT
    dgl[2] = (parameters["a_3"] + parameters["k_2"] * y[1]) * parameters["alpha_1"] / (parameters["alpha_2"] + y[5]) * (1 - parameters["R"]) - parameters["d_3"] * y[2]  # mTOR
    dgl[3] = (parameters["k_alpha"] * y[8] - parameters["d_4"] * y[3] - parameters["k_13"] * parameters["k_O2"] * (parameters["delta"] * y[5] + parameters["a_11"]) * (1 - parameters["rho_6"] * parameters["D"]) * y[3] / (parameters["xi_44"] + y[3]) - parameters["k_4"] * y[3] * y[4] + parameters["k_5"] * y[5] - parameters["k_10"] * parameters["k_O2"] * parameters["phi"] * (1 - parameters["rho_6"] * parameters["D"]) * y[3] / (parameters["xi_4"] + y[3]) + parameters["k_11"] * y[9])  # HIF-1alpha protein
    dgl[4] = parameters["a_5"] - parameters["k_4"] * y[3] * y[4] + parameters["k_5"] * y[5] - parameters["d_5"] * y[4]  # HIF-1beta
    dgl[5] = parameters["k_4"] * y[3] * y[4] - parameters["k_5"] * y[5] - parameters["d_6"] * y[5]  # HIF-1 complex
    dgl[6] = (parameters["a_7"] + parameters["k_7"] * y[0] + parameters["k_14"] * y[5] + parameters["k_15"] * y[2]) * (1 - parameters["NFkBinhi"]) - parameters["d_7"] * y[6]  # NF-kB
    dgl[7] = (parameters["a_8"] + parameters["k_8"]*y[2] + parameters["k_6"]*(1-parameters["rho_4"]*parameters["D"])*y[0])*(1 - parameters["rho_3"]*parameters["S_3"]) - parameters["d_8"]*y[7]  # Stat3
    #dgl[7] = -d[7] * y[7]  # Stat3
    dgl[8] = parameters["a_9"] + parameters["k_9"] * y[6] + parameters["k_3"] * y[7] - parameters["d_9"] * y[8]  # HIF-1 alpha mRNA?
    dgl[9] = (parameters["k_10"] * parameters["k_O2"] * (1 - parameters["rho_6"] * parameters["D"]) * parameters["phi"] * y[3] / (parameters["xi_4"] + y[3]) - parameters["k_12"] * parameters["k_O2"] * (1 - parameters["rho_6"] * parameters["D"]) * (parameters["delta"] * y[5] + parameters["a_11"]) * y[9] / (parameters["xi_10"] + y[9]) - parameters["k_11"] * y[9] - parameters["d_10"] * y[9])  # HIF-1alpha-aOH

    return dgl


def get_max(erg):
    """ Returns the time (double) of the max koncentration """
    max_y = np.max(erg.y[5, :]+erg.y[9, :] + erg.y[0, :])
    return max_y


def get_time_of_max(erg):
    """ Returns the time (double) of the max koncentration """
    max_y = np.max(erg.y[5, :]+erg.y[9, :] + erg.y[0, :])
    max_t = erg.t[np.where(erg.y[5, :]+erg.y[9, :] + erg.y[0, :] == max_y)[0][0]]
    return max_t


def get_parameters(variables, p=0.25, primed_IL15=False, O2=False, D=0, S_3=0, R=0, NFkBinhi=0):
    """
    Return the parameters for a given Test condition

    Return values:
    y0(double array) -- Initial condition
    bounds(dict) -- mapping the variables to an array with its bounds
    args(dict -- mapping additional parameters to a fixed value

    Keyword arguments:
    variables(string array) -- names of the variables
    primed_IL15(bool, default False) -- external regulation of IL-15
    O2(bool, default False) -- cells were cultivated in normoxia
    D(double default 0) -- presence of DMOG
    S_3(double default 0) -- presence of S31_201
    R(double default 0) -- presence of Rapamycin
    NFkBinhi(double default 0) -- presence of NFkappa Inhibitor
    """

    # Initialize Initial condition
    y0 = [0, 1, 1, 0.05, 1, 0.05, 1, 1, 1, 0.9]
    if primed_IL15:
        y0[0] = 1
    # Initialize Parameter values
    args = {
        "rho_6": 0.99,
        "D": D,
        "S_3": S_3,
        "R": R,
        "NFkBinhi": NFkBinhi,

        "a_1": 0,
        "a_2": 0.848,
        "a_3": 0.037,
        "a_5": 0.211,
        "a_7": 0,
        "a_8": 0,
        "a_9": 0,
        "a_11": 4.17,
        "k_1": 2e-5,
        "k_2": 0.307,
        "k_3": 0.181,
        "k_4": 76.196,
        "k_5": 75.895,
        "k_6": 25.001,
        "k_7": 2.903,
        "k_8": 0.577,
        "k_9": 0.753,
        "k_10": 421.353,
        "k_11": 0.211,
        "k_12": 0.061,
        "k_13": 12.152,
        "k_14": 16.528,
        "k_15": 0.088,
        "d_1": 0.062,
        "d_2": 0.848,
        "d_3": 0.919,
        "d_4": 0.623,
        "d_5": 0.196,
        "d_6": 0.301,
        "d_7": 0.914,
        "d_8": 0.577,
        "d_9": 0.934,
        "d_10": 0.935,
        "rho_3": 1,
        "rho_4": 0.863,
        "k_alpha": 1.034,
        "xi_10": 8.127,
        "xi_4": 15.018,
        "xi_28": 38.44,
        "xi_44": 128.022,
        "delta": 200,
        "k_s": 9e-4,
        "n_2": 2,
        "phi": 1.163,
        "alpha_1": 1.163,
        "alpha_2": 0.386
    }
    if O2:
        args["k_O2"] = 0.06  # (19) Hypoxia
    else:
        args["k_O2"] = 0.96  # Supplementary Material: S1

    bounds = dict()

    for entry in variables:
        bounds[entry] = args.pop(entry)
    bounds = get_parameter_intervalls(bounds, p)

    return y0, bounds, args


def get_parameter_intervalls(variables, p):
    """
    Transform the variables into a SAlib problem

    Return value:
    problem(dict) -- bounds for the variables

    Keyword arguments:
    variables(dict) -- variables and their estimated value
    p(double) -- percentage of max deviation from the estimated value
    """

    bounds = dict()
    for entry in variables.items():
        if entry[1] == 0:
            bounds[entry[0]] = [0, 0.1]
        else:
            bounds[entry[0]] = [max(0, (1 - p) * entry[1]), (1 + p) * entry[1]]

    return bounds
