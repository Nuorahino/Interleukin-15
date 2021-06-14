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
    parameters -- further function parameters
    """

    dgl = np.empty(shape=5, dtype=used_datatype)
    dgl[0] = -parameters["d_i"] * y[0]  # IL-15
    dgl[1] = (1 - parameters["NFkBinhi"]) * (parameters["a_n"] + parameters["a_mn"] * y[3] + parameters["a_hn"] * y[4]) - parameters["d_n"] * y[1]   # NFkB
    dgl[2] = (1 - parameters["S_3"]) * (parameters["a_s"] + parameters["a_ms"] * y[3] - parameters["d_s"] * y[2])  # STAT3
    dgl[3] = (1 - parameters["R"]) * (parameters["a_m"] + parameters["a_sm"] * y[2])/(parameters["b_m"] + y[4]) - parameters["d_m"] * y[3]  # mTOR
    dgl[4] = parameters["a_h"] - parameters["d_h"] * y[4] - (1 - parameters["D"]) * parameters["d_O2"] * y[4] + parameters["a_sh"] * y[2] + parameters["a_nh"] * y[1]  # HIF-1

    return dgl


def get_max(erg):
    """ Returns the time (double) of the max koncentration """
    max_y = np.max(erg.y[4, :])
    return max_y


def get_time_of_max(erg):
    """ Returns the time (double) of the max koncentration """
    max_y = np.max(erg.y[4, :])
    max_t = erg.t[np.where(erg.y[4, :] == max_y)[0][0]]
    return max_t


# Change the name to estimated
# def get_parameters(variables, p=0.25, primed_IL15=False, O2=False, D=0, S_3=0, R=0, NFkBinhi=0):
#    """
#    Return the parameters for a given Test condition
#
#    Return values:
#    y0(double array) -- Initial condition
#    bounds(dict) -- mapping the variables to an array with its bounds
#    args(dict) -- mapping additional parameters to a fixed value
#
#    Keyword arguments:
#    variables(string array) -- names of the variables
#    p (double default 0.25) -- max deviation from the estimated value
#    primed_IL15(bool, default False) -- external regulation of IL-15
#    O2(bool, default False) -- cells were cultivated in normoxia
#    D(double default 0) -- presence of DMOG
#    S_3(double default 0) -- presence of S31_201
#    R(double default 0) -- presence of Rapamycin
#    NFkBinhi(double default 0) -- presence of NFkappa Inhibitor
#    """
#
#    # Initialize Initial condition
#    y0 = [0, 1, 1, 1, 1]
#    if primed_IL15:
#        y0[0] = 1
#    # Initialize Parameter values
#    args = {
#        "D": D,
#        "S_3": S_3,
#        "R": R,
#        "NFkBinhi": NFkBinhi,
#
#        "d_i": 0.062,
#        "a_mn": 0.088,
#        "a_h": ,
#        "a_hn": 16.528,
#        "a_n": 0,
#        "d_n": 0.914,
#        "d_s": 0.577,
#        "a_ms": 0.577,
#        "a_s": 0,
#        "d_m": 0.919,
#        "a_m": 0.037,
#        "a_sm": 0.307, # unsure
#        "b_m": 0.386,
#
#        "a_k":,
#        "d_h":,
#        "d_O2": 0.96,
#        "a_sh":,
#        "a_nh":,
#    }
#    bounds = dict()
#
#    for entry in variables:
#        bounds[entry] = args.pop(entry)
#    bounds = get_parameter_intervalls(bounds, p)
#
#    return y0, bounds, args


def get_parameters_log(primed_IL15=False, D=0, S_3=0, R=0, NFkBinhi=0):
    """
    Return the parameters for a given Test condition

    Return values:
    y0(double array) -- Initial condition
    bounds(dict) -- mapping the variables to an array with its bounds
    args(dict -- mapping additional parameters to a fixed value

    Keyword arguments:
    primed_IL15(bool, default False) -- external regulation of IL-15
    D(double default 0) -- presence of DMOGcondition[0]
    S_3(double default 0) -- presence of S31_201
    R(double default 0) -- presence of Rapamycin
    NFkBinhi(double default 0) -- presence of NFkappa Inhibitor
    """

    # Initialize Initial condition
    y0 = [0, 1, 1, 1, 1]
    if primed_IL15:
        y0[0] = 1
    # Initialize Parameter values
    args = {
        "D": D,
        "S_3": S_3,
        "R": R,
        "NFkBinhi": NFkBinhi,
        "d_O2": 0.96
        }

    bounds = {
        "d_i": [-2, 0],
        "a_mn": [-2, 0],
        "a_h": [-2, 0],
        "a_hn": [-2, 0],
        "a_n": [-2, 0],
        "d_n": [-2, 0],
        "d_s": [-2, 0],
        "a_ms": [-2, 0],
        "a_s": [-2, 0],
        "d_m": [-2, 0],
        "a_m": [-2, 0],
        "a_sm": [-2, 0],
        "b_m": [-2, 0],

        "a_k": [-2, 0],
        "d_h": [-2, 0],
        "a_sh": [-2, 0],
        "a_nh": [-2, 0]
    }

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
