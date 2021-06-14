import numpy as np
import matplotlib.pyplot as plt

used_datatype = np.float64


def network_model(y, a, d, k, k_s, k_O2, k_alpha, alpha, n_2, xi_4, xi_10, xi_28, xi_44, delta, phi, p_3, p_4, p_6, D, R, S_3, NFkBinhi):
    dgl     = np.empty(shape=10, dtype = used_datatype)
    dgl[0]  = a[0] - d[0] * y[0]                                                                                                                                                            #IL-15
    dgl[1]  = a[1] + k[0]*y[0] + k_s*(y[7]**n_2)/(xi_28**n_2 + y[7]**n_2) - d[1]*y[1]                                                                                                       #AKT
    dgl[2]  = (a[2] + k[1]*y[1]) * alpha[0]/(alpha[1]+y[5])*(1-R) - d[2]*y[2]                                                                                                               #mTOR
    dgl[3]  = k_alpha*y[8] - d[3]*y[3] - k[12]*k_O2*(delta*y[5] + a[10])*(1-p_6*D)*y[3]/(xi_44+y[3]) - k[3]*y[3]*y[4] + k[4]*y[5]- k[9]*k_O2*phi*(1-p_6*D)*y[3]/(xi_4 + y[3]) + k[10]*y[9]  #HIF-1alpha protein
    dgl[4]  = a[4] - k[3]*y[3]*y[4] + k[4]*y[5] - d[4]*y[4]                                                                                                                                 #HIF-1beta
    dgl[5]  = k[3]*y[3]*y[4] - k[4]*y[5] - d[5]*y[5]                                                                                                                                        #HIF-1 complex
    dgl[6]  = (a[6] + k[6]*y[0] + k[13]*y[5] + k[14]*y[2])*(1-NFkBinhi) - d[6]*y[6]                                                                                                         #NF-kB
    #dgl[7]  = (a[7] + k[7]*y[2] + k[5]*(1-p_4*D)*y[0])*(1 - p_3*S_3) - d[7]*y[7]                                                                                                            #Stat3
    dgl[7]  = - d[7]*y[7]                                                                                                            #Stat3
    dgl[8]  = a[8] + k[8]*y[6] + k[2]*y[7] - d[8]*y[8]                                                                                                                                      #HIF-1 alpha mRNA?
    dgl[9]  = k[9]*k_O2*(1-p_6*D)*phi*y[3]/(xi_4 + y[3]) - k[11]*k_O2*(1-p_6*D)*(delta*y[5] + a[10])*y[9]/(xi_10 + y[9]) -k[10]*y[9] - d[9]*y[9]                                            #HIF-1alpha-aOH

    return dgl


# Default is that the cells are not treated, and have normal amount of oxygen
def parameterized_model(primed_IL15 = False, O2 = True, D = 0, S_3 = 0, R = 0, NFkBinhi = 0):
    #Initialize Initial condition
    #y0 = [0,1,1,0,1,0,1,1,1,0]
    y0 = [0,1,1,0.05,1,0.05,1,1,1,0.9]
    if (primed_IL15):
        y0[0] = 1
    #Initialize Parameter values
    parameter               = dict()
    parameter["a"]          = np.empty(shape=11, dtype = used_datatype)
    parameter["d"]          = np.empty(shape=10, dtype = used_datatype)
    parameter["k"]          = np.empty(shape=15, dtype = used_datatype)
    parameter["alpha"]      = np.empty(shape=2,  dtype = used_datatype)

    #Predetermined param    eters
    parameter["a"][0]       = 0                 # Steady state cond.
    parameter["a"][1]       = 0.848             # Prefit + sens. anal
    parameter["a"][2]       = 0.037             # Prefit + sens. anal
    parameter["a"][6]       = 0                 # Biol. assumption
    parameter["a"][7]       = 0                 # Biol. assumption
    parameter["a"][8]       = 0                 # Biol. assumption
    parameter["a"][10]      = 4.17              # (10,19)
    parameter["p_6"]        = 0.99              # (19)                  (in DGL for inhibitors modeling)
    if (O2):
        parameter["k_O2"]   = 0.96              # (19)
    else:
        parameter["k_O2"]   = 0.06              # Supplementary Material: S1
    parameter["k"][0]       = 2e-5              # Prefit + sens. anal
    parameter["k"][1]       = 0.307             # Prefit + sens. anal
    parameter["k"][4]       = 75.895            # Prefit + sens. anal
    parameter["k"][9]       = 421.353           # Prefit + sens. anal, cf. (19)
    parameter["k"][10]      = 0.211             # Prefit + sens. anal
    parameter["k"][11]      = 0.061             # Prefit + sens. anal
    parameter["k"][14]      = 0.088             # Prefit + sens. anal
    parameter["k_s"]        = 9e-4              # Prefit + sens. anal
    parameter["n_2"]        = 2                 # Biol. assumption, cf. (23)
    parameter["xi_28"]      = 38.44             # Prefit + sens. anal
    parameter["delta"]      = 200               # (10,19)
    parameter["xi_4"]       = 15.018            # Prefit + sens. anal, cf. (19)
    parameter["xi_44"]      = 128.022           # Prefit + sens. anal, cf. (19)

    parameter["D"]          = D
    parameter["S_3"]        = S_3
    parameter["R"]          = R
    parameter["NFkBinhi"]   = NFkBinhi


    def model_with_fixed_parameters(t,y, a_5, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_9, d_10, p_3, p_4, k_3, k_4, k_6, k_7, k_8, k_9, k_13, k_14, k_alpha, xi_10, phi, alpha_1, alpha_2):
        parameter["d"][7]       = k_8       # Steady state cond.
        parameter["a"][4]       = a_5
        parameter["d"][0]       = d_1
        parameter["d"][1]       = d_2
        parameter["d"][2]       = d_3
        parameter["d"][3]       = d_4
        parameter["d"][4]       = d_5
        parameter["d"][5]       = d_6
        parameter["d"][6]       = d_7
        parameter["d"][8]       = d_9
        parameter["d"][9]       = d_10
        parameter["p_3"]        = p_3
        parameter["p_4"]        = p_4
        parameter["k"][2]       = k_3
        parameter["k"][3]       = k_4
        parameter["k"][5]       = k_6
        parameter["k"][6]       = k_7
        parameter["k"][7]       = k_8
        parameter["k"][8]       = k_9
        parameter["k"][12]      = k_13
        parameter["k"][13]      = k_14
        parameter["k_alpha"]    = k_alpha
        parameter["xi_10"]      = xi_10
        parameter["phi"]        = phi
        parameter["alpha"][0]   = alpha_1
        parameter["alpha"][1]   = alpha_2

        return network_model(y,**parameter)

    return model_with_fixed_parameters, y0


def get_max_of_model(erg):
#    max_t = 0
#    #max_y = np.linalg.norm(erg.y[:,0])
#    max_y = erg.y[5,0]
#    for i in range(len(erg.t)):
#        if (erg.y[5,i] > max_y):
#            max_y = erg.y[5,i]
#            max_t = erg.t[i]

    max_y = np.max(erg.y[5,:])
    max_t = erg.t[np.where(erg.y[5,:] == max_y)[0][0]]
    return max_t, max_y



def all_parameterized_model(primed_IL15 = False, O2 = True, D = 0, S_3 = 0, R = 0, NFkBinhi = 0):
    #Initialize Initial condition
    #y0 = [0,1,1,0,1,0,1,1,1,0]
    y0 = [0,1,1,0.05,1,0.05,1,1,1,0.9]
    if (primed_IL15):
        y0[0] = 1
    #Initialize Parameter values
    parameter               = dict()
    parameter["a"]          = np.empty(shape=11, dtype = used_datatype)
    parameter["d"]          = np.empty(shape=10, dtype = used_datatype)
    parameter["k"]          = np.empty(shape=15, dtype = used_datatype)
    parameter["alpha"]      = np.empty(shape=2,  dtype = used_datatype)

    parameter["p_6"]        = 0.99              # (19)                  (in DGL for inhibitors modeling)
    if (O2):
        parameter["k_O2"]   = 0.96              # (19)
    else:
        parameter["k_O2"]   = 0.06              # Supplementary Material: S1
    parameter["D"]          = D
    parameter["S_3"]        = S_3
    parameter["R"]          = R
    parameter["NFkBinhi"]   = NFkBinhi


    def model_with_fixed_parameters(t,y, a_5, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_9, d_10, p_3, p_4, k_3, k_4, k_6, k_7, k_8, k_9, k_13, k_14, k_alpha, xi_10, phi, alpha_1, alpha_2, a_1 = 0, a_2 = 0.848, a_3=0.037, a_7=0, a_8=0, a_9=0, a_11=4.17, d_8=-1, k_1= 2e-5, k_2=0.307, k_5=75.895, k_10=421.353, k_11=0.211, k_12=0.061, k_15=0.088, k_S=9e-4, n_2=2, xi_28=38.44, delta=200, xi_4=15.018, xi_44=128.022):
        if (d_8 == -1):
            d_8                 = k_8
        parameter["a"]          = [a_1, a_2, a_3, 0, a_5, 0, a_7, a_8, a_9, 0, a_11]
        parameter["d"]          = [d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9, d_10]
        #parameter["p_3"]        = p_3
        parameter["p_3"]        = 1
        parameter["p_4"]        = p_4
        parameter["k"]          = [k_1, k_2, k_3, k_4, k_5, k_6, k_6, k_7, k_8, k_9, k_10, k_11, k_12, k_13, k_14, k_15]
        parameter["k_alpha"]    = k_alpha
        parameter["xi_10"]      = xi_10
        parameter["xi_4"]       = xi_4
        parameter["xi_28"]      = xi_28
        parameter["xi_44"]      = xi_44
        parameter["delta"]      = delta
        parameter["k_s"]        = k_S
        parameter["n_2"]        = n_2
        parameter["phi"]        = phi
        parameter["alpha"]      = [alpha_1, alpha_2]

        return network_model(y,**parameter)

    return model_with_fixed_parameters, y0


def get_parameter_intervalls(percentage = 0.25, table_1 = False):
    problem                     = dict()
    estimated_parameter         = [0.211,0.062,0.848,0.919,0.623,0.196,0.301,0.914,0.934,0.935,1,0.863,0.181,76.196,25.001,2.903,0.577,0.753,12.152,16.528,1.034,8.127,0.829,1.163,0.386]
    problem["names"] = ["a_5", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_9", "d_10", "p_3", "p_4", "k_3", "k_4", "k_6", "k_7", "k_8", "k_9", "k_13", "k_14", "k_alpha", "xi_10", "phi", "alpha_1", "alpha_2"]

    if (table_1):
        estimated_parameter_table_1 = [0, 0.848, 0.037, 0, 0, 0, 4.17, 0.577, 2e-5, 0.307, 75.895, 421.353, 0.211, 0.061, 0.088, 9e-4, 2, 38.44, 200, 15.018, 128.022]
        names_table_1               = ["a_1", "a_2", "a_3", "a_7", "a_8", "a_9", "a_11", "d_8", "k_1", "k_2", "k_5", "k_10", "k_11", "k_12", "k_15", "k_S", "n_2", "xi_28", "delta", "xi_4", "xi_44"]
        problem["names"] += names_table_1
        estimated_parameter += estimated_parameter_table_1

    problem["num_vars"] = len(estimated_parameter)
    problem["bounds"]   = []
    for est in estimated_parameter:
        if est == 0:
            problem["bounds"].append([0,0.1])
        else:
            problem["bounds"].append([max(0,(1-percentage)*est), (1+percentage)*est])

    return problem


def create_barplot(filename, title, names, Si):
    xpos = 2*np.arange(len(names))
    plt.bar(xpos, Si["mu_star"], width = 1)
    plt.xticks(xpos, names ,rotation = 45)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def create_barplot_sobol(filename, title, names, Si):
    xpos = np.arange(len(names))
    plt.bar(xpos-0.2, Si["S1"], width = 0.4, label="First order Sensitivity")
    plt.bar(xpos+0.2, Si["ST"], width = 0.4, label="Total Sensitivity")
    plt.xticks(xpos, names ,rotation = 45)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()
