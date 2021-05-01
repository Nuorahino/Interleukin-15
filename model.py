import numpy as np

used_datatype = np.float64


def network_model(y, a, d, k, k_s, k_O2, k_alpha, alpha, n_2, xi_4, xi_10, xi_28, xi_44, delta, phi, p_3, p_4, p_6, D, R, S_3):
#    print("a: ", a)
#    print("d: ", d)
#    print("k: ", k)
#    print("k_s:", k_s)
#    print("k_O2:", k_O2)
#    print("k_alpha:", k_alpha)
#    print("alpha:", alpha)
#    print("n_2:", n_2)
#    print("xi_4:", xi_4)
#    print("xi_10:", xi_10)
#    print("xi_28:", xi_28)
#    print("xi_44:", xi_44)
#    print("delta:", delta)
#    print("phi:", phi)
#    print("p_3:", p_3)
#    print("p_4:", p_4)
#    print("p_6:", p_6)
#    print("D:", D)
#    print("R:", R)
#    print("S_3:", S_3)
    dgl     = np.empty(shape=10, dtype = used_datatype)
    dgl[0]  = a[0] - d[0] * y[0]
    dgl[1]  = a[1] + k[0]*y[0] + k_s*(y[7]**n_2)/(xi_28**n_2 + y[7]**n_2) - d[1]*y[1]
    #dgl[2]  = (a[2] + k[1]*y[1]) * alpha[0]/(alpha[1]+y[5]) - d[2]*y[2]
    #dgl[3]  = k_alpha*y[8] - d[3]*y[3] - k[3]*y[3]*y[4] + k[4]*y[5] - k[12]*k_O2*(delta*y[5] + a[10])*y[3]/(xi_44+y[3]) - k[9]*k_O2*phi*y[3]/(xi_4 + y[3]) + k[10]*y[9]
    dgl[4]  = a[4] - k[3]*y[3]*y[4] + k[4]*y[5] - d[4]*y[4]
    dgl[5]  = k[3]*y[3]*y[4] - k[4]*y[5] - d[5]*y[5]
    #dgl[6]  = a[6] + k[6]*y[0] + k[13]*y[5] + k[14]*y[2] - d[6]*y[6]
    #dgl[7]  = a[7] + k[7]*y[2] + k[5]*y[0] - d[7]*y[7]
    dgl[8]  = a[8] + k[8]*y[6] + k[2]*y[7] - d[8]*y[8]
    #dgl[9]  = k[9]*k_O2*phi*y[3]/(xi_4 + y[3]) - k[11]*k_O2*(delta*y[5] + a[10])*y[9]/(xi_10 + y[9]) -k[10]*y[9] - d[9]*y[9]

    # D,S,R are 1 if the respectife inhibitor is present else 0
    # Modeling inhibitors
    dgl[2]  = (a[2] + k[1]*y[1]) * alpha[0]/(alpha[1]+y[5])*(1-R) - d[2]*y[2]
    dgl[3]  = k_alpha*y[8] - d[3]*y[3] - k[12]*k_O2*(delta*y[5] + a[10])*(1-p_6*D)*y[3]/(xi_44+y[3]) - k[3]*y[3]*y[4] + k[4]*y[5]- k[9]*k_O2*phi*(1-p_6*D)*y[3]/(xi_4 + y[3]) + k[10]*y[9]
    dgl[6]  = (a[6] + k[6]*y[0] + k[13]*y[5] + k[14]*y[2])*(1-R) - d[6]*y[6]
    dgl[7]  = (a[7] + k[7]*y[2] + k[5]*(1-p_4*D)*y[0])*(1 - p_3*S_3) - d[7]*y[7]
    dgl[9]  = k[9]*k_O2*(1-p_6*D)*phi*y[3]/(xi_4 + y[3]) - k[11]*k_O2*(1-p_6*D)*(delta*y[5] + a[10])*y[9]/(xi_10 + y[9]) -k[10]*y[9] - d[9]*y[9]

    return dgl


# Default is that the cells are not treated, and have normal amount of oxygen
def parameterized_model(O2 = 1, D = 0, S_3 = 0, R = 0):
    parameter           = dict()
    parameter["a"]      = np.empty(shape=11, dtype = used_datatype)
    parameter["d"]      = np.empty(shape=10, dtype = used_datatype)
    parameter["k"]      = np.empty(shape=15, dtype = used_datatype)
    parameter["alpha"]  = np.empty(shape=2,  dtype = used_datatype)

    #Predetermined parameters
    parameter["a"][0]   = 0                 # Steady state cond.
    parameter["a"][1]   = 0.848             # Prefit + sens. anal
    parameter["a"][2]   = 0.037             # Prefit + sens. anal
    parameter["a"][6]   = 0                 # Biol. assumption
    parameter["a"][7]   = 0                 # Biol. assumption
    parameter["a"][8]   = 0                 # Biol. assumption
    parameter["a"][10]  = 4.17              # (10,19)
    # Neeeding to adjust based on how the parameter
    #parameter["d_8"]    = k_8               # steady state cond.
    parameter["p_6"]    = 0.99              # (19)                  (in DGL for inhibitors modeling)
    if (O2):
        parameter["k_O2"]   = 0.96              # (19)
    else:
        parameter["k_O2"]   = 0.06              # Supplementary Material: S1
    parameter["k"][0]   = 2e-5              # Prefit + sens. anal
    parameter["k"][1]   = 0.307             # Prefit + sens. anal
    parameter["k"][4]   = 75.895            # Prefit + sens. anal
    parameter["k"][9]   = 421.353           # Prefit + sens. anal, cf. (19)
    parameter["k"][10]  = 0.211             # Prefit + sens. anal
    parameter["k"][11]  = 0.061             # Prefit + sens. anal
    parameter["k"][14]  = 0.088             # Prefit + sens. anal
    parameter["k_s"]    = 9e-4              # Prefit + sens. anal
    parameter["n_2"]    = 2                 # Biol. assumption, cf. (23)
    parameter["xi_28"]  = 38.44             # Prefit + sens. anal
    parameter["delta"]  = 200               # (10,19)
    parameter["xi_4"]   = 15.018            # Prefit + sens. anal, cf. (19)
    parameter["xi_44"]  = 128.022           # Prefit + sens. anal, cf. (19)

    parameter["D"]      = D
    parameter["S_3"]    = S_3
    parameter["R"]      = R

    def model_with_fixed_parameters(y, t, a_5, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_9, d_10, p_3, p_4, k_3, k_4, k_6, k_7, k_8, k_9, k_13, k_14, k_alpha, xi_10, phi, alpha_1, alpha_2):
        parameter["d"][7]       = k_8       # Steady state cond.
        #combine u with parameter
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
#    def model_with_fixed_parameters(y, t, u):
#        parameter["d"][7]       = k_8       # Steady state cond.
#        #combine u with parameter
#        parameter["a"][5]   = u[0]
#        parameter["d"]      = u[1:8]
#        parameter["d"][8]   = u[8]
#        parameter["d"][9]   = u[9]
#        parameter["p_3"]    = u[10]
#        parameter["p_4"]    = u[11]
#        parameter["k"][2]   = u[12]
#        parameter["k"][3]   = u[13]
#        parameter["k"][5]   = u[14]
#        parameter["k"][6]   = u[15]
#        parameter["k"][7]   = u[16]
#        parameter["k"][8]   = u[17]
#        parameter["k"][12]  = u[18]
#        parameter["k"][13]  = u[19]
#        parameter["k_alpha"]= u[20]
#        parameter["xi_10"]  = u[21]
#        parameter["phi"]    = u[22]
#        parameter["alpha"]  = u[23:25]
#
#        return network_model(**parameter)

    return model_with_fixed_parameters
