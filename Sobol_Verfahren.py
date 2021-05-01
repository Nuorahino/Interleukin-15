from tqdm import tqdm
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.util import read_param_file
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Definiere das Problem und lege für jeden Parameter den Definitionsbereich fest:
problem = read_param_file("problem_data",",")

Y = np.linspace(1, 2, 163840)


#Definiere das System aus 10 Differenzialgleichungen ersten Grades
def model(f, t, a_2, a_3, a_5, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_9, d_10, k_1, k_2, k_3, k_4, k_5, k_6, k_7,k_8, k_9, k_10, k_11, k_12, k_13, k_14, k_15, k_alpha,K_S, alpha_1,alpha_2, xi_28,xi_4, xi_44, xi_10, phi,p_3, p_4):
  n = [0 - d_1 * f[0],
    a_2 + k_1 * f[0] + K_S*((f[7] ** 2)/((xi_28**2)+(f[7] ** 2))) - d_2 * f[1],
    (a_3 + k_2 * f[1]) * (alpha_1 / (alpha_2 +f[5])) - d_3 * f[2],
    k_alpha * f[8] - d_4 * f[3] - k_4 * f[3] * f[4] + k_5 * f[5] - k_13 * 0.96 * (200 * f[5] + 4.17) * (f[3]/ (xi_44 + f[3])) - k_10 * 0.96 * phi * (f[3] / (xi_4 + f[3])) + k_11 * f[9],
    a_5 - k_4 * f[3] * f[4] + k_5 * f[5] - d_5 * f[4],
    k_4 * f[3] * f[4] - k_5 * f[5] - d_6 * f[5],
    0 + k_7 * f[0] + k_14 * f[5] + k_15 * f[2] - d_7 * f[6],
    (0 + k_8 * f[2] + k_6*(1-p_4) *f[0])*(1-p_3) - k_8 * f[7],
    0 + k_9 * f[6] + k_3 * f[7] - d_9 *f[8],
    k_10 * 0.96 * phi * (f[3] / (xi_4 + f[3])) - k_12 * 0.96 * (200 * f[5] + 4.17) * (f[9] /(xi_10 + f[9])) - k_11 * f[9] - d_10 * f[9]]
  return n


#Führe das Sampling nach der Methode von Saltelli durch und
#generiere eine -Matrix:
X = saltelli.sample(problem, 4096,calc_second_order=False, seed=1)

print(np.shape(X))

y0 = [1,1,1,0.05,1,0.05,1,1,1,0.9]
t = [0,11]

for i in tqdm(range(163840)):
    erg = odeint(model,y0,t,args=tuple(X[i]))

    Y[i] = np.linalg.norm(erg[1])

Si = sobol.analyze(problem, Y,calc_second_order=False, print_to_console=True, conf_level=0.95)

