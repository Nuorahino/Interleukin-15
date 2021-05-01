from SALib.sample import morris
from SALib.util import read_param_file
from SALib.analyze.morris import analyze as morris_analyze
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import model
from tqdm import tqdm

#Definiere das Problem und lege für jeden Parameter den Definitionsbereich fest
#The order in the file matters
problem = read_param_file("problem_data_new",",")

primed = True

number_of_trajectories = 8
number_of_parameters = len(problem["names"]) + 1

Y = np.linspace(1, 2, number_of_trajectories * number_of_parameters)

estimated_parameter = [0.211,0.062,0.848,0.919,0.623,0.196,0.301,0.914,0.934,0.935,1,0.863,0.181,76.196,25.001,2.903,0.577,0.753,12.152,16.528,1.034,8.127,0.829,1.163,0.386]

#Definiere das System aus 10 Differenzialgleichungen ersten Grades
model = model.parameterized_model(1,0,1,0)

def model2(f, t, a_5, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_9, d_10, p_3, p_4, k_3, k_4, k_6, k_7,k_8, k_9, k_13, k_14, k_alpha, xi_10, phi, alpha_1,alpha_2):
  n = [0 - d_1 * f[0],
    0.848 + 2e-5 * f[0] + 9e-4*((f[7] ** 2)/((38.44**2)+(f[7] ** 2))) - d_2 * f[1],
    (0.037 + 0.307 * f[1]) * (alpha_1 / (alpha_2 +f[5])) - d_3 * f[2],
    k_alpha * f[8] - d_4 * f[3] - k_4 * f[3] * f[4] + 75.895 * f[5] - k_13 * 0.96 * (200 * f[5] + 4.17) * (f[3]/ (128.022 + f[3])) - 421.353 * 0.96 * phi * (f[3] / (15.018 + f[3])) + 0.211 * f[9],
    a_5 - k_4 * f[3] * f[4] + 75.895 * f[5] - d_5 * f[4],
    k_4 * f[3] * f[4] - 75.895 * f[5] - d_6 * f[5],
    0 + k_7 * f[0] + k_14 * f[5] + 0.088 * f[2] - d_7 * f[6],
    (0 + k_8 * f[2] + k_6*(1-p_4) *f[0])*(1-p_3) - k_8 * f[7],
    0 + k_9 * f[6] + k_3 * f[7] - d_9 *f[8],
    421.353 * 0.96 * phi * (f[3] / (15.018 + f[3])) - 0.061 * 0.96 * (200 * f[5] + 4.17) * (f[9] /(xi_10 + f[9])) - 0.211 * f[9] - d_10 * f[9]]
  return n

#Führe das Sampling nach der Methode von Morris durch und
#generiere eine (39*k_M) x 38-Matrix:
X = morris.sample(problem, number_of_trajectories, num_levels=4, seed = 1)
print(np.shape(X))

#Definiere die Anfangswerte für das AWP
if (primed):
    y0 = [1,1,1,0.05,1,0.05,1,1,1,0.9] #Treated cells
else:
    y0 = [0,1,1,1,1,1,1,1,1,1] #Untreated cells
#Lege die Zeitpunkte fest:
t = [0,10]
#t= [0,2]

#print(model(y0,t,*estimated_parameter))

#print(model(y0,t,*estimated_parameter))
#print(model2(y0,t,*estimated_parameter))
#
#print(odeint(model2,y0,t,args=tuple(estimated_parameter)))
#print(odeint(model,y0,t,args=tuple(estimated_parameter)))
print(solve_ivp(model, t, y0, args=tuple(estimated_parameter)))


#for i in tqdm(range(number_of_trajectories * number_of_parameters)):
#    #Löse die das Differenzialgleichungssystem:
#    erg = odeint(model,y0,t,args=tuple(X[i]))
#    #Bestimme die euklidische Norm der Lösung und übergebe sie dem i-ten Eintrag des Vektors Y
#    Y[i] = np.linalg.norm(erg[1], ord=np.inf)
#print('y min: ', np.min(Y))
#print('y max: ', np.max(Y))
#
##Führe nun die Sensitivitätsanalyse durch:
#Si = morris_analyze(problem, X, Y, conf_level=0.95 , print_to_console=True, num_levels=4)
