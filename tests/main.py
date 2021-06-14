import sys
import Sobol_Verfahren
import large_problem

def run_analysis(condition, use_all_parameters=False, percentage=0.25):
    variables = ["a_5", "d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_9", "d_10", "rho_3", "rho_4", "k_3", "k_4", "k_6", "k_7", "k_8", "k_9", "k_13", "k_14", "k_alpha", "xi_10", "phi", "alpha_1", "alpha_2"]
    if use_all_parameters:
        variables += ["a_1", "a_2", "a_3", "a_7", "a_8", "a_9", "a_11", "d_8", "k_1", "k_2", "k_5", "k_10", "k_11", "k_12", "k_15", "k_s", "n_2", "xi_28", "delta", "xi_4", "xi_44"]
    print(condition)
    y0, bounds, args = large_problem.all_parameterized_model(variables, percentage, *condition)
    objective_function = [large_problem.get_max, large_problem.get_time_of_max]
    res = Sobol_Verfahren.run_sensitivity_analysis(large_problem.network_model, y0, bounds, args, objective_function, t = [0,100], n=10, method="Sobol")
    print(res)


if __name__ == "__main__":
    simple_conditions = [
        [False, True, 0, 0, 0, 0],
        [True, True, 0, 0, 0, 0],
        [True, True, 0, 1, 0, 0],
        [True, True, 0, 0, 1, 0],
        [True, True, 0, 0, 0, 1],
        [False, False, 1, 0, 0, 0],
        [True, False, 1, 0, 0, 0],
        [True, False, 1, 1, 0, 0],
        [True, False, 1, 0, 1, 0],
        [True, False, 1, 0, 0, 1],
    ]
    double_conditions = [
        [True, True, 0, 1, 1, 0],
        [True, True, 0, 1, 0, 1],
        [True, True, 0, 0, 1, 1],
        [True, False, 1, 1, 1, 0],
        [True, False, 1, 1, 0, 1],
        [True, False, 1, 0, 1, 1],
    ]
    condition = simple_conditions + double_conditions
    if len(sys.argv) == 4:
        run_analysis(
            *(condition[int(sys.argv[1])]), bool(int(sys.argv[2])), float(sys.argv[3])
        )
    else:
        for x in condition:
            run_analysis(condition[0], True, 0.1)
        # run_sobol_analysis(*double_conditions[-1], False, 0.25)
#        for condition in tqdm(double_conditions[2:]):
#            run_sobol_analysis(*condition, False, 0.25)
