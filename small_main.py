import sys
import Sensitivity_analysis
import small_problem


def run_analysis(condition, use_all_parameters=False, percentage=0.25):
    y0, bounds, args = small_problem.get_parameters_log(*condition)
    objective_function = [small_problem.get_max, small_problem.get_time_of_max]
    print(args)
    print(bounds)
    res = Sensitivity_analysis.run_sensitivity_analysis(small_problem.network_model, y0, bounds, args, objective_function, t=[0, 100], n=4960, method="Morris", use_log_scale=True)
    directory = "plots/small/"+"morris/"
    filename = "untreated"
    if condition[0]:
        filename += "+IL-15"
    if condition[1]:
        filename += "+DMOG"
    if condition[2]:
        filename += "+S31-201"
    if condition[3]:
        filename += "+Rapa"
    if condition[4]:
        filename += "+NF-kBi"

    filename += "_interval_" + str(percentage)

    Sensitivity_analysis.create_barplot_morris(
        directory + "HIF/" + filename + ".png",
        "Sobol: " + filename,
        res[0],
        bounds.keys()
    )
    Sensitivity_analysis.create_barplot_morris(
        directory + "time/" + filename + ".png",
        "Sobol: " + filename,
        res[1],
        bounds.keys()
    )
    with open(directory + "data/" + filename + ".txt", "w") as f:
        f.write("Result for Konzentration\n")
        f.write(str(res[0]))
        f.write("\nResult for Time\n")
        f.write(str(res[1]))


if __name__ == "__main__":
    run_analysis([True, 1, 0, 0, 0], True, 0.25)
