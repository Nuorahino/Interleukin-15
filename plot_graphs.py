import matplotlib.pyplot as plt



def plot_graphs(erg, aLegend):
    Graph_names = ["mTor", "NF-kb", "STAT3", "HIF-1alpha mRNA", "HIF-1alpha", "HIF-1", "HIF-1alpha-aOH", "HIF-1beta"]
    col = [2, 6, 7, 8, 3, 5, 9, 4]
    for j in range(len(Graph_names)):
        plt.xlabel("Time [h]")
        for i in range(len(aLegend)):
            Y = erg[i].y
            T = erg[i].t
            plt.plot(T, Y[col[j],:], label = aLegend[i])
        plt.title(Graph_names[j])
        plt.legend()
        plt.show()

