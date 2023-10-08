import matplotlib.pyplot as plt
import numpy as np
import pickle

class MultiLinePlotter:
    def __init__(self):
        # 设置x的范围
        self.x = list(range(1, 201))

    def plot_graphs(self, y_values, save_as="pdf"):
        if len(y_values) != 12:
            print("Error: Please provide exactly 12 lists for y-values.")
            return

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2行2列的布局

        titles = ['T1', 'T2', 'T3', 'T4']


        # 第一个子图
        axs[0, 0].plot(self.x, y_values[2], color='black')
        axs[0, 0].set_title(titles[0])

        # 其他子图
        index_start = 3  # Starting from y4
        for i in range(1, 4):
            ax = axs.flatten()[i]
            for j in range(3):
                ax.plot(self.x, y_values[index_start], label = f'{round(0.1 ** (j + 1), 3)}')
                index_start += 1
            ax.legend()
            ax.set_title(titles[i])

        plt.tight_layout()

        # Save the plot based on desired format
        if save_as == "pdf":
            plt.savefig("multi_line_plot.pdf", format='pdf')
        elif save_as == "svg":
            plt.savefig("multi_line_plot.svg", format='svg')
        else:
            print("Unsupported file format. Saving as PNG.")
            plt.savefig("multi_line_plot.png")

        plt.show()


