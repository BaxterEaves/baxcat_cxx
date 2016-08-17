""" An example using the MNIST handwritter digits dataset

Original, badly-packaged data from http://yann.lecun.com/exdb/mnist/
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from baxcat.engine import Engine


def row_to_img(df, row_idx):
    pixels = df.iloc[row_idx, 1:].values.reshape((28, 28,))
    return pixels


assert __name__ == "__main__"

exdir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(exdir, 'mnist.csv.gz'), compression='gzip')
df = df.sample(500)

engine = Engine(df)
engine.init_models(8)
engine.run(200, checkpoint=4, quiet=False)

engine.convergence_plot()
plt.show()

engine.heatmap('row_similarity')
plt.show()

engine.heatmap('dependence_probability')
plt.show()
