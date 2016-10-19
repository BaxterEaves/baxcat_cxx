""" An example for animals with attributes data takes from
http://attributes.kyb.tuebingen.mpg.de/

Each row is an animal and each column is an attribute. Each cell is a binary
value denoting whether the attribute is present (1) or absent (0).

Usage:
    $ python animals.py
"""

import os

from baxcat.engine import Engine
from math import exp

import pandas as pd
import matplotlib.pyplot as plt


assert __name__ == "__main__"

exdir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(exdir, 'animals.csv'), index_col=0)


# Let's create out engine. Well just pass in the data and let baxcat decide
# how to model each column.
engine = Engine(df, n_models=32)

# We can see how baxcat decided to model each column by checking `col_info`
col_info = engine.col_info()
print(col_info)

# To do inference, we intialize some cross-categorization states with
# `init_models` then `run` the inference. We intitialize many models to hedge
# the inferences we make. Every model is a draw from the posterior. We want to
# make inference about the data given the posterior distribution of states, so
# we take several models.
print('Initializing 32 models...')
engine.init_models()
print('Running models for 200 iterations...')
engine.run(200, checkpoint=5)

# To check whether inference has converges, we plot the log score for each
# model as a function of time and make sure they all have leveled out.
engine.convergence_plot()
plt.show()

# We can view which columns are dependent on which other columns by plotting
# a n_cols by n_cols matrix where each cell is the dependence probability
# between two columns. Note that the dependence probability is simply the
# probability that a dependence exists, not the strength of the dependence.
engine.heatmap('dependence_probability', plot_kwargs={'figsize': (10, 10,)})
plt.show()

engine.heatmap('row_similarity', plot_kwargs={'figsize': (10, 10,)})
plt.show()

# The paint job is an important part of what makes a pine wood derby car fast,
# but does it matter for animals? We'll use the linfoot information to
# determine how predictive variables are of whether an animal is fast. Linfoot
# if basically the information-theoretic counterpart to correlation.
linfoot_lean = engine.mutual_information('fast', 'lean', linfoot=False)
linfoot_stripes = engine.mutual_information('fast', 'stripes', linfoot=False)

print('Linfoot(fast, lean) = %f' % (linfoot_lean,))
print('Linfoot(fast, stripes) = %f' % (linfoot_stripes,))

# We can also figure out which animals are more similar. Is a wolf more
# similar to a dalmatian or a rat.
sim_wolves = engine.row_similarity('chihuahua', 'wolf')
sim_rats = engine.row_similarity('chihuahua', 'rat')

print('Similarity between Chihuahuas and wolves is %f' % (sim_wolves,))
print('Similarity between Chihuahuas and rats is %f' % (sim_rats,))


# Which animals are outliers with respect to their being fast. We can find out
# by calculating the surprisal (self infotmation).
s = engine.surprisal('fast')
s.sort(['surprisal'], ascending=False, inplace=True)
print(s.head(10))

# Lets say we're out in the woods and we see a lean, spotted animal with a
# tail. What is the probability that it is fierce and fast?
# Note that for continuous variables, Engine.probability returns the log PDF
# of an event given observations.
logp = engine.probability([1, 1], ['fierce', 'fast'],
                          given=[('lean', 1,), ('spots', 1,), ('tail', 1,)])
print('p(fierce, fast | lean, spots, tail) = %s' % (exp(logp),))
