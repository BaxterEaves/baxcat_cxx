import os

from baxcat.engine import Engine
import pandas as pd
import matplotlib.pyplot as plt


assert __name__ == "__main__"

exdir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(exdir, 'zoo.csv'), index_col=0)


# Let's create out engine. Well just pass in the data and let baxcat decide
# how to model each column.
engine = Engine(df)

# We can see how baxcat decided to model each column by checking `col_info`
col_info = engine.col_info()
print(col_info)

# To do inference, we intialize some cross-categorization states with
# `init_models` then `run` the inference. We intitialize many models to hedge
# the inferences we make. Every model is a draw from the posterior. We want to
# make inference about the data given the posterior distribution of states, so
# we take several models.
print('Initializing 16 models...')
engine.init_models(16)
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
sim_dalmatian = engine.row_similarity('wolf', 'dalmatian')
sim_rat = engine.row_similarity('wolf', 'rat')

print('Similarity between wolves and dalmatians is %f' % (sim_dalmatian,))
print('Similarity between wolves and rats is %f' % (sim_rat,))
