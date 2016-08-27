Modeling assumptions: how :mod:`baxcat_cxx` works
=================================================

Cross-categorization (Crosscat) fits a joint distribution to tables of data. Each column in the table is modeled as an infinite mixture model. Columns are grouped into `views`. Within each column, rows are grouped into `categories`. Crosscat does this via a nested `Dirichlet Process <https://en.wikipedia.org/wiki/Dirichlet_process>`_. A Diriclet process a prior on how n data are partitioned into bewteen 1 and n categories. baxcat uses the `Chinese Restaurant Process (CRP) <https://en.wikipedia.org/wiki/Chinese_restaurant_process>`_ as a prior on view and cateogory assignments. 

The rows of a column in a category correspond to one component of a mixture model.

The cross-categorization generative processs is as follows

* draw CRP parameter, :math:`\alpha_z \sim InvGamma(1, 1)`
* draw a partition of cololumns to views, :math:`z \sim CRP(\alpha_z)`
* For each view :math:`v \in z`
    * draw a CRP parameter :math:`\alpha_v \sim InvGamma(1, 1)`
    * draw a partition of rows to categories :math:`c_v \sim CRP(\alpha_v)`
    * For each column in the view, :math:`\{j : z_j = v\}`
        * draw prior parameters, :math:`\phi` from the hyper-prior (see FIXME)
        * for each category in the row partition :math:`k \in c_v`:
            * draw component parameters, :math:`\theta_{k}^{(v)} \sim G(\phi_j)`, from the prior
            * for each row in the category, :math:`i : c_i^{(v)} = k`:
                * :math:`x_{i,j} \sim F\left(\theta_k^{(v)}\right)`


Currently implemented component models
--------------------------------------

Continuous
^^^^^^^^^^

Continuous columns are modeled as normal distributions with a Normal prior on the mean, :math:`mu`, and Gamma prior on the precision :math:`\rho` (see `this PDF <http://www.stats.ox.ac.uk/~teh/research/notes/GaussianInverseGamma.pdf>`_):

.. math::
    x \sim \mathcal{N}(\mu,\rho) \\
    \mu \sim \mathcal{N}(m, r\rho) \\
    \rho \sim \text{Gamma}(\nu/2, s/2)

Each hyper parameter has a vague hyper prior based on the data:

.. math::
    m \sim \mathcal{N}(\bar{x}, \text{std}(x))\\
    r \sim \text{Gamma}(1, \text{std}(x))\\
    s \sim \text{Gamma}(1, \text{std}(x))\\
    \nu \sim \text{Gamma}(2, .5)


Categorical
^^^^^^^^^^^

Categorical columns are implemented at categorical/discrete dsitributions with a symmetric Dirichlet prior on the dirichler prameter, :math:`\alpha`:

.. math::
    x \sim \text{Categorical}_k(\hat{\theta}) \\
    \hat{\theta} \sim \text{Dirichlet}_k(\alpha)

:math:`\alpha` has a vague prior

.. math::
    \alpha \sim \text{Gamma}(1, 1/k)
