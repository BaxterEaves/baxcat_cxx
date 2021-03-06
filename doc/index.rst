.. baxcat_cxx documentation master file, created by
   sphinx-quickstart on Thu Aug 25 16:59:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directiv
   
baxcat_cxx
==========

Easy analysis of tabular data without all the data torture.

.. note:: This software is in development. The interface and modeling assumptions are subject to change.

.. warning:: The Engine metadata has changed in verion 0.3.0. Models saved with prior versions will not load with the current version (see :doc:`changelog`).

The point of baxcat is to ease the process of conducting analyses on tabular data. Once you have your data in the proper format, simply load it, run the Engine and start asking questions::

    from baxcat.engine import Engine
    import pandas as pd

    engine = Engine('mydata.csv', n_models=16)
    engine.init_models()
    engine.run(200)

    print(engine.dependence_probability('ice_cream_sales', 'murders'))

We've also made scaling easy. Just supply baxcat with a parallel map-like function. For example, using baxcat on cluster is simple with IPython Parallel::

    import ipyparallel as ipp
    
    c = ipp.Client()
    v = c[:]

    engine = Engine('mydata.csv', mapper=v.map)


Table of contents
=================

.. toctree::
    :maxdepth: 1

    getting_started
    tutorial
    model
    engine
    metric
    changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`changelog`

