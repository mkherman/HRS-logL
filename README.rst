HRS-logL
========

**Log-likelihood calculation for high-resolution planetary spectra cross-correlated with atmospheric models.**



logL.py
--------------

This code computes the cross-correlation function and full log-likelihood map for a set of planetary emission spectra paired with atmospheric emission models. It was specifically written for CFHT ESPaDOnS observations of WASP-33b, cross-correlated with Fe I emission models.

To run this code:

.. code-block:: bash

    python logL.py -nights {nights} -d {datapath} -m {modelpath} -o {outpath}

where: 

* ``{nights}`` is a list of MJDs identifying the set of observations
* ``{datapath}`` is the path to the raw data
* ``{modelpath}`` is the path to the models
* ``{outpath}`` is the desired path for the output

You can also specify:

* ``-ext {extension}`` for the output file name extension (default is .fits)

The output is a .fits file containing the full log-likelihood map for that set of observations. Fair warning, the file can be pretty large (> 1 GB) depending on your parameter ranges and stepsizes.



cond+marg_distributions.py
--------------

This code computes the conditional and marginalized likelihood distributions from the log-likelihood output of logL.py. It also provides the constrained value, uncertainty, and significance for each parameter.

To run this code I recommend using IPython or Jupyter Notebooks, as it's mostly useful for plotting the various distributions. But to just view the parameter constraints, you can run it via:

.. code-block:: bash

    python cond+marg_distributions.py

