Installation
============

Install using pip
-----------------

To install IDPET using pip, follow these steps:

Step 1: Install NeoForceScheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing IDPET, you need to install `NeoForceScheme <https://github.com/visml/neo_force_scheme/tree/0.0.3>`_.

.. code-block:: shell

    pip install git+https://github.com/visml/neo_force_scheme@0.0.1

Step 2: Install MDTraj
~~~~~~~~~~~~~~~~~~~~~~

IDPET is built on top of `MDTraj <https://mdtraj.org/>`_. As MDTraj can have compatibility issues on Windows, we recommend installing it using conda. This step can be omitted when using Linux.

.. code-block:: shell

    conda install -c conda-forge mdtraj

Step 3: Install IDPET
~~~~~~~~~~~~~~~~~~~~~

Finally, you can install IDPET using pip.

.. code-block:: shell

    pip install dpet

Install using conda
-------------------

Step 1: Install NeoForceScheme
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing IDPET, you need to install NeoForceScheme.

.. code-block:: shell

    pip install git+https://github.com/visml/neo_force_scheme@0.0.1

Step 2: Install IDPET
~~~~~~~~~~~~~~~~~~~~~

Install IDPET from the specified conda channel.

.. code-block:: shell

    conda install -c ivanovicnikola dpet