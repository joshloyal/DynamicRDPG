.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Coveralls|_ |CircleCI|_ |License|_

.. |Travis| image:: https://travis-ci.org/joshloyal/dynrdpg.svg?branch=master
.. _Travis: https://travis-ci.org/joshloyal/cookiecutter.project_slug}}

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/54j060q1ukol1wnu/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/joshloyal/dynrdpg/history

.. |Coveralls| image:: https://coveralls.io/repos/github/joshloyal/dynrdpg/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/joshloyal/dynrdpg?branch=master

.. |CircleCI| image:: https://circleci.com/gh/joshloyal/dynrdpgtree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/joshloyal/dynrdpg/tree/master

.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
.. _License: https://opensource.org/licenses/MIT


.. _scikit-learn: https://github.com/scikit-learn/scikit-learn

Dynamic RDPG
=============================
Dynamic RDPG It is compatible with scikit-learn_.


Documentation / Website: https://joshloyal.github.io/dynrdpg


Example
-------
.. code-block:: python

    print("Hello, world!")

Installation
------------

Dependencies
------------
Dynamic RDPG requires:

- Python (>= 2.7 or >= 3.4)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Scikit-learn (>=0.17)

Additionally, to run examples, you need matplotlib(>=2.0.0).

Installation
------------
You need a working installation of numpy and scipy to install Dynamic RDPG. If you have a working installation of numpy and scipy, the easiest way to install dynrdpg is using ``pip``::

    pip install -U dynrdpg

If you prefer, you can clone the repository and run the setup.py file. Use the following commands to get the copy from GitHub and install all the dependencies::

    git clone https://github.com/joshloyal/dynrdpg.git
    cd dynrdpg
    pip install .

Or install using pip and GitHub::

    pip install -U git+https://github.com/joshloyal/dynrdpg.git


Testing
-------
After installation, you can use pytest to run the test suite via setup.py::

    python setup.py test

References:
-----------
