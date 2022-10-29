====
FLoX - **F**\ ederated **L**\ earning on func\ **X**
====
|docs| |licence|

.. |docs| image:: https://readthedocs.org/projects/pyflox/badge/?version=latest
   :target: https://pyflox.readthedocs.io/en/latest/index.html
   :alt: Documentation Status
.. |licence| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/globus-labs/FLoX/blob/main/LICENSE.TXT 
   :alt: Apache Licence V2.0

FLoX (**F**\ ederated **L**\ earning on func\ **X**) is a Python library
for serverless Federated Learning experiments. Federated Learning (FL) experiments. 
FLoX makes it easy and fast to set up your FL infrastructure, allowing you to start
running experiments in under 5 minutes. 
Start with :ref:`Installation`, and then follow our Quickstart tutorial and examples to get started!

Installation
============

*Controller* is a device from which Federated Learning is facilitated, such as your laptop. 
*Endpoints* are devices that participate in Federated Learning, such as Raspberry Pis.

Use the package manager `pip <https://pip.pypa.io/en/stable/>`_ to install flox
on the *Controller*:

.. code-block:: console

   (.venv) $ pip install pyflox

To be able to include your *endpoints* into the Federated Learning process,
you need to install `funcx-endpoint <https://funcx.readthedocs.io/en/latest/endpoints.html>`_:

.. code-block:: console

   (.endpoint_venv) $ python3 -m pipx install funcx_endpoint

Finally, install compatible versions of Tensorflow on both the Controller and endpoints.
As of now, FLoX only supports Tensorflow, although support for PyTorch will be added soon.

.. code-block:: console

   (.venv) $ pip install tensorflow==1.14.0 

*Note*: ``funcX-endpoint`` is only supported on Linux. 
FLoX Controller functionality is supported on MacOS, Linux and Windows.

Quickstart
=====

To get started with FLoX, check out this `Google Colab tutorial`_.

|colab_quickstart|

.. |colab_quickstart| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/10en48ipDi9qsDQdgOCvQiYQ58Rqqk8mB?usp=sharing
   :alt: Documentation Status

.. _Google Colab tutorial: https://colab.research.google.com/drive/10en48ipDi9qsDQdgOCvQiYQ58Rqqk8mB#scrollTo=sL0dIUCTEURR

Contributing to FLoX
====================

Contributions are welcome. Please see `CONTRIBUTING.md <https://github.com/globus-labs/FLoX/blob/main/CONTRIBUTING.md>`_.

Documentation
=============
Complete documentation for FLoX is available `here <https://pyflox.readthedocs.io/en/latest/>`_.