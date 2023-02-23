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
for serverless Federated Learning experiments.
FLoX makes it easy and fast to set up your FL infrastructure, allowing you to start
running FL workflows in under 5 minutes.
Follow our Quickstart tutorial and examples to get started! You can read the associated paper
`here <https://ieeexplore.ieee.org/document/9973578>`_.

Quickstart
=====

To get started with FLoX, check out this `Google Colab tutorial`_.

|colab_quickstart|

.. |colab_quickstart| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/10en48ipDi9qsDQdgOCvQiYQ58Rqqk8mB?usp=sharing
   :alt: Documentation Status

.. _Google Colab tutorial: https://colab.research.google.com/drive/10en48ipDi9qsDQdgOCvQiYQ58Rqqk8mB#scrollTo=sL0dIUCTEURR

Installation
============

*Controller* is a device from which Federated Learning is facilitated, such as your laptop.
*Endpoints* are devices that participate in Federated Learning, such as Raspberry Pis.

Use the package manager `pip <https://pip.pypa.io/en/stable/>`_ to install flox
on the *Controller*:

.. code-block:: console

   (.venv) $ pip install pyflox

To be able to include your *endpoints* into the Federated Learning process,
you need to install `funcx-endpoint <https://funcx.readthedocs.io/en/latest/endpoints.html>`_ on the *endpoints*:

.. code-block:: console

   (.endpoint_venv) $ python3 -m pipx install funcx_endpoint

Finally, install compatible versions of `PyTorch <https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html) & [Torchvision](https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html#:~:text=rm%20%2Drf%20~/pytorch-,TorchVision.,-Install%20torchvision%20on>`_
or `Tensorflow <https://qengineering.eu/install-tensorflow-2.1.0-on-raspberry-pi-4.html>`_ on both the Controller and endpoints.
As of now, FLoX supports Tensorflow and PyTorch, although you can add support for your frameworks by creating a new
ModelTrainer. See ``flox.logic`` and examples of ModelTrainers in ``flox.model_trainers``.

*Note*: ``funcX-endpoint`` is only supported on Linux.
FLoX Controller functionality is supported on MacOS, Linux and Windows.

Contributing to FLoX
====================

Contributions are welcome. Please see `CONTRIBUTING.md <https://github.com/globus-labs/FLoX/blob/main/CONTRIBUTING.md>`_.

Documentation
=============
Complete documentation for FLoX is available `here <https://pyflox.readthedocs.io/en/latest/>`_.

Citation
========
If you publish work that uses FLoX, please cite FLoX as follows:
```bibtex
@INPROCEEDINGS{9973578,
  author={Kotsehub, Nikita and Baughman, Matt and Chard, Ryan and Hudson, Nathaniel and Patros, Panos and Rana, Omer and Foster, Ian and Chard, Kyle},
  booktitle={2022 IEEE 18th International Conference on e-Science (e-Science)},
  title={FLoX: Federated Learning with FaaS at the Edge},
  year={2022},
  volume={},
  number={},
  pages={11-20},
  doi={10.1109/eScience55777.2022.00016}}
```

