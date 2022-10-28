====
FLoX - **F**\ ederated **L**\ earning on func\ **X**
====
|docs| |licence|

.. |docs| image:: https://readthedocs.org/projects/pyflox/badge/?version=latest
   :target: https://pyflox.readthedocs.io/en/latest/index.html
   :alt: Documentation Status
.. |licence| image:: https://img.shields.io/badge/license-MIT-blue
   :target: https://choosealicense.com/licenses/mit/
   :alt: MIT License

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

Finally, install Tensorflow on both the Controller and endpoints.
As of now, FLoX only supports Tensorflow, although support for PyTorch will be added soon too.

.. code-block:: console

   (.venv) $ pip install tensorflow==1.14.0 

*Note*: ``funcX-endpoint`` is only supported on Linux. 
FLoX Controller functionality is supported on MacOS, Linux and Windows.

Usage
=====

For a full example, see this `Google Colab tutorial`_.

.. _Google Colab tutorial: https://colab.research.google.com/drive/19X1N8E5adUrmeE10Srs1hSQqCCecv23m?usp=sharing

.. code-block:: python

   from flox.flox import federated_learning

   # performs 5 rounds of Federated Learning train global_model on given endpoint_ids
   # uses 10 epochs and 100 samples from fashion_mnist dataset for training
   federated_learning(global_model=global_model, 
                     endpoint_ids=endpoint_ids,
                     loops=5,
                     epochs=10,
                     keras_dataset="fashion_mnist", 
                     num_samples=100, 
                     input_shape=(32, 28, 28, 1))


Contributing
============

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
=======

`MIT <https://choosealicense.com/licenses/mit/>`_
