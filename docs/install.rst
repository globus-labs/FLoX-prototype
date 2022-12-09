.. _installation:

Installation
------------
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