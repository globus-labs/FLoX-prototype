============
Quickstart PyTorch Tutorial
===========

This example is in development and includes only basic setup and execution instructions.

On your *Controller*, follow these instructions:
1. Create a conda or virtualenv environment
2. Install FLoX with ``pip install -e .`` since the latest version of the library is not out yet.
3. Install PyTorch: ``pip install torch==1.12.0`` and ``pip install torchvision==0.13.0``
4. Create a dist file with ``python setup.py bdist_wheel`` and transfer the created dist file to the Clients
This step will be eliminated once the library is published on PyPI.
5. Configure your *Clients* as per instructions below
6. Run ``python flox/examples/quickstart_pytorch/pytorch_funcx.py``


On your *Clients*:
1. Make sure you are using a 64-bit OS, this will make PyTorch installation easier
2. Install the FLoX dependencies from the dist file you created on the Controller: ``pip install pyflox-0.1.5-py3-none-any.whl``
3. Install `PyTorch <https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html>`_ & `Torchvision <https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html#:~:text=rm%20%2Drf%20~/pytorch-,TorchVision.,-Install%20torchvision%20on>`_
4. Install ``funcx-endpoint`` with ``pip install funcx-endpoint``
5. Configure (``funcx-endpoint configure ep1``) and start (``funcx-endpoint start ep1``) an endpoint.
6. Paste the endpoint's id (``funcx-endpoint list``) and paste it into ``pytorch_funcx.py`` on the *Controller* at the endpoint defintion line


