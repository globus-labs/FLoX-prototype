
# Quickstart Tensorflow Tutorial

This example is in development and includes only basic setup and execution instructions.

### Controller instructions

On your *Controller*, follow these instructions:

1. Create a conda or virtualenv environment with ``python=3.7``: ``conda create --name tf python=3.7``

2. Install FLoX with ``pip install -e .`` since the latest version of the library is not out yet.

3. Install [Tensorflow 2.1.0](https://www.tensorflow.org/install/pip>): ``pip install tensorflow==2.1.0``.
Tensorflow Verstion between *Controller* and *Clients* need match.
We use 2.1.0 because it's the version that was easily installable on our Rasperry Pi 4.

4. (*if using real endpoints*) Create a dist file with ``python setup.py bdist_wheel`` and transfer the created dist file to the Clients
This step will be eliminated once the library is published on PyPI.

5. (*if using real endpoints*) Configure your *Clients* as per instructions below

6. Run ``python flox/examples/quickstart_tensorflow/tf_funcx.py``

### Clients instructions

On your *Clients*:
1. Install the FLoX dependencies from the dist file you created on the Controller: ``pip install pyflox-0.1.5-py3-none-any.whl``

2. Install [Tensorflow 2.1.0](https://qengineering.eu/install-tensorflow-2.1.0-on-raspberry-pi-4.html)

3. Install ``funcx-endpoint`` with ``pip install funcx-endpoint``

4. Configure (``funcx-endpoint configure ep1``) and start (``funcx-endpoint start ep1``) an endpoint.

5. Paste the endpoint's id (``funcx-endpoint list``) and paste it into ``pytorch_funcx.py`` on the *Controller* at the endpoint definition line

6. For local dataset execution, place the files ``x_animal10_32.npy`` and ``y_animal10_32.npy`` (which you can find [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10)) under ``/home/pi/datasets`` directory. Otherwise, if you want to have a different directory structure and filenames, just change those variables when instantiating ``flox_controller``.