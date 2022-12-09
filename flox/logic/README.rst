FLoX Client-Controller Logic
============================

FLoX is designed to be a highly modular and customizable framework for
serverless, FL processes. It is built on top of a general 10-step
abstraction for FL processes grouped into controller-side and
client-side logical steps. FLoX considers two abstract classes,
``FloxControllerLogic`` and ``FloxClientLogic``, for
wrapping/implementing the logic needed for the controller-side and
client-side, respectively. Each step is run on either the Controller (S)
or the clients (C). All 7 steps are listed below with the respective
function that corresponds with them:

1.  **(S)** Model Initialization: ``on_model_init``
2.  **(S)** Model Sharing: ``on_model_broadcast``
3.  **(C)** Receiving Model on Client: ``on_model_receive``
4.  **(C)** Data Fetching: ``on_data_retrieve``
5.  **(C)** Local Model Training: ``on_model_fit``
6.  **(C)** Model Parameter Submission: ``on_model_send``
7.  **(S)** Receiving Model on Controller: ``on_model_receive``
8.  **(S)** Model Aggregation: ``on_model_aggregate``
9.  **(S)** Model Updating: ``on_model_update``
10. **(S)** Model Evaluation: ``on_model_evaluate``

Below is a (``mermaid.js``) figure showing the sequence of the logic
steps. Then, we further describe each of the logic steps.

.. code:: mermaid

   flowchart
       id1[[on_model_init]]
       id2[/on_model_broadcast\]

       id3[\on_model_receive/]
       id4[(on_data_retrieve)]
       id5{on_model_fit}
       id6[on_model_send]

       id7[on_model_receive]
       id8[on_model_aggregate]
       id9[on_model_update]
       id10[on_model_evaluate]



       subgraph controller
           id1-->id2
           id7-->id8
           id8-->id9
           id9-->id10
           id10-->id2
       end

       subgraph client
           direction TB
           id3-->id4
           id4-->id5
           id5-->id6
       end

       id2-->|model params|id3
       id6-->|model params|id7

1. Model Initialization
-----------------------

… ``Controller.on_model_init()`` is where one would provide initial
setup scripts that need to be run only once rather than needing to run
every FL round. In ``flox.controllers.TensorflowController`` we use
``.on_model_init()`` to set up variables that were not provided by the
user but will be reused in all of the FL rounds going forward.

2. Model Sharing
----------------

… Model sharing happend in ``Controller.on_model_broadcast`` where one
would assemble all variables into a config file, encrypt data is
necessary, and then send the tasks to the *clients* using an Executor,
such as FuncXExecutor. The method should return a list of futures/tasks
that can later be parsed out by ``Controller.on_model_receive()`` once
*clients* return the results.

3. Receiving Model on Client
----------------------------

… Once *clients* receive the model and the config with necessary
parameters, ``Client.on_model_receive()`` is responsible for the initial
actions such as decrypting the data if it wsa encrypted.

4. Data Retrieval
-----------------

… ``Client.on_data_retrieve()`` is where *clients* retrieve and prepare
their data for training.

5. Local Model Training
-----------------------

… ``Client.on_data_fit()`` is where the training process is defined and
executed.

6. Model Parameter Submission
-----------------------------

… When *clients* retrieved local data and made updates to the global
model, the new model weights are returned to the *Controller* and
``Client.on_data_send()`` is for things like encryption of data before
it is sent back.

7. Receiving Model on Controller
--------------------------------

Once the *Controller* receives the results back from the *clients*,
``Controller.on_model_receive()`` parses the results and decrypts if
necessary.

8. Model Aggregation
--------------------

… ``Controller.on_model_aggregate()`` takes the parsed results from
``Controller.on_model_receive()`` and aggregates weights from the
endpoints.

9. Model Updating
-----------------

… ``Controller.on_model_update()`` simply takes the new weights from
``Controller.on_model_aggregate()`` and assigns them to the global
model.

10. Model Evaluation
--------------------

… Finally, ``Controller.on_model_evaluate()`` evaluates the model using
a user-provided testing dataset, reports the results, and then the
entire loop from Step 2 to Step 10 is repeated for as many rounds as was
specified by the user.

We implemented the abstract base classes in
``flox.logic.base_client.py`` and ``flox.logic.base_controller.py``. We
also implemented a base class for Machine Learning Model Trainers,
located in ``flox.logic.base_model_trainer.py``. We are providing
practical examples on top of these classes to illustrate how all of
these steps come together: - ``flox.examples.quickstart_pytorch`` makes
use of ``PyTorchController``, ``PyTorchClient``, and ``PyTorchTrainer``
to run a Federated Learning workflow on PyTorch. -
``flox.examples.quickstart_tensorflow`` makes use of
``TensorflowController``, ``TensorflowClient``, and
``TensorflowTrainer`` to run a Federated Learning workflow on
Tensorflow.
