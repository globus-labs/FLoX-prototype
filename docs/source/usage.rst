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

