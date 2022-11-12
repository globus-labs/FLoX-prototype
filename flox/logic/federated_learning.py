def federated_learning(
    ServerLogic, ClientLogic, ModelTrainer, Executor, rounds, endpoint_ids
):

    # 1. Get model architecture
    global_model_config = ModelTrainer.get_architecture()

    # 2. Start FL rounds
    for _ in range(rounds):

        # 3. Get the model weights
        global_model_weights = ModelTrainer.get_weights()

        # 4. Put data in a dictionary/JSON
        data = {
            "model_config": global_model_config,
            "model_weights": global_model_weights,
        }

        # 4. Submit the tasks to endpoints
        for e in endpoint_ids:
            Executor.submit(ClientLogic, data, e)

        # 5. Retrieve results
        results = Executor.get_results()

        # 6. Aggregate weights
        updated_weights = ServerLogic.aggregate(results)

        # 7. Update weights
        ModelTrainer.update_weights(updated_weights)

        # 8. Evaluate model
        ModelTrainer.evaluate()
