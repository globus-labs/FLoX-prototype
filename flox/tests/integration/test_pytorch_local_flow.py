def test_broadcast_local(pytorch_controller):
    """The simple federated learning process runs without throwing errors"""
    pytorch_controller.run_federated_learning()
    assert True
