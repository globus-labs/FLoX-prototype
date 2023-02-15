def test_broadcast_local(tf_controller):
    """The simple federated learning process runs without throwing errors"""
    tf_controller.run_federated_learning()
    assert True


def test_broadcast_local_running_average(tf_controller):
    """The running_average version of federated learning process runs without throwing errors"""
    tf_controller.running_average = True
    tf_controller.run_federated_learning()
    assert True
