from flox.common import (
    AggregateIns,
    AggregateRes,
    BroadcastRes,
    ReceiveIns,
    ReceiveRes,
    UpdateIns,
)


class FloxControllerLogic:
    def on_model_init(self) -> None:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_broadcast(self) -> BroadcastRes:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_aggregate(self, ins: AggregateIns) -> AggregateRes:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_receive(self, ins: ReceiveIns) -> ReceiveRes:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")

    def on_model_update(self, ins: UpdateIns) -> None:
        """DocString"""
        raise NotImplementedError("Abstract class method. Cannot be called directly.")
