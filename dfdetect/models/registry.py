from typing import Any, Dict

from .unet import UNetClassifier


def build_model(model_cfg: Dict[str, Any]):
    name = model_cfg.get("type", "")
    params = model_cfg.get("params", {})
    if name == "unet":
        # no defaults here: require all fields in YAML
        return UNetClassifier(
            depth=int(params["depth"]),
            base_channels=int(params["base_channels"]),
            out_channels=int(params["out_channels"]),
            conv_kernel=int(params["conv_kernel"]),
            conv_padding=int(params["conv_padding"]),
            norm=str(params["norm"]),
            activation=str(params["activation"]),
            activation_negative_slope=float(params["activation_negative_slope"]),
            up_kernel=int(params["up_kernel"]),
            up_stride=int(params["up_stride"]),
            pool_kernel=int(params["pool_kernel"]),
            pool_stride=int(params["pool_stride"]),
            classifier=str(params["classifier"]),
        )
    # resnet
    # whatever else we are adding...
    raise NotImplementedError(f"Model '{name}' is not implemented yet. Add implementation and update registry.")
