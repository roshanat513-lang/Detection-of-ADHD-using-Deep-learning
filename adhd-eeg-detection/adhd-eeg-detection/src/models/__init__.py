from src.models.eegnet import EEGNet
from src.models.cnn_lstm import CNNLSTM


def build_model(config: dict, n_channels: int, n_timesteps: int):
    """Instantiate the model named in config['model']['name']."""
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    num_classes = model_cfg.get("num_classes", 2)

    if name == "eegnet":
        p = model_cfg["eegnet"]
        return EEGNet(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            F1=p["F1"],
            D=p["D"],
            F2=p["F2"],
            kernel_length=p["kernel_length"],
            dropout=p["dropout"],
        )

    if name == "cnn_lstm":
        p = model_cfg["cnn_lstm"]
        return CNNLSTM(
            n_channels=n_channels,
            n_timesteps=n_timesteps,
            num_classes=num_classes,
            cnn_channels=p["cnn_channels"],
            lstm_hidden=p["lstm_hidden"],
            lstm_layers=p["lstm_layers"],
            bidirectional=p["bidirectional"],
            dropout=p["dropout"],
        )

    raise ValueError(f"Unknown model name: {name}")
