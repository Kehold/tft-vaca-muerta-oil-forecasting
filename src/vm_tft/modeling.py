from typing import Dict, List, Optional
import torch
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.callbacks import TFMProgressBar

DEFAULT_QUANTILES = [0.1,0.5,0.9]

def default_add_encoders(encode_year):
    return {
        "cyclic": {"past": ["month"], "future": ["month"]},
        "datetime_attribute": {"past": ["month","year"], "future": ["month","year"]},
        "position": {"past": ["relative"], "future": ["relative"]},
        "custom": {"past": [encode_year], "future": [encode_year]},
    }

def build_tft(categorical_embedding_sizes: Dict[str,int],
              input_chunk_length: int, output_chunk_length: int,
              add_encoders: dict,
              lr: float = 0.000820009391166529,
              hidden_size: int = 64,
              hidden_continuous_size: int = 24,
              lstm_layers: int = 2,
              n_heads: int = 4,
              dropout: float = 0.05,
              n_epochs: int = 50,
              batch_size: int = 128,
              quantiles: List[float] = DEFAULT_QUANTILES,
              seed: int = 42,
              accelerator: str = "gpu",
              devices: List[int] = [0]):
    return TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        num_attention_heads=n_heads,
        n_epochs=n_epochs,
        lstm_layers=lstm_layers,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        dropout=dropout,
        add_relative_index=False,
        categorical_embedding_sizes=categorical_embedding_sizes,
        use_static_covariates=True,
        add_encoders=add_encoders,
        loss_fn=None,
        likelihood=QuantileRegression(quantiles=quantiles),
        random_state=seed,
        pl_trainer_kwargs={"accelerator": accelerator, "devices": devices, "callbacks": [TFMProgressBar()]},
        show_warnings=True,
    )
