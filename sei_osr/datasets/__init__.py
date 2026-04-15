def build_data_module(config: dict):
    from .build import build_data_module as _build_data_module

    return _build_data_module(config)


__all__ = ["build_data_module"]
