from brown import config, observability

__all__ = ["config"]

if config.get_settings().OPIK_ENABLED:
    observability.configure()
