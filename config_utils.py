import os


def get_config_value(name, secrets=None):
    value = os.getenv(name)
    if value:
        return value

    if secrets is None:
        return None

    try:
        value = secrets.get(name)
    except (AttributeError, FileNotFoundError, KeyError):
        return None

    return value or None
