from importlib import import_module

actions = {}


def register(action_name):
    def decorator(func):
        actions[action_name] = func
        return func
    return decorator


def register_all():
    module = import_module("spellcaster.actions")
    for submodule in module.__all__:
        import_module(f"spellcaster.actions.{submodule}")

register_all()