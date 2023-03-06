import importlib


def get_runner(name):
    path = f'core.runner.{name}'
    runner = importlib.import_module(path).Runner
    return runner
