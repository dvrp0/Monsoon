import os, importlib

dirname = os.path.dirname(__file__)

for card in os.listdir(dirname):
    if card != "__init__.py" and os.path.isfile(f"{dirname}/{card}") and card[-3:] == ".py":
        module = importlib.import_module(f".{card[:-3]}", __package__)

        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isinstance(attribute, type):
                locals()[attribute_name] = attribute