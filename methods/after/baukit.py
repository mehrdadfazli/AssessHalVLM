from types import SimpleNamespace


class Trace:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class TraceDict:
    def __init__(self, model, layers, edit_output=None):
        self.model = model
        self.layers = list(layers)
        self.edit_output = edit_output
        self.handles = []
        self.output = {}

    def __enter__(self):
        for layer in self.layers:
            module = self._resolve_module(layer)
            if module is None:
                raise ValueError(f"Could not resolve layer '{layer}' on model {self.model}")
            handle = module.register_forward_hook(self._make_hook(layer))
            self.handles.append(handle)
        return self

    def _resolve_module(self, layer_name):
        if hasattr(self.model, "get_submodule"):
            try:
                return self.model.get_submodule(layer_name)
            except (AttributeError, KeyError, ValueError):
                pass

        module = self.model
        for part in layer_name.split('.'):
            if not hasattr(module, part):
                return None
            module = getattr(module, part)
        return module

    def _make_hook(self, layer_name):
        def hook(module, inputs, output):
            result = output
            if self.edit_output is not None:
                edited = self.edit_output(output, layer_name)
                if edited is not None:
                    result = edited
            self.output[layer_name] = SimpleNamespace(output=result)
            return result

        return hook

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False

    # Match common `baukit.TraceDict` ergonomics: allow `ret[layer_name]` instead of `ret.output[layer_name]`.
    def __getitem__(self, layer_name):
        return self.output[layer_name]
