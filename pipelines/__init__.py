MODELS = ['flux2-dev', 'flux2-klein', 'qwen-2511']

def load_editor(model, cache_dir, device):
    if model in ('flux2-dev', 'flux2-klein'):
        from .flux2 import Flux2Editor
        return Flux2Editor(model, cache_dir, device)
    if model == 'qwen-2511':
        from .qwen import QwenEditor
        return QwenEditor(cache_dir, device)
    raise ValueError(f"Unknown model '{model}', choose from {MODELS}")
