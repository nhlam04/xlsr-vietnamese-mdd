"""
Re-export shim.

Importing from 'datasets' inside this package would shadow HuggingFace's
'datasets' library.  All project data-loading code lives in datasets.py;
external callers should import from here under the alias 'datasets_loader'.
"""

# Explicitly import from the project module (not the installed library).
import importlib, sys

_mod = importlib.import_module('datasets', package=None)
# If the resolved module is HuggingFace datasets, fall back to local file.
if not hasattr(_mod, 'load_all_datasets'):
    import importlib.util, os
    _spec = importlib.util.spec_from_file_location(
        '_project_datasets',
        os.path.join(os.path.dirname(__file__), 'datasets.py'),
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

load_all_datasets           = _mod.load_all_datasets
prepare_dataset_with_labels = _mod.prepare_dataset_with_labels
load_timit_data             = _mod.load_timit_data
load_lsvsc_data             = _mod.load_lsvsc_data
load_l2_arctic_data         = _mod.load_l2_arctic_data
split_timit_by_speakers     = _mod.split_timit_by_speakers

__all__ = [
    'load_all_datasets',
    'prepare_dataset_with_labels',
    'load_timit_data',
    'load_lsvsc_data',
    'load_l2_arctic_data',
    'split_timit_by_speakers',
]
