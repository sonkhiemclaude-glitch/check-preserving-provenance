import importlib

import pytest

MODULES = ["provpub", "provpub.cli", "provpub.data", "provpub.data.schemas", "provpub.data.spec", "provpub.data.spec_io",
           "provpub.data.synthetic", "provpub.data.prov_io", "provpub.data.adapters", "provpub.methods",
           "provpub.methods.operators", "provpub.methods.exact_dlap", "provpub.checks", "provpub.baselines",
           "provpub.attacks", "provpub.metrics", "provpub.experiments.runner", "provpub.experiments.e1_e3",
           "provpub.experiments.e2", "provpub.experiments.e4", "provpub.experiments.tables",
           "provpub.experiments.plots", "provpub.experiments.real_data", "provpub.utils.io",
           "provpub.utils.validation", "provpub.utils.random"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_version():
    import provpub
    assert provpub.__version__
