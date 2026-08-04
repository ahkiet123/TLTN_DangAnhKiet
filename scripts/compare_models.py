from __future__ import annotations

import runpy

import _bootstrap  # noqa: F401


if __name__ == "__main__":
    runpy.run_module("durian_leaf_disease.models.transfer", run_name="__main__")

