"""Launcher for the IRR UDAF example.

``examples/irr.py`` defines ``IRRStateType`` at module level. Running
``irr.py`` directly would set ``IRRStateType.__module__`` to ``__main__``,
which gives the class a fresh identity each process and breaks numba's
warm-cache type inference. This launcher imports ``irr`` so the class
gets a stable ``__module__ == "irr"`` and runs ``main()`` from there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from irr import main  # noqa: E402


if __name__ == "__main__":
    main()
