"""Run the IRR aggregate example: ``python examples/run_irr.py``.

The example is ``irr.py``, which must be imported rather than run. numbox's ``make_structref`` generates the code of
the aggregate's state, and numba caches that code together with the state's type, ``IRRStateType``. A class defined in
the script that Python runs belongs to ``__main__``, and it does not match the cached copy in the next process. On a
warm cache, type inference then fails with a ``TypingError`` that reports no conversion from ``IRRStateType`` to
``IRRStateType``. This launcher puts its own directory on ``sys.path``, imports the module as ``irr`` and calls its
``main()``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from irr import main  # noqa: E402


if __name__ == "__main__":
    main()
