"""
dynamo_validation_demo.py
=========================

Run:   python dynamo_validation_demo.py

Expected output:

    Loading GoodService               ... OK
    Loading GrandChildService         ... OK
    Loading OverlapGoodService        ... OK
    Loading MissingEndpointService    ... FAILED (missing implementation(s): baz)
    Loading UndecoratedService        ... FAILED (method(s) not decorated with @dynamo_endpoint: foo)
    Loading BadStackService           ... FAILED (method(s) not decorated with @dynamo_endpoint: bar)
    Loading ShadowedService           ... FAILED (bar must be callable, got int)

This version **does not perform any signature checks**; it only verifies
presence, callability and decoration.
"""

import abc
import functools
import inspect
from typing import Callable, Set


# ────────────────────────────────────────────────────────────────────────────
# 1.  Endpoint decorators + base ABC
# ────────────────────────────────────────────────────────────────────────────
def dynamo_endpoint(func: Callable | None = None):
    """Mark a concrete endpoint."""
    def _wrap(f: Callable):
        @functools.wraps(f)
        def wrapper(*args, **kw):
            # … real runtime work would go here …
            return f(*args, **kw)

        wrapper.__is_dynamo_endpoint__ = True
        return wrapper

    return _wrap(func) if func else _wrap


def abstract_dynamo_endpoint(func: Callable):
    """Mark an abstract endpoint in an interface."""
    func.__is_abstract_dynamo__ = True
    return abc.abstractmethod(func)


class DynamoServiceInterface(abc.ABC):
    """Just an ABC anchor – uses stock ABCMeta."""
    pass


# ────────────────────────────────────────────────────────────────────────────
# 2.  Helper utilities for the validator
# ────────────────────────────────────────────────────────────────────────────
def _is_dynamo(func: Callable | None) -> bool:
    """True if *any* wrapper in the chain carries the marker flag."""
    while func:
        if getattr(func, "__is_dynamo_endpoint__", False):
            return True
        func = getattr(func, "__wrapped__", None)  # unwrap one layer
    return False


# ────────────────────────────────────────────────────────────────────────────
# 3.  The enhanced class decorator  (no signature checks)
# ────────────────────────────────────────────────────────────────────────────
def dynamo_service(cls: type):
    """
    Validate that *cls* fully implements every @abstract_dynamo_endpoint
    declared in its ancestors and that each implementation is
    decorated with @dynamo_endpoint.
    """

    # 1️⃣  collect abstract endpoint names from the full MRO
    required: Set[str] = {
        name
        for base in cls.mro()
        for name, val in base.__dict__.items()
        if getattr(val, "__is_abstract_dynamo__", False)
    }

    missing, undecorated, not_callable = [], [], []

    for name in required:
        impl = getattr(cls, name, None)  # walk MRO (handles grand-parent impls)
        if impl is None:
            missing.append(name)
            continue

        # guard against attribute shadowing with a non-callable
        if not callable(impl):
            not_callable.append((name, type(impl).__name__))
            continue

        if not _is_dynamo(impl):
            undecorated.append(name)

    problems = []
    if missing:
        problems.append(f"missing implementation(s): {', '.join(missing)}")
    if undecorated:
        problems.append(
            f"method(s) not decorated with @dynamo_endpoint: {', '.join(undecorated)}"
        )
    if not_callable:
        problems.append(
            ", ".join(f"{n} must be callable, got {kind}" for n, kind in not_callable)
        )

    if problems:
        raise TypeError(f"{cls.__name__} violates Dynamo interface — " + "; ".join(problems))

    return cls


# ────────────────────────────────────────────────────────────────────────────
# 4.  Interfaces
# ────────────────────────────────────────────────────────────────────────────
class CoreInterface(DynamoServiceInterface):
    @abstract_dynamo_endpoint
    def foo(self, x: int) -> int: ...

    @abstract_dynamo_endpoint
    def bar(self, y: int) -> int: ...


class ExtraInterface(DynamoServiceInterface):
    @abstract_dynamo_endpoint
    def baz(self, z: int) -> int: ...


class AltCoreInterface(DynamoServiceInterface):
    @abstract_dynamo_endpoint
    def foo(self, x: int) -> int: ...


# ────────────────────────────────────────────────────────────────────────────
# 5.  Concrete services to test
# ────────────────────────────────────────────────────────────────────────────
@dynamo_service
class GoodService(CoreInterface, ExtraInterface):
    @dynamo_endpoint
    def foo(self, x: int) -> int: return x + 1

    @dynamo_endpoint
    def bar(self, y: int) -> int: return y + 1

    @dynamo_endpoint
    def baz(self, z: int) -> int: return z + 1


# grand-parent implements everything; child adds nothing
@dynamo_service
class ParentService(CoreInterface):
    @dynamo_endpoint
    def foo(self, x): return x + 1
    @dynamo_endpoint
    def bar(self, y): return y + 1

@dynamo_service
class GrandChildService(ParentService):
    pass


# overlapping name in two interfaces
@dynamo_service
class OverlapGoodService(CoreInterface, AltCoreInterface):
    @dynamo_endpoint
    def foo(self, x): return x + 1
    @dynamo_endpoint
    def bar(self, y): return y + 1


# — fails: baz missing
try:
    @dynamo_service
    class MissingEndpointService(CoreInterface, ExtraInterface):
        @dynamo_endpoint
        def foo(self, x): return x + 1
        @dynamo_endpoint
        def bar(self, y): return y + 1
except TypeError as e:
    MissingEndpointService = e


# — fails: foo present but not decorated
try:
    @dynamo_service
    class UndecoratedService(CoreInterface):
        def foo(self, x): return x + 1      # ← missing decorator
        @dynamo_endpoint
        def bar(self, y): return y + 1
except TypeError as e:
    UndecoratedService = e


# — fails: decorator stacking loses marker
def other_deco(f):
    def w(*a, **k): return f(*a, **k)
    return w

try:
    @dynamo_service
    class BadStackService(CoreInterface):
        @other_deco                 # outermost, hides the marker
        @dynamo_endpoint
        def foo(self, x): return x + 1

        @other_deco
        @dynamo_endpoint
        def bar(self, y): return y + 1
except TypeError as e:
    BadStackService = e


# — fails: method shadowed by non-callable
try:
    @dynamo_service
    class ShadowedService(ParentService):
        bar = 123                   # hides the callable
except TypeError as e:
    ShadowedService = e


# ────────────────────────────────────────────────────────────────────────────
# 6.  Quick sanity printout
# ────────────────────────────────────────────────────────────────────────────
def _status(obj):
    return f"FAILED ({obj})" if isinstance(obj, TypeError) else "OK"


if __name__ == "__main__":
    tests = {
        "GoodService": GoodService,
        "GrandChildService": GrandChildService,
        "OverlapGoodService": OverlapGoodService,
        "MissingEndpointService": MissingEndpointService,
        "UndecoratedService": UndecoratedService,
        "BadStackService": BadStackService,
        "ShadowedService": ShadowedService,
    }

    for name, obj in tests.items():
        print(f"Loading {name:<24} ... {_status(obj)}")
