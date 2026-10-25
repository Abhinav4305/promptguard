"""Evaluator registry: name -> class. Adding an evaluator = define a class and @register it."""

from typing import Any, Mapping

from app.core.evaluators.base import Evaluator, EvaluatorConfigError, check_evaluator_class

_REGISTRY: dict[str, type[Evaluator]] = {}


def register(cls: type[Evaluator]) -> type[Evaluator]:
    check_evaluator_class(cls)
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"evaluator name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def create_evaluator(name: str, config: Mapping[str, Any] | None = None) -> Evaluator:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise EvaluatorConfigError(f"unknown evaluator {name!r}; available: {available()}") from None
    return cls(config)
