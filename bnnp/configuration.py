import argparse
import functools
import json
import os
from collections.abc import Callable
from typing import get_type_hints

from pydantic import BaseModel, TypeAdapter, ValidationError


def _set_override(config, key, value, annotation):
    if isinstance(config, BaseModel):
        if config.model_config.get("frozen"):
            setattr(config, key, value)  # Preserve Pydantic's frozen-instance error.
        # Validate once with field/model validators, even without validate_assignment.
        config.__pydantic_validator__.validate_assignment(config, key, value)
    else:
        setattr(config, key, TypeAdapter(annotation).validate_python(value))


def parse_config(main: Callable):
    @functools.wraps(main)
    def wrapped_main():
        main_hints = get_type_hints(main)
        main_hints.pop("return", None)
        if len(main_hints) == 0:
            raise ValueError("config argument in main() must be type annotated")
        if len(main_hints) > 1:
            raise ValueError("expected main() function to take only one argument")
        try:
            config = next(iter(main_hints.values()))()
        except Exception as e:
            raise ValueError("failed to initialize default config") from e

        parser = argparse.ArgumentParser()
        parser.add_argument("overrides", nargs="*")
        args = parser.parse_args()

        casted_overrides = []
        for override in args.overrides:
            key, value_str = override.split("=", 1)
            cur = config
            *prefix, final_key = key.split(".")
            for k in prefix:
                if not hasattr(cur, k):
                    raise AttributeError(f"{type(cur)} has no config option {k!r}")
                cur = getattr(cur, k)
            hints = get_type_hints(type(cur), include_extras=True)
            if final_key not in hints:
                raise AttributeError(f"{type(cur)} has no config option {final_key!r}")

            if value_str.lower() == "none":
                _set_override(cur, final_key, None, hints[final_key])
            else:
                # Keep scalar/string overrides intact; decode JSON when the field cannot
                # accept a string (e.g. a list, tuple, dict, or nested model).
                try:
                    _set_override(cur, final_key, value_str, hints[final_key])
                except ValidationError as exc:
                    try:
                        value = json.loads(value_str)
                    except json.JSONDecodeError:
                        raise exc from None
                    _set_override(cur, final_key, value, hints[final_key])

            casted_overrides.append(f"{key}={getattr(cur, final_key)}")

        if len(args.overrides) > 0 and int(os.environ.get("RANK", "0")) == 0:
            print(f"running with overrides: {' '.join(casted_overrides)}")

        return main(config)

    return wrapped_main
