import argparse
import functools
import json
import os
from collections.abc import Callable
from typing import get_type_hints

from pydantic import TypeAdapter, ValidationError


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
            hints = get_type_hints(type(cur))
            if final_key not in hints:
                raise AttributeError(f"{type(cur)} has no config option {final_key!r}")

            if value_str.lower() == "none":
                value_str = None
            else:
                # Keep scalar/string overrides intact; decode JSON when the field cannot
                # accept a string (e.g. a list, tuple, dict, or nested model).
                try:
                    TypeAdapter(hints[final_key]).validate_python(value_str)
                except ValidationError:
                    try:
                        value_str = json.loads(value_str)
                    except json.JSONDecodeError:
                        pass  # Let assignment validation report the invalid value.

            setattr(cur, final_key, value_str)
            casted_overrides.append(f"{key}={getattr(cur, final_key)}")

        if len(args.overrides) > 0 and int(os.environ.get("RANK", "0")) == 0:
            print(f"running with overrides: {' '.join(casted_overrides)}")

        return main(config)

    return wrapped_main
