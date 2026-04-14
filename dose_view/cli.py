import argparse
import json
import inspect
import sys

from dose_view import io, patient, visualization, comparison


def list_functions():
    """List all available functions from all modules."""
    modules = {
        'io': io,
        'patient': patient,
        'visualization': visualization,
        'comparison': comparison,
    }
    
    public = []
    for module_name, module in modules.items():
        for name, obj in vars(module).items():
            if callable(obj) and not name.startswith("_"):
                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "(...)"
                public.append(f"{module_name}.{name}{sig}")
    return "\n".join(sorted(public))


def main():
    parser = argparse.ArgumentParser(
        description="Run a function from dose_view package"
    )
    parser.add_argument(
        "func", 
        help="Function name to call (format: module.function), or 'list' to show available"
    )
    parser.add_argument("--kwargs", help="JSON dict of keyword args", default="{}")
    args = parser.parse_args()

    if args.func == "list":
        print(list_functions())
        sys.exit(0)

    # Parse module.function
    if '.' not in args.func:
        print(f"Function must be in format 'module.function'", file=sys.stderr)
        print("Available:\n" + list_functions(), file=sys.stderr)
        sys.exit(1)
    
    module_name, func_name = args.func.rsplit('.', 1)
    
    modules = {
        'io': io,
        'patient': patient,
        'visualization': visualization,
        'comparison': comparison,
    }
    
    if module_name not in modules:
        print(f"Unknown module: {module_name}", file=sys.stderr)
        print("Available:\n" + list_functions(), file=sys.stderr)
        sys.exit(1)
    
    module = modules[module_name]
    
    if not hasattr(module, func_name):
        print(f"No such function: {args.func}", file=sys.stderr)
        print("Available:\n" + list_functions(), file=sys.stderr)
        sys.exit(1)

    func = getattr(module, func_name)
    try:
        kwargs = json.loads(args.kwargs) if args.kwargs else {}
    except Exception as e:
        print(f"Failed to parse --kwargs JSON: {e}", file=sys.stderr)
        sys.exit(2)

    res = func(**kwargs)
    try:
        print(res)
    except Exception:
        print(repr(res))


if __name__ == "__main__":
    main()

