import argparse
import json
import inspect
import sys

from dose_view import io, patient, visualization, comparison, triplanar, preview


def list_functions():
    """List all available functions from all modules."""
    modules = {
        'io': io,
        'patient': patient,
        'visualization': visualization,
        'comparison': comparison,
        'triplanar': triplanar,
        'preview': preview,
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


def _preview_main(argv):
    parser = argparse.ArgumentParser(prog="dose-view preview", description="Render dose-preview geometry reports using the source DICOM files.")
    parser.add_argument("preview_path", help="Preview directory or geometry JSON file")
    parser.add_argument("--output-dir", help="Defaults to a dose_view subdirectory")
    parser.add_argument("--recursive", action="store_true", help="Find reports in nested case directories")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously generated outputs")
    parser.add_argument("--path-map", default="{}", help="JSON map of old source-directory prefixes to new paths")
    parser.add_argument("--kwargs", default="{}", help="JSON display options, e.g. crop_mm, subtitle, dpi")
    args = parser.parse_args(argv)
    try:
        options = json.loads(args.kwargs)
        if not isinstance(options, dict):
            raise ValueError("--kwargs must be a JSON object.")
        reports = preview.plot_dose_preview(
            args.preview_path, args.output_dir, recursive=args.recursive,
            overwrite=args.overwrite, path_map=json.loads(args.path_map), **options)
    except (OSError, ValueError, TypeError, ImportError) as error:
        parser.exit(2, f"dose-view preview: {error}\n")
    print(json.dumps([{"figure": r["figure"], "geometry_report": r["geometry_report"]} for r in reports], indent=2))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "preview":
        return _preview_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Run a dose_view function, or use 'dose-view preview --help' to render a preview directory."
    )
    parser.add_argument(
        "func", 
        help="Function name to call (format: module.function), or 'list' to show available"
    )
    parser.add_argument("--kwargs", help="JSON dict of keyword args", default="{}")
    args = parser.parse_args(argv)

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
        'triplanar': triplanar,
        'preview': preview,
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

