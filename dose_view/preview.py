"""Render previews from geometry sidecars, reading the original DICOM sources."""

import json
from pathlib import Path

from .triplanar import save_triplanar_overlay


def _source_path(value, manifest, path_map):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{manifest}: source paths must be nonempty strings.")
    path = Path(value).expanduser()
    # Match whole path components, longest prefix first (not string prefixes).
    for old, new in sorted(path_map.items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
        try:
            tail = path.relative_to(Path(old).expanduser())
        except ValueError:
            continue
        path = Path(new).expanduser() / tail
        break
    path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{manifest}: DICOM source is unavailable: {path}. "
            "Mount the source volume or supply path_map / --path-map."
        )
    return str(path)


def _preview_options(manifest, path_map):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{manifest}: expected a geometry JSON object.")
    if "sources" in data:
        sources = data["sources"]
        if not isinstance(sources, dict):
            raise ValueError(f"{manifest}: sources must be an object.")
        series = data.get("ct_series_instance_uid")
    elif isinstance(data.get("ct"), dict) and isinstance(data.get("dose"), dict):
        references = data.get("references", {})
        if not isinstance(references, dict):
            raise ValueError(f"{manifest}: references must be an object.")
        sources = {"CT": data["ct"].get("path"), "RTDOSE": data["dose"].get("path"),
                   "RTPLAN": references.get("rtplan_path"), "RTSTRUCT": references.get("rtstruct_path")}
        series = data["ct"].get("SeriesInstanceUID")
    else:
        raise ValueError(f"{manifest}: expected sources.CT / sources.RTDOSE or ct.path / dose.path.")
    for modality in ("CT", "RTDOSE"):
        if not sources.get(modality):
            raise ValueError(f"{manifest}: missing {modality} source path.")
    options = {}
    for modality, argument in (("CT", "ct_path"), ("RTDOSE", "rtdose_path"),
                               ("RTPLAN", "rtplan_path"), ("RTSTRUCT", "rtstruct_path")):
        if sources.get(modality) is not None:
            options[argument] = _source_path(sources[modality], manifest, path_map)
    if series:
        options["ct_series_uid"] = series

    display = data.get("display", {})
    if not isinstance(display, dict):
        raise ValueError(f"{manifest}: display must be an object.")
    for stored, argument in (("ct_window_level_HU", "ct_wl"), ("ct_window_width_HU", "ct_ww"),
                             ("center_xyz_mm", "center_xyz_mm"), ("crop_mm", "crop_mm"),
                             ("dose_threshold_fraction", "dose_threshold_fraction"),
                             ("alpha", "alpha"), ("cmap", "cmap"), ("dpi", "dpi"),
                             ("title", "title"), ("subtitle", "subtitle")):
        if display.get(stored) is not None:
            options[argument] = display[stored]
    if "center_xyz_mm" not in options and "crosshair_zyx_mm" in display:
        center = display["crosshair_zyx_mm"]
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError(f"{manifest}: crosshair_zyx_mm must have three coordinates.")
        options["center_xyz_mm"] = center[::-1]
    if "dose_threshold_fraction" not in options and "dose_threshold_Gy" in display:
        maximum = data.get("dose_max_Gy")
        if not isinstance(maximum, (int, float)) or maximum <= 0:
            raise ValueError(f"{manifest}: a positive dose_max_Gy is needed to recover the display threshold.")
        options["dose_threshold_fraction"] = display["dose_threshold_Gy"] / maximum
    return options


def plot_dose_preview(preview_path, output_dir=None, *, recursive=False,
                      overwrite=False, path_map=None, **plot_options):
    """Render the ``*_geometry.json`` / ``*.geometry.json`` reports in a directory.

    A single JSON report is also accepted. Both the original dose_preview
    sidecars and reports from :func:`save_triplanar_overlay` are supported.
    Actual CT/RTDOSE files must be accessible; a PNG and sidecar contain no
    volume pixels. Saved geometry checks are never treated as current checks.

    Saved crosshairs, window, threshold and other recorded display options are
    reused unless overridden. Supplying ``pick`` or ``center_threshold_fraction``
    recomputes the center unless ``center_xyz_mm`` is also supplied. Extra keyword
    arguments are display options for :func:`save_triplanar_overlay`.

    Output defaults to a new ``dose_view`` subdirectory. Recursive input keeps
    relative subdirectories; the output tree is excluded from discovery.
    Existing outputs require ``overwrite=True``. Input reports are never
    overwritten. ``path_map`` remaps source-directory prefixes for moved mounts.
    Returns one newly computed geometry report per rendered preview.
    """
    preview = Path(preview_path).expanduser().resolve()
    if not preview.exists():
        raise FileNotFoundError(preview)
    base = preview if preview.is_dir() else preview.parent
    destination = (Path(output_dir).expanduser() if output_dir is not None else base / "dose_view").resolve()
    if preview.is_dir() and (destination == base or destination in base.parents):
        raise ValueError("output_dir must be separate from the input directory (a subdirectory is allowed).")
    path_map = {} if path_map is None else path_map
    if not isinstance(path_map, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v for k, v in path_map.items()
    ):
        raise ValueError("path_map must map nonempty source prefixes to replacement paths.")
    reserved = {"ct_path", "rtdose_path", "rtplan_path", "rtstruct_path", "output_path"} & plot_options.keys()
    if reserved:
        raise ValueError(f"Source/output paths come from the preview; use path_map or output_dir instead: {sorted(reserved)}")
    if preview.is_file():
        if preview.suffix.lower() != ".json":
            raise ValueError("Supply a geometry JSON report or directory, not a PNG.")
        manifests = [preview]
    else:
        candidates = base.rglob("*.json") if recursive else base.glob("*.json")
        manifests = sorted(p for p in candidates if p.is_file()
                           and p.name.endswith(("_geometry.json", ".geometry.json"))
                           and destination not in p.resolve().parents)
    if not manifests:
        raise ValueError(f"No geometry sidecars found in {preview}. Use recursive=True / --recursive for nested directories.")

    jobs, targets = [], set()
    input_paths = {p.resolve() for p in manifests}
    # Resolve all source paths and output collisions before writing any plots.
    for manifest in manifests:
        stem = manifest.name
        for suffix in ("_geometry.json", ".geometry.json", ".json"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        filename = stem if stem.endswith("_triplanar") else stem + "_triplanar"
        output = destination / manifest.parent.relative_to(base) / (filename + ".png")
        for target in (output, output.with_suffix(".geometry.json")):
            resolved = target.resolve()
            if resolved in input_paths or resolved in targets:
                raise ValueError(f"Conflicting output path: {target}")
            if target.exists() and not overwrite:
                raise FileExistsError(f"Output already exists: {target}. Use overwrite=True / --overwrite to replace it.")
            targets.add(resolved)
        options = _preview_options(manifest, path_map)
        if "pick" in plot_options or "center_threshold_fraction" in plot_options:
            options.pop("center_xyz_mm", None)
        options.update(plot_options)
        jobs.append((manifest, output, options))

    reports = []
    for manifest, output, options in jobs:
        report = save_triplanar_overlay(output_path=output, **options)
        report["preview_source"] = str(manifest)
        Path(report["geometry_report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append(report)
    return reports
