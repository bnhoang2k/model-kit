"""The `model-kit` command: init, guide, gate, sheet, profile and bars.

    model-kit init [--dir D]
    model-kit guide [--where]
    model-kit gate <scene> [--checks a,b] [--allow FILE]
    model-kit sheet <pack.pack.json> --piece P --style S [--bar auto|<slug>|none] ...
    model-kit profile <build.py> [--blend B] [--top 30] [-- args ...]
    model-kit bars [--root bars] [--check]

Exit codes: 0 done, 1 a finding (a failed or partial gate, a stale bars index), 2 cannot start
or the run was not ok. A usage error prints the usage and exits 2. Nothing here falls back
silently: a run that is not ok says why on stderr.
"""

from __future__ import annotations

import argparse
import pstats
import sys
import tempfile
from pathlib import Path

from model_kit import bars as bars_module
from model_kit import init as init_module
from model_kit import sheet
from model_kit.blender import runner
from model_kit.core import gate
from model_kit.core.errors import ManifestError

CANNOT_START = gate.CANNOT_START
PROFILE_SCRIPT = Path(__file__).resolve().parent / "blender" / "_profile.py"
PROFILE_TIMEOUT = 1800


def _cannot_start(exc: Exception) -> int:
    message = " ".join(str(exc).split())  # one line, whatever the error's text holds
    print(f"cannot start: {type(exc).__name__}: {message}", file=sys.stderr)
    return CANNOT_START


def _sheet_views(text: str) -> list[sheet.View]:
    known = {view[0]: view for view in sheet.VIEWS}
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown view {', '.join(map(repr, unknown)) or repr(text)}; the views are "
            f"{', '.join(known)}")
    return [known[name] for name in names]


def _run_sheet(args: argparse.Namespace) -> int:
    out = sheet.make_sheet(args.pack, args.piece, args.style, bar=args.bar, bars=args.bars,
                           out=args.out, views=args.views)
    rows = 1 if args.bar == "none" else 2
    print(f"wrote {out}: {rows} rows x {len(args.views)} views")
    return 0


def _run_profile(args: argparse.Namespace) -> int:
    build = args.build
    if not build.is_file():
        raise ManifestError(f"{build}: no build script there")
    if args.blend is not None and not args.blend.is_file():
        raise ManifestError(f"{args.blend}: no .blend file there")
    runner.blender_path()  # fail early, naming MODEL_KIT_BLENDER
    prof = Path.cwd() / f"{build.stem}.prof"
    prof.unlink(missing_ok=True)  # an old profile must never pass for this run's
    with tempfile.TemporaryDirectory(prefix="model-kit-profile-") as tmp:
        blend = runner.frozen_copy(args.blend, Path(tmp)) if args.blend is not None else None
        result = runner.run_blender(
            PROFILE_SCRIPT, [str(build.resolve()), str(prof), *args.args], blend=blend,
            timeout=PROFILE_TIMEOUT)
    if not result.ok:
        print(f"the profiled run was not ok (Blender exit {result.code}):\n{result.log[-2000:]}",
              file=sys.stderr)
        return CANNOT_START
    if not prof.is_file():
        print(f"the run was ok but wrote no {prof}", file=sys.stderr)
        return CANNOT_START
    print(f"wrote {prof}")
    stats = pstats.Stats(str(prof), stream=sys.stdout)
    stats.sort_stats("cumulative").print_stats(args.top)
    return 0


def _run_bars(args: argparse.Namespace) -> int:
    root = bars_module.bars_root(args.root, "--root")
    text = bars_module.render_index(bars_module.read_bars(root))
    index = root / "INDEX.md"
    if args.check:
        current = index.read_text() if index.is_file() else None
        if current != text:
            print(f"{index} is stale; run `model-kit bars` to rewrite it", file=sys.stderr)
            return 1
        print(f"{index} is up to date")
        return 0
    index.write_text(text)
    print(f"wrote {index}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-kit", description="Build and check 3D packs: init, the guide, the gate, the sheet, the profile, the bars.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    init_p = commands.add_parser(
        "init", help="install the kit for a project and point its agent at the guide",
        description="Install this release of the kit for a project (a uv dependency inside a "
                    "Python project, else a uv tool) and add the guide line to CLAUDE.md, and to "
                    "AGENTS.md when the project has one. A second run changes nothing.")
    init_p.add_argument("--dir", type=Path, default=Path.cwd(), help="the project folder (default: here)")
    init_p.set_defaults(run=init_module.run_init)

    guide_p = commands.add_parser("guide", help="print the steps a coding agent follows",
                                  description="Print the kit's AGENTS.md.")
    guide_p.add_argument("--where", action="store_true",
                         help="print where this install keeps its bars and kind rules")
    guide_p.set_defaults(run=init_module.run_guide)

    gate_p = commands.add_parser("gate", help="run named checks on a saved scene folder",
                                 description="Run named checks on a saved scene folder.")
    gate.add_arguments(gate_p)
    gate_p.set_defaults(run=gate.run)

    sheet_p = commands.add_parser("sheet", help="render a piece beside its bar as a contact sheet",
                                  description="Render a piece beside its bar as a contact sheet.")
    sheet_p.add_argument("pack", type=Path, help="the <name>.pack.json file")
    sheet_p.add_argument("--piece", required=True, help="the piece to render")
    sheet_p.add_argument("--style", required=True, type=Path, help="the style file")
    sheet_p.add_argument("--bar", default="auto", help="auto (the class's default), a slug, or none")
    sheet_p.add_argument("--bars", type=Path, help="the bars folder (default: MODEL_KIT_BARS, else the copy shipped "
                            "with the package, else the bars/ folder of the source tree)")
    sheet_p.add_argument("--out", type=Path, default=Path("sheet.png"), help="default sheet.png")
    sheet_p.add_argument("--views", type=_sheet_views, default=list(sheet.VIEWS),
                         help=f"comma-separated views (default: all): "
                              f"{', '.join(v[0] for v in sheet.VIEWS)}")
    sheet_p.set_defaults(run=_run_sheet)

    profile_p = commands.add_parser(
        "profile", help="run a build script under cProfile inside Blender",
        usage="model-kit profile [-h] [--blend BLEND] [--top TOP] build [-- build_args ...]",
        description="Run a build script under cProfile inside Blender. Writes <build>.prof in "
                    "the current folder and prints the top functions by cumulative time. Put "
                    "the build's own arguments after `--`.")
    profile_p.add_argument("build", type=Path, help="the build script")
    profile_p.add_argument("--blend", type=Path, help="a .blend file to open first (copied)")
    profile_p.add_argument("--top", type=int, default=30, help="functions to print (default 30)")
    profile_p.set_defaults(run=_run_profile)

    bars_p = commands.add_parser("bars", help="write the bars INDEX.md",
                                 description="Write the generated INDEX.md of a bars folder.")
    bars_p.add_argument("--root", type=Path, default=Path("bars"), help="the bars folder")
    bars_p.add_argument("--check", action="store_true",
                        help="exit 1 if INDEX.md is stale; write nothing")
    bars_p.set_defaults(run=_run_bars)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand and return its exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after the first `--` is the build's own arguments (profile only). argparse
    # cannot take them as a trailing positional once an option has come before it.
    build_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, build_args = argv[:cut], argv[cut + 1:]
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if build_args and args.command != "profile":
            parser.error(f"{args.command} takes no arguments after `--`")
    except SystemExit as exc:  # argparse has printed the usage, or the help
        return CANNOT_START if exc.code else 0
    args.args = build_args
    try:
        return args.run(args)
    except Exception as exc:  # noqa: BLE001 - a ManifestError or any other error before a verdict
        return _cannot_start(exc)


if __name__ == "__main__":
    sys.exit(main())
