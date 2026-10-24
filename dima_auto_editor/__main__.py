from __future__ import annotations

import argparse
import sys

INSTALL_HINT = (
    "PySide6 is not installed in this Python environment.\n"
    "Run install.ps1 (powershell -ExecutionPolicy Bypass -File install.ps1) and start the app with run.bat."
)


def report_startup_error(message: str) -> None:
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    elif sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "Dima Auto Editor", 0x10)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dima_auto_editor", description="Dima Auto Editor")
    parser.add_argument("--quit-after", type=float, default=None, metavar="SECONDS",
                        help="Close the window automatically after SECONDS (used for smoke checks).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        from .gui.app import run_app
    except ImportError as error:
        report_startup_error(f"{INSTALL_HINT}\n\nDetails: {error}")
        return 1
    return run_app(sys.argv[:1], quit_after_s=args.quit_after)


if __name__ == "__main__":
    sys.exit(main())
