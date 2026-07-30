from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_streamlit_cli():
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        project_dir = Path(__file__).resolve().parent.parent
        requirements = project_dir / "requirements.txt"
        print(
            "Streamlit is not installed for this Python interpreter.\n"
            f"Python: {sys.executable}\n"
            f"Install the dependencies with:\n"
            f'  "{sys.executable}" -m pip install -r "{requirements}"\n\n'
            "On Windows you can instead double-click run_live_dashboard.bat "
            "in the project folder.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return stcli


if __name__ == "__main__":
    stcli = _load_streamlit_cli()
    app_path = Path(__file__).with_name("live_dashboard.py").resolve()
    port = os.environ.get("DASHBOARD_PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "localhost",
        "--server.port",
        port,
        "--server.headless",
        "true",
    ]
    raise SystemExit(stcli.main())
