#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} was not found." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Error: Python 3.10 or newer is required." >&2
  exit 1
fi

if [[ -d "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Removing incomplete virtual environment at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
      echo >&2
      echo "Error: Python could not create a virtual environment." >&2
      echo "On Ubuntu, install the matching venv package, then rerun this script:" >&2
      echo "  sudo apt update" >&2
      echo "  sudo apt install -y python3-venv" >&2
      exit 1
    fi
  fi
else
  echo "Reusing virtual environment at ${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "The venv module did not bundle pip; bootstrapping it with the host pip."
  if ! "${PYTHON_BIN}" -m pip --python "${VENV_DIR}/bin/python" install --upgrade pip; then
    echo >&2
    echo "Error: pip could not be installed in the virtual environment." >&2
    echo "Install python3-venv with apt, remove .venv, then rerun this script." >&2
    exit 1
  fi
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo
echo "Environment ready. Activate it with:"
echo "  source .venv/bin/activate"
echo "Then verify it with:"
echo "  python scripts/check_env.py"
