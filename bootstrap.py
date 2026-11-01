#!/usr/bin/env python3
"""Strata for Linux and macOS (Apple Silicon, docs/MACOS.md): the first run installs everything and starts the model;
later runs just start it.  `python3 bootstrap.py [setup.py's options]`, or `make setup` / `make run`.

It needs only a Python 3 to start with: the one macOS gives with Apple's Command Line Tools (/usr/bin/python3, 3.9;
running it the first time offers to install them), or a Linux distribution's.  It makes sure of the rest and then
hands over to .venv's Python:

- macOS: Apple's Command Line Tools (the compiler, git, make); Python 3.10+ from Homebrew when there is none (Homebrew
  itself first if it is missing: its official installer, which says what it does and asks for your password)
- Linux: Python 3.10+ with venv through apt / dnf / pacman when it is missing (sudo asks for your password); on NixOS
  (or with STRATA_UV=1) uv makes the environment
- .venv, Strata's own environment inside this folder (the system's Python stays untouched)

Then `--onboard` runs metal/onboard_mac.py (`make setup` on a Mac: tools, builds, a model, its run config, a first
answer, Strata.app), anything else runs setup.py with the same options.  Written for Python 3.6+: no newer syntax.
"""

import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
MAC = platform.system() == "Darwin"


# Apple's SoC names (the device tree's compatible, "apple,t8103") -> the chip.  Asahi's GPU driver covers M1 and M2
CHIPS = {"t8103": "Apple M1", "t6000": "Apple M1 Pro", "t6001": "Apple M1 Max", "t6002": "Apple M1 Ultra",
         "t8112": "Apple M2", "t6020": "Apple M2 Pro", "t6021": "Apple M2 Max", "t6022": "Apple M2 Ultra",
         "t8122": "Apple M3", "t6030": "Apple M3 Pro", "t6031": "Apple M3 Max", "t6034": "Apple M3 Max",
         "t8132": "Apple M4", "t6040": "Apple M4 Pro", "t6041": "Apple M4 Max"}


def _read(path, binary=False):
    try:
        with open(path, "rb" if binary else "r") as f:
            return f.read()
    except OSError:
        return b"" if binary else ""


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True).stdout.strip()
    except OSError:
        return ""


def apple_silicon():
    """Is this an Apple Silicon Mac, and can Strata reach its GPU?  Each signal is read on the fly:

    - macOS: sysctl hw.optional.arm64 = 1 (true under Rosetta too; sysctl.proc_translated = 1 says this process is an
      Intel one translated, so a native Terminal / Python is needed); the chip from machdep.cpu.brand_string; GPU: Metal
    - Linux: the device tree's compatible ("apple,j293 apple,t8103 ...": a 13" MacBook Pro with the M1), else the
      CPU's implementer in /proc/cpuinfo (0x61 = Apple; a Linux VM on a Mac has it too, but no Apple GPU); GPU: the
      kernel's asahi driver bound to "apple,agx-..." (/sys/class/drm/*/device/uevent), reached through Vulkan

    {"apple", "os", "chip", "soc", "gpu" ("metal" | "asahi" | None), "rosetta", "how"}"""
    if MAC:
        arm = _sysctl("hw.optional.arm64") == "1"
        return {"apple": arm, "os": "macos", "chip": _sysctl("machdep.cpu.brand_string") or None, "soc": None,
                "gpu": "metal" if arm else None, "rosetta": _sysctl("sysctl.proc_translated") == "1",
                "how": "sysctl hw.optional.arm64"}
    info = {"apple": False, "os": platform.system().lower(), "chip": None, "soc": None, "gpu": None, "rosetta": False, "how": ""}
    if platform.system() != "Linux":
        return info
    compat = [c.decode(errors="replace") for c in _read("/proc/device-tree/compatible", True).split(b"\0") if c]
    soc = next((c.split(",", 1)[1] for c in compat if c.startswith("apple,") and c.split(",", 1)[1] in CHIPS), None)
    if any(c.startswith("apple,") for c in compat):
        info.update(apple=True, soc=soc, chip=CHIPS.get(soc or "", "Apple Silicon"), how="device tree " + " ".join(compat))
    elif any(l.split(":")[-1].strip() == "0x61" for l in _read("/proc/cpuinfo").splitlines() if l.startswith("CPU implementer")):
        info.update(apple=True, chip="Apple Silicon (no device tree: a VM?)", how="/proc/cpuinfo CPU implementer 0x61")
    try:
        cards = os.listdir("/sys/class/drm")
    except OSError:
        cards = []
    for c in cards:
        ev = _read("/sys/class/drm/%s/device/uevent" % c)
        if "DRIVER=asahi" in ev and "apple,agx" in ev:
            info["gpu"] = "asahi"
            break
    return info


def say(msg):
    print(msg, flush=True)


def ok_py(exe):
    """A Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (the separate
    python3-venv package), and a venv made without it has no pip."""
    if not exe:
        return False
    test = "import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
    try:
        return subprocess.run([exe, "-c", test], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


def brew_on_path():
    """Homebrew's folder on PATH (a fresh shell after its install may not have it)."""
    for b in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if os.access(b, os.X_OK):
            d = os.path.dirname(b)
            if d not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return b
    return shutil.which("brew")


def command_line_tools():
    """macOS: Apple's Command Line Tools; their installer is a window, so this waits until they are there."""
    if subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return
    say("Installing Apple's Command Line Tools (the compiler): a window opens - click Install, then wait here ...")
    subprocess.run(["xcode-select", "--install"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        time.sleep(5)
    say("Command Line Tools: installed")


def install_python():
    """A Python 3.10+ with venv from the system's package manager (macOS: Homebrew); its path, or None."""
    say("Python 3.10+ with venv is needed; installing it ...")
    if MAC:
        brew = brew_on_path()
        if not brew:
            say("Installing Homebrew (https://brew.sh) for Python: it says what it does and asks for your password ...")
            script = subprocess.run(["curl", "-fsSL", "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"],
                                    stdout=subprocess.PIPE).stdout
            if not script or subprocess.run(["/bin/bash", "-c", script.decode()]).returncode != 0:
                return None
            brew = brew_on_path()
        if not brew or subprocess.run([brew, "install", "python@3.12"]).returncode != 0:
            return None
        prefix = subprocess.run([brew, "--prefix", "python@3.12"], stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        return os.path.join(prefix, "bin", "python3.12")
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    if shutil.which("apt-get"):
        subprocess.run(sudo + ["apt-get", "update"])
        subprocess.run(sudo + ["apt-get", "install", "-y", "python3", "python3-venv", "python3-pip"])
    elif shutil.which("dnf"):
        subprocess.run(sudo + ["dnf", "install", "-y", "python3", "python3-pip"])
    elif shutil.which("pacman"):
        subprocess.run(sudo + ["pacman", "-S", "--noconfirm", "python", "python-pip"])
    return shutil.which("python3")


def make_venv():
    # a .venv from an earlier run that failed half-way has a python but no pip: start it again
    if os.access(VENV_PY, os.X_OK) and subprocess.run([VENV_PY, "-m", "pip", "--version"], stdout=subprocess.DEVNULL,
                                                      stderr=subprocess.DEVNULL).returncode != 0:
        shutil.rmtree(os.path.join(ROOT, ".venv"), ignore_errors=True)
    if os.access(VENV_PY, os.X_OK):
        return
    # uv's own Python only on NixOS (its Python cannot load pip wheels' libraries), or when asked: STRATA_UV=1.
    # Everywhere else the standard venv stays the default (#662: a silent toolchain change was not wanted).
    uv = None
    nixos = False
    try:
        with open("/etc/os-release") as f:
            nixos = any(l.strip().lower() == "id=nixos" for l in f)
    except OSError:
        pass
    if os.environ.get("STRATA_UV") == "1" or nixos:
        uv = shutil.which("uv")
    if uv:
        # an uv-managed interpreter: pip wheels may dlopen libraries outside the Nix store
        if subprocess.run([uv, "venv", "--managed-python", "--seed", "--python", "3.12", ".venv"], cwd=ROOT).returncode != 0:
            sys.exit("Could not create the uv environment in .venv.")
        return
    py = next((p for p in (sys.executable, shutil.which("python3"), shutil.which("python3.12"), shutil.which("python"))
               if ok_py(p)), None)
    if not py:
        py = install_python()
    if not ok_py(py):
        sys.exit("Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv; macOS:\n"
                 "https://www.python.org/downloads/ or brew install python), then run python3 bootstrap.py again.\n"
                 "On NixOS, add uv to environment.systemPackages.")
    # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
    if subprocess.run([py, "-m", "venv", ".venv"], cwd=ROOT).returncode != 0:
        shutil.rmtree(os.path.join(ROOT, ".venv"), ignore_errors=True)
        sys.exit("could not create .venv: sudo apt install python3-venv")


def main():
    os.chdir(ROOT)
    if sys.argv[1:2] == ["--detect"]:          # what this machine is, as setup sees it
        import json
        print(json.dumps(apple_silicon(), indent=1))
        return
    if MAC:
        command_line_tools()
        brew_on_path()
    make_venv()
    args = sys.argv[1:]
    # make setup on a Mac: metal/onboard_mac.py (tools, builds, the model, its run config, a first answer, Strata.app)
    if args[:1] == ["--onboard"]:
        os.execv(VENV_PY, [VENV_PY, os.path.join(ROOT, "metal", "onboard_mac.py")] + args[1:])
    os.execv(VENV_PY, [VENV_PY, os.path.join(ROOT, "setup.py")] + args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
