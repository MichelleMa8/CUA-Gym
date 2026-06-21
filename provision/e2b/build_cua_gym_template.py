"""
Builds an e2b desktop sandbox template for CUA-Gym.

Mirrors the CUA-Gym Docker image (provision/docker/Dockerfile) as an e2b template so
that run_trajectories.py --env-backend e2b uses the same app set as Docker.

Apps installed (matching select_config.json):
  gimp, libreoffice (Calc/Impress/Writer), vlc, vscode, evince (pdf)

Plus supporting tools that task scripts depend on:
  ffmpeg, imagemagick, ghostscript, poppler-utils, tesseract-ocr, nodejs,
  google-chrome, firefox-esr

Python libraries pre-installed to avoid per-task pip delays:
  pymupdf, pikepdf, fpdf2, reportlab, pyautogui, python-docx, openpyxl, ...

Usage:
    cd /path/to/CUA-Gym
    pip install e2b           # build API (separate from e2b-desktop runtime)
    E2B_API_KEY=e2b_... python provision/e2b/build_cua_gym_template.py
"""

from dotenv import load_dotenv
from e2b import Template, default_build_logger

load_dotenv()

LIBREOFFICE_VERSION = "1:7.3.7-0ubuntu0.22.04.10"
VSCODE_VERSION = "1.117.0-1776814346"
VSCODE_DEB_URL = (
    "https://packages.microsoft.com/repos/code/pool/main/c/code/"
    f"code_{VSCODE_VERSION}_amd64.deb"
)

template = (
    Template()
    .from_template("desktop")   # e2b base: Ubuntu 22.04 + XFCE + scrot + Xvfb
    .set_user("root")

    # ── 1. Core desktop apps + system tools ──────────────────────────────────
    .run_cmd(["apt-get update"])
    .apt_install([
        # PDF viewer (evince) — primary GUI app for pdf tasks
        "evince",
        # GIMP
        "gimp",
        # LibreOffice suite (Calc, Impress, Writer)
        f"libreoffice={LIBREOFFICE_VERSION}",
        # VLC
        "vlc",
        # PDF CLI tools
        "ghostscript",       # gs — batch PDF manipulation
        "poppler-utils",     # pdftotext, pdfinfo, pdftoppm
        "pdftk",             # pdftk — merge/split/stamp PDFs (21 tasks)
        "tesseract-ocr",     # OCR engine
        "tesseract-ocr-eng", # English language data
        # Media (multi_apps tasks: ffmpeg-based video/audio processing)
        "ffmpeg",
        "imagemagick",
        # Browsers (multi_apps web-lookup tasks)
        "firefox-esr",
        # Text editors / utilities
        "gedit",
        "git",
        "gnome-terminal",
        # Node.js prerequisites (nodesource installed separately below)
        "curl",
        # Python build deps
        "python3-tk",
        "python3-dev",
        # Accessibility / xdotool (agent interaction)
        "gir1.2-atspi-2.0",
        "at-spi2-core",
        "xdotool",
        "xclip",
        # Audio (VLC needs a working audio sink in the container)
        "pulseaudio",
        "pulseaudio-utils",
        # Misc runtime deps
        "dbus-x11",
        "libsecret-1-0",
        "gnupg",
        "wget",
        "unzip",
        "xz-utils",
        "nautilus",
        "pcmanfm",
        "playerctl",
        "procps",
    ])

    # ── 2. PulseAudio virtual sink (VLC needs an audio output device) ─────────
    .run_cmd([
        "grep -q 'module-null-sink sink_name=virtual_speaker' /etc/pulse/default.pa "
        "|| printf '\\nload-module module-null-sink sink_name=virtual_speaker "
        "sink_properties=device.description=Virtual_Speaker\\n"
        "set-default-sink virtual_speaker\\n' >> /etc/pulse/default.pa",
        "mkdir -p /etc/xdg/autostart",
        "cat > /etc/xdg/autostart/pulseaudio.desktop << 'DESKTOP'\n"
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=PulseAudio Sound System\n"
        "Exec=pulseaudio --start --exit-idle-time=-1\n"
        "NoDisplay=true\n"
        "X-GNOME-Autostart-enabled=true\n"
        "DESKTOP",
        "mkdir -p /home/user/.config/pulse",
        "chown -R user:user /home/user/.config",
    ])

    # ── 3. Node.js 20.x (multi_apps tasks that call node/npm) ─────────────────
    .run_cmd([
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "rm -rf /var/lib/apt/lists/*",
    ])

    # ── 4. VS Code .deb ────────────────────────────────────────────────────────
    .run_cmd([
        f"wget -q -O /tmp/code.deb '{VSCODE_DEB_URL}'",
        "apt-get install -y /tmp/code.deb",
        "rm /tmp/code.deb",
    ])

    # ── 5. Google Chrome ───────────────────────────────────────────────────────
    .run_cmd([
        "wget -q -O /tmp/google-chrome.deb "
        "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb",
        "apt-get install -y /tmp/google-chrome.deb",
        "rm /tmp/google-chrome.deb",
        "rm -rf /var/lib/apt/lists/*",
    ])

    # ── 6. Firefox ESR tarball ─────────────────────────────────────────────────
    # Ubuntu 22.04 ships Firefox as a snap by default; the tarball gives us
    # a regular binary that works inside containers without snap infrastructure.
    .run_cmd([
        "wget -q -O /tmp/firefox-esr.tar.xz "
        "'https://download.mozilla.org/?product=firefox-esr-latest-ssl&os=linux64&lang=en-US'",
        "tar -xJf /tmp/firefox-esr.tar.xz -C /opt/",
        "rm /tmp/firefox-esr.tar.xz",
        "mv /opt/firefox /opt/firefox-esr",
        "ln -sf /opt/firefox-esr/firefox /usr/local/bin/firefox-bin",
        "printf '%s\\n' "
        "'[Desktop Entry]' "
        "'Name=Firefox ESR' "
        "'Type=Application' "
        "'Exec=/opt/firefox-esr/firefox %u' "
        "'Icon=/opt/firefox-esr/browser/chrome/icons/default/default128.png' "
        "'Categories=Network;WebBrowser;' "
        "> /usr/share/applications/firefox-esr.desktop",
    ])

    # ── 7. Python packages (mirrors provision/docker/Dockerfile pip3 install block) ──────
    # Pre-installed so that task setup/reward scripts never need to pip-install
    # at run time (slow and flaky inside the sandbox).
    .run_cmd([
        "pip3 install --no-cache-dir "
        "fpdf2 "
        "lxml "
        "marionette_driver "
        "numpy "
        "odfpy "
        "openpyxl "
        "pandas "
        "pillow "
        "pikepdf "
        "pyautogui "
        "pymupdf "
        "pyperclip "
        "pytesseract "
        "python-docx "
        "python-pptx "
        "python-xlib "
        "reportlab "
        "requests",
    ])

    # ── 8. Disable keyring prompt for Electron apps (VS Code, Chrome) ─────────
    .run_cmd([
        "apt-get install -y -qq gnome-keyring",
        "for app in google-chrome code; do "
        "  desktop_file=$(find /usr/share/applications -name \"*${app}*.desktop\" -print -quit 2>/dev/null); "
        "  if [ -n \"$desktop_file\" ] && ! grep -q -- '--password-store=basic' \"$desktop_file\"; then "
        "    sed -i 's|Exec=\\(.*\\)|Exec=\\1 --password-store=basic|' \"$desktop_file\"; "
        "  fi; "
        "done",
    ])

    .set_user("user")
    .set_workdir("/home/user")
)

Template.build(
    template,
    alias="cua-gym-desktop",
    cpu_count=2,
    memory_mb=2048,
    on_build_logs=default_build_logger(),
)

print("Template 'cua-gym-desktop' built successfully.")
print("Update E2B_ENV_TEMPLATE=cua-gym-desktop in your .env file.")
