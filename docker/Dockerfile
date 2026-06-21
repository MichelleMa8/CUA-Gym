FROM ubuntu:22.04

ARG TARGETARCH
ARG VSCODE_VERSION=1.117.0-1776814346
ARG LIBREOFFICE_VERSION=1:7.3.7-0ubuntu0.22.04.10

ENV VSCODE_DEB_URL=https://packages.microsoft.com/repos/code/pool/main/c/code/code_${VSCODE_VERSION}_amd64.deb

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0
ENV HOME=/home/user
ENV USER=user
ENV SHELL=/bin/bash
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV XDG_RUNTIME_DIR=/tmp/runtime-user

RUN test "${TARGETARCH}" = "amd64" || \
    (echo "This image supports only linux/amd64. Build with --platform linux/amd64." >&2 && exit 1)

# ── Bootstrap (ca-certs + curl + gnupg) ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg && \
    rm -rf /var/lib/apt/lists/*

# ── Core desktop + apps ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    dbus-x11 \
    evince \
    ffmpeg \
    gedit \
    gimp \
    git \
    gnome-terminal \
    gnupg \
    ghostscript \
    imagemagick \
    libatk-bridge2.0-0 \
    libdbus-glib-1-2 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libglib2.0-bin \
    libnss3 \
    libsecret-1-0 \
    libreoffice=${LIBREOFFICE_VERSION} \
    locales \
    nautilus \
    novnc \
    pcmanfm \
    playerctl \
    poppler-utils \
    procps \
    pulseaudio \
    pulseaudio-utils \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-tk \
    python3-venv \
    scrot \
    software-properties-common \
    sudo \
    tar \
    tesseract-ocr \
    tesseract-ocr-eng \
    unzip \
    vlc \
    websockify \
    wget \
    x11-utils \
    x11vnc \
    xauth \
    xclip \
    xdotool \
    xfce4 \
    xfce4-terminal \
    xfonts-base \
    xfonts-scalable \
    xvfb \
    xz-utils && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8

# ── Node.js (needed by some multi_apps tasks) ─────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# ── User account ──────────────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash user && \
    usermod -aG sudo user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/user && \
    chmod 0440 /etc/sudoers.d/user && \
    mkdir -p /home/user/Downloads /tmp/runtime-user && \
    chown -R user:user /home/user /tmp/runtime-user

# ── Python packages ───────────────────────────────────────────────────────────
# Pre-install all packages that task setup scripts commonly pip-install at
# runtime (pymupdf, pikepdf, fpdf2, etc.) to avoid per-task download delays.
RUN pip3 install --no-cache-dir \
    fpdf2 \
    lxml \
    marionette_driver \
    numpy \
    odfpy \
    openpyxl \
    pandas \
    pillow \
    pikepdf \
    pyautogui \
    pymupdf \
    pyperclip \
    pytesseract \
    python-docx \
    python-pptx \
    python-xlib \
    reportlab \
    requests

# ── VS Code ───────────────────────────────────────────────────────────────────
RUN wget -qO /tmp/code.deb "${VSCODE_DEB_URL}" && \
    apt-get update && \
    apt-get install -y /tmp/code.deb && \
    rm -f /tmp/code.deb && \
    rm -rf /var/lib/apt/lists/*

# ── Google Chrome ─────────────────────────────────────────────────────────────
RUN wget -qO /tmp/google-chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get update && \
    apt-get install -y /tmp/google-chrome.deb && \
    rm -f /tmp/google-chrome.deb && \
    rm -rf /var/lib/apt/lists/*

# ── Firefox ESR ───────────────────────────────────────────────────────────────
RUN wget -qO /tmp/firefox-esr.tar.xz \
      "https://download.mozilla.org/?product=firefox-esr-latest-ssl&os=linux64&lang=en-US" && \
    tar -xJf /tmp/firefox-esr.tar.xz -C /opt/ && \
    rm -f /tmp/firefox-esr.tar.xz && \
    mv /opt/firefox /opt/firefox-esr && \
    ln -sf /opt/firefox-esr/firefox /usr/local/bin/firefox-bin && \
    printf '%s\n' \
      '[Desktop Entry]' \
      'Name=Firefox ESR' \
      'Type=Application' \
      'Exec=/opt/firefox-esr/firefox %u' \
      'Icon=/opt/firefox-esr/browser/chrome/icons/default/default128.png' \
      'Categories=Network;WebBrowser;' \
      > /usr/share/applications/firefox-esr.desktop

# ── Keystore bypass for Electron/Chromium apps ────────────────────────────────
RUN for app in google-chrome code; do \
      desktop_file=$(find /usr/share/applications -name "*${app}*.desktop" -print -quit 2>/dev/null); \
      if [ -n "$desktop_file" ] && ! grep -q -- '--password-store=basic' "$desktop_file"; then \
        sed -i 's|Exec=\(.*\)|Exec=\1 --password-store=basic|' "$desktop_file"; \
      fi; \
    done

# ── Virtual audio sink (for vlc tasks) ───────────────────────────────────────
RUN grep -q 'module-null-sink sink_name=virtual_speaker' /etc/pulse/default.pa || \
    printf '\nload-module module-null-sink sink_name=virtual_speaker sink_properties=device.description=Virtual_Speaker\nset-default-sink virtual_speaker\n' \
      >> /etc/pulse/default.pa

# ── Entrypoint scripts ────────────────────────────────────────────────────────
COPY docker/entrypoint.sh  /usr/local/bin/gui-synth-entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/gui-synth-healthcheck.sh
RUN chmod +x \
    /usr/local/bin/gui-synth-entrypoint.sh \
    /usr/local/bin/gui-synth-healthcheck.sh

USER user
WORKDIR /home/user

EXPOSE 6080
EXPOSE 5900

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD ["/usr/local/bin/gui-synth-healthcheck.sh"]

ENTRYPOINT ["/usr/local/bin/gui-synth-entrypoint.sh"]
