#!/usr/bin/env bash
DISPLAY="${DISPLAY:-:0}" xset q >/dev/null 2>&1 || exit 1
pgrep x11vnc >/dev/null 2>&1 || exit 1
pgrep websockify >/dev/null 2>&1 || exit 1
