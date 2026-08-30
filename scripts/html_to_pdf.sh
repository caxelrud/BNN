#!/usr/bin/env bash
# Converts each statically-exported Pluto notebook (html/*.html + the
# vendored html/frontend-dist/, produced by scripts/export_pdfs.jl and
# scripts/vendor_frontend_dist.sh) into a PDF under pdfs/.
#
# Pluto's static export loads its viewer as ES module <script> tags, which
# browsers refuse to run over file:// (CORS). So this serves html/ over a
# throwaway local HTTP server and drives headless Chromium with Playwright
# (waiting for Pluto's own "loading" class to clear -- a plain
# `chromium --print-to-pdf` fires on window.onload, long before Pluto has
# finished parsing/rendering the baked cell outputs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_DIR="$ROOT/html"
PDF_DIR="$ROOT/pdfs"
PORT="${PLUTO_EXPORT_HTTP_PORT:-8791}"

export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/opt/pw-browsers}"
export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
export NODE_PATH="${NODE_PATH:-$(npm root -g)}"

if [ ! -d "$HTML_DIR/frontend-dist" ]; then
  "$(dirname "${BASH_SOURCE[0]}")/vendor_frontend_dist.sh"
fi

mkdir -p "$PDF_DIR"

python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$HTML_DIR" > /tmp/pluto_export_httpd.log 2>&1 &
HTTPD_PID=$!
trap 'kill "$HTTPD_PID" 2>/dev/null || true' EXIT
sleep 1

node "$(dirname "${BASH_SOURCE[0]}")/html_to_pdf.js" "$HTML_DIR" "$PDF_DIR" "http://127.0.0.1:$PORT"

echo "PDFs written to $PDF_DIR"
ls -la "$PDF_DIR"
