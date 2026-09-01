#!/usr/bin/env bash
# Copies Pluto's built frontend assets (JS/CSS/fonts) next to the static
# HTML export and rewrites the export's CDN links to point at the local
# copy, so PDF rendering works without outbound network access. Pluto's
# static export otherwise references https://cdn.jsdelivr.net/... for its
# own viewer JS -- fine for web hosting, unusable in a network-restricted
# render step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_DIR="$ROOT/html"

PLUTO_PKG_DIR="$(julia --project="$ROOT" -e 'using Pluto; print(pkgdir(Pluto))' 2>/dev/null)"
SRC="$PLUTO_PKG_DIR/frontend-dist"
PLUTO_VERSION="$(julia --project="$ROOT" -e 'using Pluto, Pkg; print(Pkg.dependencies()[Base.UUID("c3e4b0f8-55cb-11ea-2926-15256bba5781")].version)' 2>/dev/null)"

if [ ! -d "$SRC" ]; then
  echo "error: $SRC not found -- Pluto not installed in this project?" >&2
  exit 1
fi

rm -rf "$HTML_DIR/frontend-dist"
cp -r "$SRC" "$HTML_DIR/frontend-dist"

for f in "$HTML_DIR"/*.html; do
  sed -i.bak "s#https://cdn.jsdelivr.net/gh/JuliaPluto/Pluto.jl@${PLUTO_VERSION}/frontend-dist/#frontend-dist/#g" "$f"
  rm -f "$f.bak"
done

echo "Vendored $SRC -> $HTML_DIR/frontend-dist (Pluto v$PLUTO_VERSION)"
