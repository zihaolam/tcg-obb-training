#!/usr/bin/env bash
# Flatten data/sources/cards/cards/{pokemon-name}/image.webp into data/sources/cards/.
# Renames files as {pokemon-name}__{original-name} to avoid collisions across folders.
set -euo pipefail

ROOT="${1:-data/sources/cards}"
NESTED="${ROOT}/cards"

if [ ! -d "$NESTED" ]; then
    echo "error: '${NESTED}' not found — nothing to flatten" >&2
    exit 1
fi

moved=0
collisions=0
while IFS= read -r -d '' file; do
    rel="${file#${NESTED}/}"               # pokemon-name/image.webp
    flat="${rel//\//__}"                   # pokemon-name__image.webp
    dest="${ROOT}/${flat}"
    if [ -e "$dest" ]; then
        collisions=$((collisions + 1))
        dest="${ROOT}/${flat%.*}__$RANDOM.${flat##*.}"
    fi
    mv "$file" "$dest"
    moved=$((moved + 1))
done < <(find "$NESTED" -type f -print0)

# Clean up the now-empty nested tree
find "$NESTED" -depth -type d -empty -delete

echo "moved ${moved} files into ${ROOT}/ (${collisions} renamed to avoid collision)"
echo "verify:"
echo "  ls ${ROOT} | head"
echo "  find ${ROOT} -type f | wc -l"
