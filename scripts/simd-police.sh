#!/usr/bin/env bash
# simd-police.sh — enforces Isa trait usage
#
# The rule is structural:
#   Implementation files (simd_avx512.rs, simd_avx2.rs, simd_isa.rs, etc.)
#   wrap raw intrinsics. Everyone else goes through the Isa trait.
#
# ALLOWED: implementation files may use raw __m512/__m256/std::simd types
# BANNED:  everything else importing simd_avx512:: directly or using raw types
#
# No baseline file needed. When a crate gets rewired to <I: Isa>, its
# violations disappear automatically.
#
# Usage: simd-police.sh [directory]   (default: current directory)
# Exit 0 = clean. Exit 1 = violations found.

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Implementation files that MAY use raw intrinsics / simd_avx512 types.
# Everything else must go through simd_isa::Isa.
is_impl_file() {
    case "$(basename "$1")" in
        simd_avx512.rs|simd_avx2.rs|simd_isa.rs|simd.rs|simd_compat.rs|\
        bf16_hamming.rs|hdr.rs|prefilter.rs) return 0 ;;
        *) return 1 ;;
    esac
}

SEARCH_DIR="${1:-.}"
VIOLATIONS=0

echo "=== SIMD POLICE: enforcing Isa trait boundary ==="
echo "Scope: $SEARCH_DIR"
echo ""

# Collect candidate files into an array (handles spaces in paths).
CANDIDATES=()
while IFS= read -r f; do
    if ! is_impl_file "$f"; then
        CANDIDATES+=("$f")
    fi
done <<FILELIST
$(find "$SEARCH_DIR" -name '*.rs' -not -path '*/target/*' -not -path '*/.archive-*/*')
FILELIST

# ---------------------------------------------------------------------------
# Rule 1: No direct simd_avx512:: imports outside implementation files
#   Uses sed to strip // comments before matching, avoiding false positives.
# ---------------------------------------------------------------------------
for f in "${CANDIDATES[@]}"; do
    # Note: grep -q would cause SIGPIPE with pipefail, so redirect to /dev/null.
    if sed 's|//.*||' "$f" 2>/dev/null | grep -E 'use.*simd_avx512::' >/dev/null 2>&1; then
        echo -e "${RED}VIOLATION: $f imports simd_avx512 directly. Use simd_isa::Isa instead.${NC}"
        grep -n 'use.*simd_avx512::' "$f" 2>/dev/null | sed 's/^/    /'
        echo ""
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done

# ---------------------------------------------------------------------------
# Rule 2: No raw __m512/__m256 types outside implementation files
# ---------------------------------------------------------------------------
for f in "${CANDIDATES[@]}"; do
    if sed 's|//.*||' "$f" 2>/dev/null | grep -E '__m512|__m256' >/dev/null 2>&1; then
        echo -e "${RED}VIOLATION: $f uses raw SIMD types. Use simd_isa::Isa instead.${NC}"
        grep -n -E '__m512|__m256' "$f" 2>/dev/null | grep -v '^\s*//' | sed 's/^/    /'
        echo ""
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== SIMD POLICE REPORT ==="
echo "Violations: $VIOLATIONS"

if [ "$VIOLATIONS" -eq 0 ]; then
    echo -e "${GREEN}All clear. Isa trait enforced.${NC}"
    exit 0
else
    echo -e "${RED}$VIOLATIONS violation(s) found. Use simd_isa::Isa instead of raw types.${NC}"
    exit 1
fi
