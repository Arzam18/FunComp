#!/bin/bash
# Writes NAME=VALUE lines (tools/spsa/values.sh, or the "Final values" of an
# SPSA log) into src/params.h as the parameters' defaults: each UCI name is
# mapped to its variable through params::tunables(), and that variable's
# `inline [FUNCOMP_PARAM] int <var> = <value>;` line is rewritten (for an
# array element, `&<var>[k]`, the k-th value of its `<var>[n] = {...};`
# line). Comments are left alone: update the ones that quote values by
# hand. Prints what it changed.
# Usage: tools/spsa/apply-values.sh values.txt [src/params.h]
values=$1
params=${2:-src/params.h}
[ -f "$values" ] || { echo "no values file $values" >&2; exit 1; }
tmp=$(mktemp)
awk '
FNR == NR { if (match($0, /\{"[A-Za-z0-9]+", &[a-z0-9_]+(\[[0-9]+\])?,/)) {
                s = substr($0, RSTART, RLENGTH); gsub(/[{",&]/, " ", s); split(s, a, " ");
                var[a[1]] = a[2] }
            next }
FILENAME == ARGV[2] { line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t\r]+$/, "", line)
            if (line ~ /=/) { split(line, kv, "=")
                              # (tested before var[kv[1]] is read: reading it would create it)
                              if (!(kv[1] in var)) print "unknown parameter " kv[1] > "/dev/stderr"
                              else want[var[kv[1]]] = kv[2] }
            next }
{ if (match($0, /^inline (FUNCOMP_PARAM )?int [a-z0-9_]+ = -?[0-9]+;/)) {
      v = $0; sub(/^inline (FUNCOMP_PARAM )?int /, "", v); sub(/ =.*/, "", v)
      if (v in want) { old = $0; sub(/= -?[0-9]+;/, "= " want[v] ";")
                       if ($0 != old) print v ": " old " -> " want[v] > "/dev/stderr"
                       done[v] = 1 } }
  else if (match($0, /^inline (FUNCOMP_PARAM )?int [a-z0-9_]+\[[0-9]+\] = \{[-0-9, ]+\};/)) {
      v = $0; sub(/^inline (FUNCOMP_PARAM )?int /, "", v); sub(/\[.*/, "", v)
      head = $0; sub(/\{.*/, "", head); body = $0; sub(/^[^{]*\{/, "", body); sub(/\}.*/, "", body)
      n = split(body, el, /, */); changed = 0
      for (k = 0; k < n; k++) if ((v "[" k "]") in want) {
          if (el[k + 1] != want[v "[" k "]"]) { print v "[" k "]: " el[k + 1] " -> " want[v "[" k "]"] > "/dev/stderr"; changed = 1 }
          el[k + 1] = want[v "[" k "]"]; done[v "[" k "]"] = 1 }
      if (changed) { body = el[1]; for (k = 2; k <= n; k++) body = body ", " el[k]; $0 = head "{" body "};" } }
  print }
END { for (v in want) if (!(v in done)) print "not found in params.h: " v > "/dev/stderr" }
' "$params" "$values" "$params" > "$tmp" && mv "$tmp" "$params"
