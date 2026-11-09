#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local_env.sh"
mkdir -p "$SHEAR_OUTPUT_ROOT"
smoke_dir=$(mktemp -d "$SHEAR_OUTPUT_ROOT/local-smoke-XXXXXX")
export SHEAR_SMOKE_DIR="$smoke_dir"
python3 - <<'PY'
import os, random, struct
from pathlib import Path
p=Path(os.environ['SHEAR_SMOKE_DIR'])
r=random.Random(42)
base=[struct.unpack('<8f',struct.pack('<8f',*[r.random() for _ in range(8)])) for _ in range(96)]
for name,rows in [('base.fbin',base),('query.fbin',base[:8])]:
 with (p/name).open('wb') as f:
  f.write(struct.pack('<II',len(rows),8))
  for row in rows:f.write(struct.pack('<8f',*row))
with (p/'gt10.bin').open('wb') as f:
 f.write(struct.pack('<II',8,10))
 for q in base[:8]:
  ids=sorted(range(96),key=lambda i:sum((a-b)**2 for a,b in zip(q,base[i])))[:10]
  f.write(struct.pack('<10I',*ids))
PY
SHEAR_PIPNN_PROFILE=1 "$SHEAR_ROOT/target/release/build_shear_index" \
  --data-type float --dist-fn l2 --data-path "$smoke_dir/base.fbin" \
  --index-path-prefix "$smoke_dir/index" --oom-enable --oom-io-mode buffered \
  --oom-temp-dir "$smoke_dir/tmp" --oom-profile-json "$smoke_dir/profile.json" \
  --partitioner gpu-subtree --pipnn-backend cuda --pipnn-gpu-device 0 \
  --pipnn-gpu-workspace-budget-gb 0.125 --leaf-size-min 4 --leaf-size-max 16 \
  --max-leaders 32 --psamp-fraction 1 --fanout-top 10 --fanout-second 3 \
  --leaf-knn 4 --max-degree 8 --m-hash-bits 4 --num-threads 4 > "$smoke_dir/build.log" 2>&1
python3 - <<'PY'
import json,os,struct
from pathlib import Path
p=Path(os.environ['SHEAR_SMOKE_DIR'])
g=json.loads((p/'profile.json').read_text())['gpu_subtree']
assert g['enabled'] and g['persistent_success_count'] > 0,g
assert g['persistent_queue_overflow_flags']==0,g
b=(p/'index/_mem.index').read_bytes()
size,degree,start,frozen=struct.unpack_from('<QIIQ',b)
assert size==len(b) and start<96 and frozen==0
pos=24
for i in range(96):
 n=struct.unpack_from('<I',b,pos)[0];pos+=4
 ids=struct.unpack_from(f'<{n}I',b,pos);pos+=4*n
 assert 0<n<=8 and all(v<96 and v!=i for v in ids),(i,ids)
assert pos==len(b)
print('GPU execution and 96-node graph validated.')
PY
"$SHEAR_ROOT/target/release/search_shear_index" --index-path-prefix "$smoke_dir/index" \
  --data-path "$smoke_dir/base.fbin" --query-file "$smoke_dir/query.fbin" \
  --gt-file "$smoke_dir/gt10.bin" --result-path "$smoke_dir/search.bin" \
  --recall-at 10 --search-list-size 96 --num-threads 1 --search-backend exact \
  --storage mmap > "$smoke_dir/search.log" 2>&1
cat "$smoke_dir/search.log"
printf 'Smoke artifacts: %s\n' "$smoke_dir"
