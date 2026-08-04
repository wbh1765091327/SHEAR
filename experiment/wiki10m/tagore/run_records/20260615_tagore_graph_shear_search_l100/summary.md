# Tagore Graph With PiPGANN Search, L=100

## Input

- Tagore graph: `/data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_175248_wiki10m_largeindex32/tagore_run/index/wiki10m.tagore.vamana`
- Adapter `_mem.index`: `/data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_tagore_graph_pipgann_search_l100/index/_mem.index`
- Base: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin`
- Query: `/data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin`
- GT: `/data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin`
- Search binary: `/home/wbh/cppwork/PiPGANN/target/release/search_mem_index`

Adapter check:

```text
/data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_tagore_graph_pipgann_search_l100/index/_mem.index -> /data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_175248_wiki10m_largeindex32/tagore_run/index/wiki10m.tagore.vamana
```

## Command

```bash
target/release/search_mem_index \
  --index-path-prefix /data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_tagore_graph_pipgann_search_l100/index \
  --data-path /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_1024.fbin \
  --query-file /data_nvme_kioxia/tyj/data/wiki-1024-35M/query.fbin \
  --gt-file /data_staging/wbh/wiki10m_gpu_subtree_measurement/data/wiki_10m_gt100.bin \
  --result-path /data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_tagore_graph_pipgann_search_l100/search_l100/tagore_graph_pipgann_search_results.fbin \
  --recall-at 10 \
  --search-list-size 100 \
  --num-threads 8
```

## Result

- Base data: `10000000` points, dim `1024`
- Graph: `10000000` nodes, start `7932297`, max_degree `32`
- Queries: `10000`
- L: `100`
- Recall@10: `0.8092`
- Search loop time: `4.68s`
- QPS: `2134.7`
- Process wall time: `0:53.67`
- Max RSS KiB: `80000512`
- File system inputs: `18931384`
- Exit status: `0`
- Result file: `/data_staging/wbh/tagore_wiki10m_measurement/runs/20260615_tagore_graph_pipgann_search_l100/search_l100/tagore_graph_pipgann_search_results.fbin`

Note: `search_stdout.log` is empty because this `search_mem_index` build wrote its application log to stderr, alongside `/usr/bin/time -v` output in `search_stderr_time.log`.

## Baseline For Comparison

Tagore graph + Tagore search original result:

- L: `100`
- Recall@10: `0.8093`
- QPS: `4603.46`
- Mean latency: `1736.30 us`
- 99.9 latency: `4404.91 us`
- Avg dist cmps: `3001.95`
- Process wall time: `49.92s`

PiPGANN graph + PiPGANN search reference result:

- Backend: `persistent-tiled-gemm-fp32`
- L: `100`
- Recall@10: `0.8381`
- QPS: `2492.7`
- Search loop time: `4.01s`
- Graph avg out-degree: `21.551250`

## Interpretation

1. Tagore graph under PiPGANN search gives `Recall@10 = 0.8092` at `L=100`.
2. This is effectively identical to Tagore graph + Tagore search (`0.8093`), with only `-0.0001` absolute difference.
3. Compared with PiPGANN graph + PiPGANN search (`0.8381`), Tagore graph is lower by `0.0289` absolute recall.
4. Because both graphs are evaluated through the same PiPGANN search path at `L=100`, this supports the conclusion that the PiPGANN graph has better navigability for this benchmark point.
5. The result does not support attributing the earlier recall gap mainly to search implementation differences. The Tagore graph remains around `0.809x` when searched by PiPGANN.
