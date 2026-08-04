# LAION100M HashPrune policy figure provenance

The Plain, DistanceFirst-Q3, and HardTier-Q3 series come from:

`/data_staging/wbh/hashprune_four_policy_cap3/laion100m/four-policy-cap3-20260725T103730/`

SoftTier-Q3 is the per-\(L\) best-observed envelope across the original run and:

`/data_staging/wbh/hashprune_softtier_rerun_bestof3/laion100m/20260725T073524Z/`

For each \(L\), the Recall--QPS pair is retained from the same measured index
and search invocation. The rerun value replaces the original value only when
its best-of-three QPS is higher. The original values remain at \(L=80\) and
\(L=100\); all other SoftTier-Q3 points use the rerun.
