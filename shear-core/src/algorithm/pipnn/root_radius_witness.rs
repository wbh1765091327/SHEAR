use rand::Rng;
use rand::seq::index;
use sha2::{Digest, Sha256};

use super::params::PiPNNParams;

#[derive(Clone, Debug, PartialEq)]
pub struct SmallKMin {
    k: usize,
    entries: Vec<(f32, u32)>,
}

impl SmallKMin {
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, distance: f32, point_id: u32) {
        if !distance.is_finite() {
            return;
        }
        if let Some(pos) = self
            .entries
            .iter()
            .position(|&(_, existing_point_id)| existing_point_id == point_id)
        {
            self.entries.remove(pos);
        }
        self.entries.push((distance.max(0.0), point_id));
        self.sort_and_truncate();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn kth_radius(&self) -> Option<f32> {
        if self.entries.len() < self.k {
            return None;
        }
        self.entries.get(self.k - 1).map(|&(distance, _)| distance)
    }

    pub fn entries(&self) -> Vec<(f32, u32)> {
        self.entries.clone()
    }

    fn sort_and_truncate(&mut self) {
        self.entries.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        self.entries.truncate(self.k);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellRadiusReservoir {
    radii: SmallKMin,
}

impl CellRadiusReservoir {
    pub fn new(k: usize) -> Self {
        Self {
            radii: SmallKMin::new(k),
        }
    }

    pub fn insert(&mut self, point_id: u32, owner_distance: f32) {
        self.radii.insert(owner_distance, point_id);
    }

    pub fn kth_radius(&self) -> Option<f32> {
        self.radii.kth_radius()
    }

    pub fn len(&self) -> usize {
        self.radii.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RootRadiusWitnessStore {
    cells: Vec<CellRadiusReservoir>,
    k: usize,
}

impl RootRadiusWitnessStore {
    pub fn new(cell_count: usize, k: usize) -> Self {
        let k = k.max(1);
        Self {
            cells: (0..cell_count)
                .map(|_| CellRadiusReservoir::new(k))
                .collect(),
            k,
        }
    }

    pub fn insert(&mut self, owner: usize, point_id: u32, owner_distance: f32) {
        if let Some(cell) = self.cells.get_mut(owner) {
            cell.insert(point_id, owner_distance);
        }
    }

    pub fn insert_batch_sorted<I>(&mut self, batch: I)
    where
        I: IntoIterator<Item = (usize, f32, u32)>,
    {
        let mut batch: Vec<_> = batch.into_iter().collect();
        batch.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        for (owner, owner_distance, point_id) in batch {
            self.insert(owner, point_id, owner_distance);
        }
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn snapshot(&self) -> RadiusWitnessSnapshot {
        let radii: Vec<_> = self
            .cells
            .iter()
            .map(CellRadiusReservoir::kth_radius)
            .collect();
        RadiusWitnessSnapshot::from_radii(radii)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RadiusWitnessSnapshot {
    radii: Vec<Option<f32>>,
    ready_cells: usize,
}

impl RadiusWitnessSnapshot {
    pub fn from_radii(radii: Vec<Option<f32>>) -> Self {
        let ready_cells = radii.iter().filter(|radius| radius.is_some()).count();
        Self { radii, ready_cells }
    }

    pub fn kth_radius(&self, cell: usize) -> Option<f32> {
        self.radii.get(cell).copied().flatten()
    }

    pub fn ready_cells(&self) -> usize {
        self.ready_cells
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadiusWitnessConfig {
    pub k: usize,
    pub fixed_fanout: usize,
    pub candidate_rs: Vec<usize>,
    pub warmup_points: usize,
    pub dry_run: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadiusWitnessDecision {
    pub kept: usize,
    pub certified: bool,
    pub fallback: bool,
    pub warmup: bool,
    pub insufficient_radius: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadiusWitnessTelemetry {
    pub k: usize,
    pub fixed_fanout: usize,
    pub candidate_rs: Vec<usize>,
    pub warmup_points: usize,
    pub total_points: usize,
    pub warmup_points_seen: usize,
    pub certified_points: usize,
    pub fallback_points: usize,
    pub insufficient_radius_points: usize,
    pub kept_hist: Vec<usize>,
    pub selector_ms: u128,
}

impl RadiusWitnessTelemetry {
    pub fn new(config: &RadiusWitnessConfig, total_points: usize) -> Self {
        Self {
            k: config.k,
            fixed_fanout: config.fixed_fanout,
            candidate_rs: config.candidate_rs.clone(),
            warmup_points: config.warmup_points,
            total_points,
            kept_hist: vec![0; config.fixed_fanout.max(1) + 1],
            ..Self::default()
        }
    }

    pub fn record(&mut self, decision: &RadiusWitnessDecision) {
        if decision.warmup {
            self.warmup_points_seen += 1;
        } else if decision.certified {
            self.certified_points += 1;
        } else {
            self.fallback_points += 1;
        }
        if decision.insufficient_radius {
            self.insufficient_radius_points += 1;
        }
        if decision.kept >= self.kept_hist.len() {
            self.kept_hist.resize(decision.kept + 1, 0);
        }
        self.kept_hist[decision.kept] += 1;
    }
}

pub fn select_radius_certified_fanout(
    config: &RadiusWitnessConfig,
    snapshot: &RadiusWitnessSnapshot,
    top_leaders: &[usize],
    top_distances_sq: &[f32],
    points_seen: usize,
) -> RadiusWitnessDecision {
    let fixed_fanout = config
        .fixed_fanout
        .max(1)
        .min(32)
        .min(top_leaders.len().max(1));
    if top_leaders.is_empty() || top_distances_sq.is_empty() {
        return RadiusWitnessDecision {
            kept: fixed_fanout,
            fallback: true,
            ..RadiusWitnessDecision::default()
        };
    }
    if points_seen < config.warmup_points {
        return RadiusWitnessDecision {
            kept: fixed_fanout,
            fallback: true,
            warmup: true,
            ..RadiusWitnessDecision::default()
        };
    }

    let nearest = top_distances_sq[0].max(0.0).sqrt();
    let mut insufficient_radius = false;
    for &r in &config.candidate_rs {
        if r == 0 || r >= fixed_fanout || r >= top_leaders.len() || r >= top_distances_sq.len() {
            continue;
        }

        let mut rho_hat = f32::INFINITY;
        for j in 0..r {
            let leader = top_leaders[j];
            let Some(radius) = snapshot.kth_radius(leader) else {
                continue;
            };
            let distance = top_distances_sq[j].max(0.0).sqrt();
            rho_hat = rho_hat.min(distance + radius);
        }
        if !rho_hat.is_finite() {
            insufficient_radius = true;
            continue;
        }

        let boundary = top_distances_sq[r].max(0.0).sqrt();
        let gap = (boundary - nearest).max(0.0);
        if gap > 2.0 * rho_hat {
            return RadiusWitnessDecision {
                kept: r,
                certified: true,
                ..RadiusWitnessDecision::default()
            };
        }
    }

    RadiusWitnessDecision {
        kept: fixed_fanout,
        fallback: true,
        insufficient_radius,
        ..RadiusWitnessDecision::default()
    }
}

pub fn select_root_leaders(
    indices: &[u32],
    params: &PiPNNParams,
    rng: &mut (impl Rng + Send),
) -> Vec<u32> {
    select_root_leaders_with_target(indices, params, None, rng)
}

pub fn select_root_leaders_with_target(
    indices: &[u32],
    params: &PiPNNParams,
    target_root_leaders: Option<usize>,
    rng: &mut (impl Rng + Send),
) -> Vec<u32> {
    select_root_leaders_with_target_impl(indices, params, target_root_leaders, true, rng)
}

pub fn select_root_leaders_with_uncapped_target(
    indices: &[u32],
    params: &PiPNNParams,
    target_root_leaders: usize,
    rng: &mut (impl Rng + Send),
) -> Vec<u32> {
    select_root_leaders_with_target_impl(indices, params, Some(target_root_leaders), false, rng)
}

fn select_root_leaders_with_target_impl(
    indices: &[u32],
    params: &PiPNNParams,
    target_root_leaders: Option<usize>,
    cap_by_max_leaders: bool,
    rng: &mut (impl Rng + Send),
) -> Vec<u32> {
    let total_size = indices.len();
    if total_size == 0 {
        return Vec::new();
    }
    let max_leaders = if cap_by_max_leaders {
        params.max_leaders.min(total_size.max(1))
    } else {
        total_size.max(1)
    };
    let sampled = ((params.adaptive_psamp_fraction(total_size, 0) * total_size as f64).round()
        as usize)
        .max(2);
    let mut num_root_leaders = target_root_leaders.unwrap_or(sampled).max(1);
    num_root_leaders = num_root_leaders.min(max_leaders).min(total_size.max(1));
    if num_root_leaders >= indices.len() {
        indices.to_vec()
    } else if target_root_leaders.is_some() {
        let sample_len = indices
            .len()
            .min(num_root_leaders.saturating_mul(16).max(num_root_leaders));
        let sample_pos = index::sample(rng, indices.len(), sample_len);
        let mut sample = sample_pos
            .iter()
            .map(|pos| indices[pos])
            .collect::<Vec<_>>();
        sample.sort_unstable();
        select_spread_root_leaders_from_sorted_sample(&sample, num_root_leaders)
    } else {
        let leader_pos = index::sample(rng, indices.len(), num_root_leaders);
        leader_pos.iter().map(|pos| indices[pos]).collect()
    }
}

pub fn select_spread_root_leaders_from_sorted_sample(sample: &[u32], target: usize) -> Vec<u32> {
    if sample.is_empty() || target == 0 {
        return Vec::new();
    }
    if target >= sample.len() {
        return sample.to_vec();
    }
    let mut leaders = Vec::with_capacity(target);
    for i in 0..target {
        let numerator = i.saturating_mul(sample.len().saturating_sub(1));
        let idx = (numerator + (target.saturating_sub(1) / 2)) / target.saturating_sub(1).max(1);
        leaders.push(sample[idx]);
    }
    leaders.dedup();
    let mut cursor = 0usize;
    while leaders.len() < target && cursor < sample.len() {
        let candidate = sample[cursor];
        if !leaders.contains(&candidate) {
            leaders.push(candidate);
        }
        cursor += 1;
    }
    leaders
}

pub fn root_leader_fingerprint(leaders: &[u32]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update((leaders.len() as u64).to_le_bytes());
    for &leader in leaders {
        hasher.update(leader.to_le_bytes());
    }
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[0..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::{
        RadiusWitnessConfig, RadiusWitnessSnapshot, RootRadiusWitnessStore, SmallKMin,
        select_radius_certified_fanout,
    };
    use crate::algorithm::pipnn::params::PiPNNParams;

    #[test]
    fn root_radius_witness_k_smallest_insertions_are_order_independent() {
        let samples = [(3.0, 30), (1.0, 20), (1.0, 10), (2.0, 40), (0.5, 50)];
        let mut forward = SmallKMin::new(3);
        let mut reverse = SmallKMin::new(3);

        for &(distance, point_id) in &samples {
            forward.insert(distance, point_id);
        }
        for &(distance, point_id) in samples.iter().rev() {
            reverse.insert(distance, point_id);
        }

        assert_eq!(forward.entries(), reverse.entries());
        assert_eq!(forward.entries(), vec![(0.5, 50), (1.0, 10), (1.0, 20)]);
    }

    #[test]
    fn root_radius_witness_snapshot_counts_ready_cells() {
        let mut store = RootRadiusWitnessStore::new(3, 2);
        store.insert(0, 10, 1.0);
        store.insert(1, 20, 2.0);
        store.insert(1, 21, 3.0);

        let snapshot = store.snapshot();

        assert_eq!(snapshot.ready_cells(), 1);
        assert_eq!(snapshot.kth_radius(1), Some(3.0));
    }

    #[test]
    fn root_radius_witness_selector_uses_sqrt_distances() {
        let config = RadiusWitnessConfig {
            k: 1,
            fixed_fanout: 3,
            candidate_rs: vec![1],
            warmup_points: 0,
            dry_run: false,
        };
        let snapshot = RadiusWitnessSnapshot::from_radii(vec![Some(1.1), Some(1.0), Some(1.0)]);

        let decision =
            select_radius_certified_fanout(&config, &snapshot, &[0, 1, 2], &[1.0, 9.0, 16.0], 10);

        assert_eq!(decision.kept, 3);
        assert!(!decision.certified);
    }

    #[test]
    fn root_radius_witness_root_leader_helpers_live_with_radius_module() {
        let params = PiPNNParams::default();
        let mut rng = StdRng::seed_from_u64(42);
        let indices: Vec<u32> = (0..128).collect();
        let leaders = super::select_root_leaders(&indices, &params, &mut rng);

        assert!(!leaders.is_empty());
        assert_ne!(super::root_leader_fingerprint(&leaders), 0);
    }

    #[test]
    fn select_root_leaders_honors_capacity_target_when_enabled() {
        let mut params = PiPNNParams::default();
        params.psamp_fraction = 0.001;
        params.max_leaders = 10_000;
        params.gpu_subtree_d0_capacity_enable = true;
        params.gpu_subtree_d0_capacity_quality_min_root_leaders = 64;
        params.gpu_subtree_d0_capacity_root_leader_cap = 10_000;

        let indices: Vec<u32> = (0..10_000).collect();
        let mut rng = StdRng::seed_from_u64(7);
        let leaders =
            super::select_root_leaders_with_target(&indices, &params, Some(200), &mut rng);
        assert_eq!(leaders.len(), 200);
    }

    #[test]
    fn select_root_leaders_uncapped_target_bypasses_max_leaders() {
        let mut params = PiPNNParams::default();
        params.psamp_fraction = 0.001;
        params.max_leaders = 1_000;

        let indices: Vec<u32> = (0..10_000).collect();
        let mut rng = StdRng::seed_from_u64(11);
        let leaders =
            super::select_root_leaders_with_uncapped_target(&indices, &params, 2_560, &mut rng);

        assert_eq!(leaders.len(), 2_560);
    }

    #[test]
    fn spread_root_leaders_cover_sorted_sample_extremes_and_middle() {
        let sample: Vec<u32> = (0..100).collect();
        let leaders = super::select_spread_root_leaders_from_sorted_sample(&sample, 5);
        assert_eq!(leaders.len(), 5);
        assert_eq!(leaders[0], 0);
        assert_eq!(leaders[4], 99);
        assert!(leaders.contains(&25) || leaders.contains(&24));
        assert!(leaders.contains(&50) || leaders.contains(&49));
        assert!(leaders.contains(&74));
    }
}
