use std::time::{Duration, Instant};

use super::params::PiPNNParams;
use super::point_store::{PointStore, WindowedGatherOptions};
use super::scheduler::LeafEmitter;
use crate::common::{AnnError, AnnResult, Metric};

const KMEANS_READ_CHUNK_POINTS: usize = 4096;
const KMEANS_MIN_CHILDREN: usize = 2;

#[derive(Clone, Debug, Default)]
pub struct KMeansPartitionTelemetry {
    pub points: usize,
    pub emitted_points: usize,
    pub leaves: usize,
    pub split_attempts: usize,
    pub assignment_passes: usize,
    pub empty_clusters: usize,
    pub fallback_splits: usize,
    pub max_leaf_size: usize,
    pub wall: Duration,
}

pub(crate) fn kmeans_partition_streaming(
    dataset: &dyn PointStore,
    indices: &[u32],
    metric: Metric,
    params: &PiPNNParams,
    leaf_emitter: &dyn LeafEmitter,
) -> AnnResult<KMeansPartitionTelemetry> {
    if metric != Metric::L2 {
        return Err(AnnError::log_index_error(
            "PiPNN k-means partitioner currently supports only L2".to_string(),
        ));
    }
    let start = Instant::now();
    let mut telemetry = KMeansPartitionTelemetry {
        points: indices.len(),
        ..KMeansPartitionTelemetry::default()
    };
    if indices.is_empty() {
        return Ok(telemetry);
    }

    let target_leaf_size = params.adaptive_c_max(indices.len()).max(1);
    let mut stack = vec![indices.to_vec()];
    while let Some(points) = stack.pop() {
        if points.len() <= target_leaf_size {
            telemetry.leaves += 1;
            telemetry.emitted_points += points.len();
            telemetry.max_leaf_size = telemetry.max_leaf_size.max(points.len());
            leaf_emitter.emit_leaf(points)?;
            continue;
        }

        telemetry.split_attempts += 1;
        let children = split_kmeans(dataset, &points, target_leaf_size, params, &mut telemetry)?;
        if children.len() <= 1 {
            telemetry.fallback_splits += 1;
            stack.extend(fallback_balanced_split(points, target_leaf_size));
        } else {
            stack.extend(children);
        }
    }

    telemetry.wall = start.elapsed();
    Ok(telemetry)
}

fn split_kmeans(
    dataset: &dyn PointStore,
    points: &[u32],
    target_leaf_size: usize,
    params: &PiPNNParams,
    telemetry: &mut KMeansPartitionTelemetry,
) -> AnnResult<Vec<Vec<u32>>> {
    let dim = dataset.dim();
    let child_count = planned_child_count(points.len(), target_leaf_size, params);
    if child_count <= 1 {
        return Ok(vec![points.to_vec()]);
    }

    let mut centers = initialize_centers(dataset, points, child_count)?;
    let mut assignments = vec![0usize; points.len()];
    for _ in 0..params.kmeans_iterations.max(1) {
        assign_points(dataset, points, &centers, &mut assignments)?;
        telemetry.assignment_passes += 1;

        let mut sums = vec![0.0f32; child_count.saturating_mul(dim)];
        let mut counts = vec![0usize; child_count];
        accumulate_cluster_sums(
            dataset,
            points,
            &assignments,
            child_count,
            &mut sums,
            &mut counts,
        )?;
        for cluster in 0..child_count {
            if counts[cluster] == 0 {
                telemetry.empty_clusters += 1;
                let fallback_point =
                    points[(cluster * points.len() / child_count).min(points.len() - 1)];
                dataset.read_point_into(
                    fallback_point,
                    &mut centers[cluster * dim..(cluster + 1) * dim],
                )?;
                continue;
            }
            let scale = 1.0f32 / counts[cluster] as f32;
            let center = &mut centers[cluster * dim..(cluster + 1) * dim];
            let sum = &sums[cluster * dim..(cluster + 1) * dim];
            for (dst, &value) in center.iter_mut().zip(sum.iter()) {
                *dst = value * scale;
            }
        }
    }

    assign_points(dataset, points, &centers, &mut assignments)?;
    telemetry.assignment_passes += 1;

    let mut children: Vec<Vec<u32>> = (0..child_count).map(|_| Vec::new()).collect();
    for (&point, &cluster) in points.iter().zip(assignments.iter()) {
        children[cluster].push(point);
    }
    children.retain(|child| !child.is_empty());
    Ok(children)
}

fn planned_child_count(n: usize, target_leaf_size: usize, params: &PiPNNParams) -> usize {
    let target = target_leaf_size.max(1);
    let base = n.div_ceil(target).max(KMEANS_MIN_CHILDREN);
    let requested = base.saturating_mul(params.kmeans_oversplit_factor.max(1));
    requested.clamp(
        KMEANS_MIN_CHILDREN,
        params.max_leaders.max(KMEANS_MIN_CHILDREN).min(n),
    )
}

fn initialize_centers(
    dataset: &dyn PointStore,
    points: &[u32],
    child_count: usize,
) -> AnnResult<Vec<f32>> {
    let dim = dataset.dim();
    let mut center_ids = Vec::with_capacity(child_count);
    for cluster in 0..child_count {
        let pos = cluster.saturating_mul(points.len()) / child_count;
        center_ids.push(points[pos.min(points.len() - 1)]);
    }
    let mut centers = vec![0.0f32; child_count.saturating_mul(dim)];
    dataset.read_points_into(&center_ids, &mut centers)?;
    Ok(centers)
}

fn assign_points(
    dataset: &dyn PointStore,
    points: &[u32],
    centers: &[f32],
    assignments: &mut [usize],
) -> AnnResult<()> {
    let dim = dataset.dim();
    let child_count = centers.len() / dim.max(1);
    let mut buffer = vec![0.0f32; KMEANS_READ_CHUNK_POINTS.saturating_mul(dim)];
    let options = default_kmeans_read_options(dataset);
    let mut stats = super::point_store::PointBatchStats::default();

    for (chunk_idx, ids) in points.chunks(KMEANS_READ_CHUNK_POINTS).enumerate() {
        let rows = ids.len();
        dataset.read_points_windowed_into_batch_stats(
            ids,
            &mut buffer[..rows * dim],
            &options,
            &mut stats,
        )?;
        for row in 0..rows {
            let point = &buffer[row * dim..(row + 1) * dim];
            let mut best_cluster = 0usize;
            let mut best_dist = f32::INFINITY;
            for cluster in 0..child_count {
                let center = &centers[cluster * dim..(cluster + 1) * dim];
                let dist = l2_distance_sq(point, center);
                if dist < best_dist || (dist == best_dist && cluster < best_cluster) {
                    best_dist = dist;
                    best_cluster = cluster;
                }
            }
            assignments[chunk_idx * KMEANS_READ_CHUNK_POINTS + row] = best_cluster;
        }
    }
    Ok(())
}

fn accumulate_cluster_sums(
    dataset: &dyn PointStore,
    points: &[u32],
    assignments: &[usize],
    child_count: usize,
    sums: &mut [f32],
    counts: &mut [usize],
) -> AnnResult<()> {
    let dim = dataset.dim();
    debug_assert_eq!(sums.len(), child_count.saturating_mul(dim));
    debug_assert_eq!(counts.len(), child_count);
    let mut buffer = vec![0.0f32; KMEANS_READ_CHUNK_POINTS.saturating_mul(dim)];
    let options = default_kmeans_read_options(dataset);
    let mut stats = super::point_store::PointBatchStats::default();

    for (chunk_idx, ids) in points.chunks(KMEANS_READ_CHUNK_POINTS).enumerate() {
        let rows = ids.len();
        dataset.read_points_windowed_into_batch_stats(
            ids,
            &mut buffer[..rows * dim],
            &options,
            &mut stats,
        )?;
        for row in 0..rows {
            let global_row = chunk_idx * KMEANS_READ_CHUNK_POINTS + row;
            let cluster = assignments[global_row];
            counts[cluster] += 1;
            let point = &buffer[row * dim..(row + 1) * dim];
            let sum = &mut sums[cluster * dim..(cluster + 1) * dim];
            for (dst, &value) in sum.iter_mut().zip(point.iter()) {
                *dst += value;
            }
        }
    }
    Ok(())
}

fn fallback_balanced_split(mut points: Vec<u32>, target_leaf_size: usize) -> Vec<Vec<u32>> {
    points.sort_unstable();
    points
        .chunks(target_leaf_size.max(1))
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn default_kmeans_read_options(dataset: &dyn PointStore) -> WindowedGatherOptions {
    let row_bytes = dataset.dim().saturating_mul(std::mem::size_of::<f32>());
    WindowedGatherOptions {
        max_gap_rows: if row_bytes <= 4096 { 1 } else { 0 },
        max_window_bytes: row_bytes
            .saturating_mul(KMEANS_READ_CHUNK_POINTS)
            .min(8 * 1024 * 1024)
            .max(256 * 1024),
        alignment_bytes: 4096,
        sort_ids: true,
    }
}

#[inline]
fn l2_distance_sq(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(&lhs, &rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{KMeansPartitionTelemetry, kmeans_partition_streaming};
    use crate::algorithm::pipnn::params::PiPNNParams;
    use crate::algorithm::pipnn::point_store::InmemDatasetPointStore;
    use crate::algorithm::pipnn::scheduler::LeafEmitter;
    use crate::common::{AnnResult, Metric};
    use crate::model::InmemDataset;

    #[derive(Default)]
    struct CollectingEmitter {
        leaves: Mutex<Vec<Vec<u32>>>,
    }

    impl LeafEmitter for CollectingEmitter {
        fn emit_leaf(&self, leaf: Vec<u32>) -> AnnResult<()> {
            self.leaves.lock().unwrap().push(leaf);
            Ok(())
        }
    }

    fn two_cluster_dataset(points_per_cluster: usize, dim: usize) -> InmemDataset<f32> {
        let num_points = points_per_cluster * 2;
        let mut dataset = InmemDataset::new(num_points, 1.0, dim).unwrap();
        for point in 0..num_points {
            let cluster_offset = if point < points_per_cluster {
                0.0
            } else {
                100.0
            };
            let offset = point * dim;
            for axis in 0..dim {
                dataset.data[offset + axis] = cluster_offset + point as f32 * 0.01 + axis as f32;
            }
        }
        dataset
    }

    #[test]
    fn kmeans_partition_emits_each_point_once() {
        let dataset = two_cluster_dataset(8, 4);
        let store = InmemDatasetPointStore::new(&dataset, dataset.num_points);
        let mut params = PiPNNParams::default();
        params.c_max = 6;
        params.max_leaders = 8;
        params.kmeans_iterations = 2;
        params.kmeans_oversplit_factor = 2;
        let emitter = CollectingEmitter::default();

        let telemetry = kmeans_partition_streaming(
            &store,
            &(0..dataset.num_points as u32).collect::<Vec<_>>(),
            Metric::L2,
            &params,
            &emitter,
        )
        .unwrap();
        let leaves = emitter.leaves.lock().unwrap();
        let mut seen = vec![0usize; dataset.num_points];
        for leaf in leaves.iter() {
            assert!(
                leaf.len() <= params.c_max,
                "leaf len {} exceeded c_max {} in {leaf:?}",
                leaf.len(),
                params.c_max
            );
            for &point in leaf {
                seen[point as usize] += 1;
            }
        }

        assert_eq!(seen, vec![1; dataset.num_points]);
        assert_eq!(telemetry.points, dataset.num_points);
        assert_eq!(telemetry.emitted_points, dataset.num_points);
    }

    #[test]
    fn kmeans_partition_telemetry_tracks_leaf_count() {
        let dataset = two_cluster_dataset(6, 3);
        let store = InmemDatasetPointStore::new(&dataset, dataset.num_points);
        let mut params = PiPNNParams::default();
        params.c_max = 4;
        params.max_leaders = 6;
        params.kmeans_iterations = 1;
        params.kmeans_oversplit_factor = 1;
        let emitter = CollectingEmitter::default();

        let telemetry: KMeansPartitionTelemetry = kmeans_partition_streaming(
            &store,
            &(0..dataset.num_points as u32).collect::<Vec<_>>(),
            Metric::L2,
            &params,
            &emitter,
        )
        .unwrap();
        let leaves = emitter.leaves.lock().unwrap();

        assert_eq!(telemetry.leaves, leaves.len());
        assert!(telemetry.assignment_passes >= 1);
    }
}
