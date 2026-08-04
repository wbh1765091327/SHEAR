use std::collections::BTreeSet;

use crate::common::{AnnError, AnnResult};

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

fn kmeans_error(message: impl Into<String>) -> AnnError {
    AnnError::log_index_error(format!("graph RaBitQ1 K-means: {}", message.into()))
}

pub fn deterministic_sample_ids(rows: usize, sample_rows: usize, seed: u64) -> AnnResult<Vec<u64>> {
    if sample_rows == 0 || sample_rows > rows {
        return Err(kmeans_error("sample_rows must be in 1..=rows"));
    }
    let mut rng = SplitMix64::new(seed);
    let mut selected = BTreeSet::new();
    for j in rows - sample_rows..rows {
        let candidate = rng.next_u64() % (j as u64 + 1);
        if !selected.insert(candidate) {
            selected.insert(j as u64);
        }
    }
    Ok(selected.into_iter().collect())
}

fn squared_distance(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| {
            let difference = f64::from(left) - f64::from(right);
            difference * difference
        })
        .sum()
}

fn nearest_center(point: &[f32], centers: &[f32], center_count: usize, dim: usize) -> usize {
    let mut best_center = 0;
    let mut best_distance = squared_distance(point, &centers[..dim]);
    for center_id in 1..center_count {
        let start = center_id * dim;
        let distance = squared_distance(point, &centers[start..start + dim]);
        if distance < best_distance {
            best_distance = distance;
            best_center = center_id;
        }
    }
    best_center
}

fn initialize_kmeans_plus_plus(
    points: &[f32],
    rows: usize,
    dim: usize,
    center_count: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut selected_rows = Vec::with_capacity(center_count);
    selected_rows.push((rng.next_u64() % rows as u64) as usize);
    let mut minimum_distances = vec![f64::INFINITY; rows];

    while selected_rows.len() < center_count {
        let newest = selected_rows[selected_rows.len() - 1];
        let newest_point = &points[newest * dim..(newest + 1) * dim];
        for row in 0..rows {
            let point = &points[row * dim..(row + 1) * dim];
            minimum_distances[row] =
                minimum_distances[row].min(squared_distance(point, newest_point));
        }
        let total_weight: f64 = minimum_distances.iter().sum();
        let selected = if total_weight > 0.0 && total_weight.is_finite() {
            let target = rng.next_unit_f64() * total_weight;
            let mut cumulative = 0.0;
            let mut chosen = rows - 1;
            for (row, &weight) in minimum_distances.iter().enumerate() {
                cumulative += weight;
                if target < cumulative {
                    chosen = row;
                    break;
                }
            }
            chosen
        } else {
            (0..rows)
                .find(|row| !selected_rows.contains(row))
                .unwrap_or(selected_rows[0])
        };
        selected_rows.push(selected);
    }

    let mut centers = Vec::with_capacity(center_count * dim);
    for row in selected_rows {
        centers.extend_from_slice(&points[row * dim..(row + 1) * dim]);
    }
    centers
}

pub fn train_centers(
    points: &[f32],
    rows: usize,
    dim: usize,
    center_count: usize,
    iterations: usize,
    seed: u64,
) -> AnnResult<Vec<f32>> {
    if rows == 0 || dim == 0 || center_count == 0 || center_count > rows || iterations == 0 {
        return Err(kmeans_error(
            "rows, dim, center_count, and iterations must be positive and center_count <= rows",
        ));
    }
    let expected_values = rows
        .checked_mul(dim)
        .ok_or_else(|| kmeans_error("sample shape overflow"))?;
    if points.len() != expected_values {
        return Err(kmeans_error(format!(
            "point length {} does not match {rows}x{dim}",
            points.len()
        )));
    }
    if points.iter().any(|value| !value.is_finite()) {
        return Err(kmeans_error("sample contains a non-finite value"));
    }

    let mut centers = initialize_kmeans_plus_plus(points, rows, dim, center_count, seed);
    let mut sums = vec![0.0_f64; center_count * dim];
    let mut counts = vec![0_usize; center_count];
    for _ in 0..iterations {
        sums.fill(0.0);
        counts.fill(0);
        for point in points.chunks_exact(dim) {
            let center_id = nearest_center(point, &centers, center_count, dim);
            counts[center_id] += 1;
            for coordinate in 0..dim {
                sums[center_id * dim + coordinate] += f64::from(point[coordinate]);
            }
        }
        for center_id in 0..center_count {
            if counts[center_id] == 0 {
                continue;
            }
            for coordinate in 0..dim {
                centers[center_id * dim + coordinate] =
                    (sums[center_id * dim + coordinate] / counts[center_id] as f64) as f32;
            }
        }
    }
    Ok(centers)
}
