// Each 8x8 INT4 dot tile is stored once. Both endpoints read the same integer
// dot, then score and insert in the directed executor's original column order.
static __device__ unsigned long long sq4_triangle_offset(unsigned int blocks,
    unsigned int first, unsigned int second) {
    return ((unsigned long long)first * (2ull * blocks - first + 1u) / 2u + second - first) * 64ull;
}

extern "C" __global__ void sq4_primary_triangle_blocks(
    const unsigned long long* tasks,
    const unsigned int* rows, const signed char* codes, unsigned int resident_rows,
    unsigned int padded_dim, unsigned int stride, int* dot_tiles
) {
    const unsigned long long* task = tasks + (unsigned long long)blockIdx.x * 5u;
    unsigned int start = (unsigned int)task[0];
    unsigned int count = (unsigned int)task[1];
    unsigned int first = (unsigned int)task[2];
    unsigned int second = (unsigned int)task[3];
    unsigned long long partial_base = task[4];
    unsigned int left_count = min(64u, count - first);
    unsigned int right_count = min(64u, count - second);
    unsigned int left_blocks = (left_count + 7u) / 8u;
    unsigned int right_blocks = (right_count + 7u) / 8u;
    bool diagonal = first == second;
    unsigned int tiles = diagonal ? left_blocks * (left_blocks + 1u) / 2u
                                  : left_blocks * right_blocks;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int blocks = (count + 7u) / 8u;
    __shared__ __align__(32) unsigned int a[8][32], b[8][32];

    for (unsigned int tile = warp; tile < tiles; tile += blockDim.x >> 5) {
        unsigned int rb = tile / right_blocks, cb = tile % right_blocks;
        if (diagonal) {
            rb = 0u;
            cb = tile;
            while (cb >= left_blocks - rb) { cb -= left_blocks - rb; ++rb; }
            cb += rb;
        }
        unsigned int i = rb * 8u + lane / 4u, j = cb * 8u + lane / 4u;
        gpu_persistent_sq4_wmma_dot8x8(codes, stride, padded_dim,
            i < left_count ? rows[start + first + i] : 0xffffffffu,
            j < right_count ? rows[start + second + j] : 0xffffffffu,
            resident_rows, a[warp], b[warp],
            dot_tiles + partial_base + sq4_triangle_offset(blocks, first / 8u + rb, second / 8u + cb), 8u);
    }
}

extern "C" __global__ void sq4_primary_triangle_merge(
    const unsigned long long* leaves, unsigned int k, const int* dot_tiles,
    const unsigned int* rows, const unsigned int* ids, const float* norms,
    const float* scales, const float* primary_ids, const float* projections,
    unsigned int* output_ids, float* output_scores
) {
    const unsigned long long* leaf = leaves + (unsigned long long)blockIdx.x * 3u;
    unsigned int start = (unsigned int)leaf[0], count = (unsigned int)leaf[1];
    unsigned int blocks = (count + 7u) / 8u;
    for (unsigned int row = threadIdx.x; row < count; row += blockDim.x) {
        unsigned int point = rows[start + row];
        float scores[4] = {3.402823466e+38F,3.402823466e+38F,3.402823466e+38F,3.402823466e+38F};
        unsigned int chosen[4] = {~0u,~0u,~0u,~0u};
        unsigned int rb = row / 8u;
        for (unsigned int cb = 0u; cb < blocks; ++cb) {
            bool reflect = cb < rb;
            const int* tile = dot_tiles + leaf[2] + sq4_triangle_offset(blocks,min(rb,cb),max(rb,cb));
            for (unsigned int col = cb * 8u; col < min(count, cb * 8u + 8u); ++col) {
                unsigned int candidate = rows[start + col];
                if (point == candidate) continue;
                int dot = tile[reflect ? (col % 8u) * 8u + row % 8u : (row % 8u) * 8u + col % 8u];
                float score = gpu_persistent_rotated_residual_sq8_l2_from_dot(
                    dot, point, candidate, norms, scales, primary_ids, projections);
                gpu_persistent_insert_topk_candidate_fp32(score, ids[candidate], k, scores, chosen);
            }
        }
        unsigned long long offset = ((unsigned long long)start + row) * k;
        for (unsigned int slot = 0; slot < k; ++slot) {
            output_ids[offset + slot] = chosen[slot];
            output_scores[offset + slot] = scores[slot];
        }
    }
}
