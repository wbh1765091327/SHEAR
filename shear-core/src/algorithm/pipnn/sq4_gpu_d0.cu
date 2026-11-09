// Adapted from SHEAR/sq8_plus_staging.cu (2026-11-05): eight warps per
// point for assignment, and one CTA per point for the four FHT rounds.
// SQ4 uses original-space D0 and R(x-c_primary), never Rx-Rc_partition.

// ndarray 0.14's nonsquare transposed RHS uses matrixmultiply 0.2.4,
// not BLAS. Its FMA kernel accumulates k in order in 256-element panels,
// then merges the panels with alpha=1, beta=0/1. Preserve that expression
// on GPU; parallelize independent point/center pairs, never the k reduction.
// Training supports the validated 64/512-center, d=512 configurations.
extern "C" __global__ void sq4_kmeans_init(const float* samples, unsigned rows, unsigned leaders, float* centers) {
    unsigned c = blockIdx.x, row = (unsigned)((size_t)c * rows / leaders);
    for (unsigned d=threadIdx.x; d<512; d+=blockDim.x)
        centers[c*512+d] = samples[(size_t)row*512+d];
}

extern "C" __global__ void sq4_kmeans_prepare(const float* centers, unsigned leaders, float* transposed, float* norms) {
    unsigned c=blockIdx.x;
    for (unsigned d=threadIdx.x; d<512; d+=blockDim.x) transposed[d*leaders+c]=centers[c*512+d];
    if (threadIdx.x==0) {
        float sum=0.0f;
        for (unsigned d=0; d<512; ++d) sum=__fadd_rn(sum,__fmul_rn(centers[c*512+d],centers[c*512+d]));
        norms[c]=sum;
    }
}

// Stable membership lists retain CPU sample order; no atomic floating sums.
extern "C" __global__ void sq4_kmeans_update(const float* samples, const unsigned* selected,
    unsigned rows, unsigned leaders, unsigned* members, float* centers) {
    unsigned c=blockIdx.x;
    __shared__ unsigned count;
    if (threadIdx.x==0) {
        unsigned n=0;
        for (unsigned row=0; row<rows; ++row)
            if (selected[row]==c) members[(size_t)c*rows+n++]=row;
        count=n;
    }
    __syncthreads();
    float scale=count ? __fdiv_rn(1.0f,(float)count) : 1.0f;
    for (unsigned d=threadIdx.x; d<512; d+=blockDim.x) {
        float sum=0.0f;
        for (unsigned i=0; i<count; ++i)
            sum=__fadd_rn(sum,samples[(size_t)members[(size_t)c*rows+i]*512+d]);
        centers[c*512+d]=count ? __fmul_rn(sum,scale) : samples[((size_t)c*rows/leaders)*512+d];
    }
}

extern "C" __global__ void sq4_primary_assign_exact(
    const float* points, const float* centers_t, const float* center_norms,
    unsigned rows, unsigned fanout, unsigned* selected,
    unsigned capture, float* dot_scores) {
    __shared__ float x[4][512];
    __shared__ float scores[4][64];
    __shared__ float norms[4];
    unsigned local = threadIdx.x / 64, center = threadIdx.x % 64;
    unsigned row = blockIdx.x * 4 + local;
    for (unsigned i = threadIdx.x; i < 4 * 512; i += 256) {
        unsigned r = blockIdx.x * 4 + i / 512;
        x[i / 512][i % 512] = r < rows ? points[(size_t)r * 512 + i % 512] : 0.0f;
    }
    __syncthreads();
    if (center == 0) {
        float norm = 0.0f;
        for (unsigned axis = 0; axis < 512; ++axis)
            norm = __fadd_rn(norm, __fmul_rn(x[local][axis], x[local][axis]));
        norms[local] = norm;
    }
    float dot = 0.0f;
    for (unsigned panel = 0; panel < 512; panel += 256) {
        float partial = 0.0f;
        #pragma unroll 4
        for (unsigned axis = panel; axis < panel + 256; ++axis)
            partial = __fmaf_rn(x[local][axis], centers_t[axis * 64 + center], partial);
        dot = __fmaf_rn(1.0f, partial, dot);
    }
    __syncthreads();
    float score = __fsub_rn(__fadd_rn(norms[local], center_norms[center]), __fmul_rn(2.0f, dot));
    scores[local][center] = score;
    if (capture && row < rows) {
        dot_scores[(size_t)row * 64 + center] = dot;
        dot_scores[(size_t)(rows + row) * 64 + center] = score;
    }
    __syncthreads();
    if (center == 0 && row < rows) {
        for (unsigned rank = 0; rank < fanout; ++rank) {
            unsigned best = 64;
            for (unsigned c = 0; c < 64; ++c) {
                bool used = false;
                for (unsigned r = 0; r < rank; ++r)
                    used |= selected[(size_t)row * fanout + r] == c;
                if (!used && (best == 64 || scores[local][c] < scores[local][best])) best = c;
            }
            selected[(size_t)row * fanout + rank] = best;
        }
    }
}

// Budget routing may have any center count. Keep the same 256-axis FP32
// accumulation order; tile only independent centers, never the k reduction.
extern "C" __global__ void sq4_budget_assign_exact(
    const float* points, const float* centers_t, const float* center_norms,
    unsigned rows, unsigned leaders, unsigned fanout, unsigned* selected,
    unsigned capture, float* dot_scores) {
    __shared__ float x[4][512], scores[4][64], norms[4], best[4][32];
    __shared__ unsigned ids[4][32];
    unsigned local = threadIdx.x / 64, lane = threadIdx.x % 64;
    unsigned row = blockIdx.x * 4 + local;
    for (unsigned i = threadIdx.x; i < 4 * 512; i += 256) {
        unsigned r = blockIdx.x * 4 + i / 512;
        x[i / 512][i % 512] = r < rows ? points[(size_t)r * 512 + i % 512] : 0.0f;
    }
    __syncthreads();
    if (lane == 0) {
        float norm = 0.0f;
        for (unsigned axis = 0; axis < 512; ++axis)
            norm = __fadd_rn(norm, __fmul_rn(x[local][axis], x[local][axis]));
        norms[local] = norm;
        for (unsigned k = 0; k < fanout; ++k) {best[local][k] = __int_as_float(0x7f800000); ids[local][k] = 0xffffffffu;}
    }
    __syncthreads();
    for (unsigned base = 0; base < leaders; base += 64) {
        unsigned center = base + lane;
        float dot = 0.0f;
        if (center < leaders) {
            for (unsigned panel = 0; panel < 512; panel += 256) {
                float partial = 0.0f;
                #pragma unroll 4
                for (unsigned axis = panel; axis < panel + 256; ++axis)
                    partial = __fmaf_rn(x[local][axis], centers_t[(size_t)axis * leaders + center], partial);
                dot = __fmaf_rn(1.0f, partial, dot);
            }
            float score = __fsub_rn(__fadd_rn(norms[local], center_norms[center]), __fmul_rn(2.0f, dot));
            scores[local][lane] = score;
            if (capture && row < rows) {
                dot_scores[(size_t)row * leaders + center] = dot;
                dot_scores[(size_t)(rows + row) * leaders + center] = score;
            }
        }
        __syncthreads();
        if (lane == 0 && row < rows) {
            for (unsigned c = 0; c < 64 && base+c < leaders; ++c) {
                float score = scores[local][c]; unsigned id = base+c;
                for (unsigned pos = 0; pos < fanout; ++pos) {
                    if (score < best[local][pos] || (score == best[local][pos] && id < ids[local][pos])) {
                        for (unsigned j=fanout-1; j>pos; --j) {best[local][j]=best[local][j-1];ids[local][j]=ids[local][j-1];}
                        best[local][pos]=score;ids[local][pos]=id;break;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows)
        for (unsigned k=0; k<fanout; ++k) selected[(size_t)row*fanout+k]=ids[local][k];
}

extern "C" __global__ void sq4_primary_assign(
    const float* points, const float* centers, const float* center_norms,
    unsigned rows, unsigned dim, unsigned leaders, unsigned fanout,
    float error_factor, float underflow_bound, unsigned centers_subnormal,
    unsigned* selected, unsigned* uncertain) {
    extern __shared__ float shared[];
    float* distances = shared + dim;
    float* lower = distances + leaders;
    float* upper = lower + leaders;
    __shared__ float norm, norm_error;
    __shared__ unsigned abnormal;
    unsigned row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (row >= rows) return;
    for (unsigned axis = threadIdx.x; axis < dim; axis += blockDim.x)
        shared[axis] = points[(size_t)row * dim + axis];
    __syncthreads();
    if (threadIdx.x == 0) {
        float n = 0.0f, magnitude = 0.0f;
        abnormal = centers_subnormal;
        for (unsigned axis = 0; axis < dim; ++axis) {
            float x = shared[axis];
            if (fabsf(x) > 0.0f && fabsf(x) < 0x1p-126f) abnormal = 1;
            n = __fadd_rn(n, __fmul_rn(x, x));
            magnitude = __fadd_ru(magnitude, __fmul_ru(fabsf(x), fabsf(x)));
        }
        norm = n;
        norm_error = __fadd_ru(__fmul_ru(error_factor, magnitude), underflow_bound);
    }
    __syncthreads();
    for (unsigned center = warp; center < leaders; center += 8) {
        float dot = 0.0f, magnitude = 0.0f;
        for (unsigned axis = lane; axis < dim; axis += 32) {
            float x = shared[axis], c = centers[(size_t)center * dim + axis];
            dot = __fadd_rn(dot, __fmul_rn(x, c));
            // Directed rounding makes this an upper bound on sum |x_i*c_i|.
            magnitude = __fadd_ru(magnitude, __fmul_ru(fabsf(x), fabsf(c)));
        }
        for (int offset = 16; offset; offset /= 2) {
            dot = __fadd_rn(dot, __shfl_down_sync(0xffffffff, dot, offset));
            magnitude = __fadd_ru(magnitude, __shfl_down_sync(0xffffffff, magnitude, offset));
        }
        if (lane == 0) {
            float error = __fadd_ru(__fmul_ru(error_factor, magnitude), underflow_bound);
            float dot_lo = __fsub_rd(dot, error), dot_hi = __fadd_ru(dot, error);
            float norm_lo = fmaxf(0.0f, __fsub_rd(norm, norm_error));
            float norm_hi = __fadd_ru(norm, norm_error);
            float c = center_norms[center];
            distances[center] = __fsub_rn(__fadd_rn(norm, c), __fmul_rn(2.0f, dot));
            lower[center] = __fsub_rn(__fadd_rn(norm_lo, c), __fmul_rn(2.0f, dot_hi));
            upper[center] = __fsub_rn(__fadd_rn(norm_hi, c), __fmul_rn(2.0f, dot_lo));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned needs_check = abnormal;
        for (unsigned center = 0; center < leaders; ++center)
            if (!isfinite(distances[center]) || !isfinite(lower[center]) || !isfinite(upper[center]))
                needs_check = 1;
        for (unsigned rank = 0; rank < fanout; ++rank) {
            unsigned best = leaders;
            for (unsigned center = 0; center < leaders; ++center) {
                bool used = false;
                for (unsigned r = 0; r < rank; ++r)
                    used |= selected[(size_t)row * fanout + r] == center;
                if (!used && (best == leaders || distances[center] < distances[best])) best = center;
            }
            // Certify the complete ordered top-k, including the primary center.
            for (unsigned center = 0; center < leaders; ++center) {
                bool used = center == best;
                for (unsigned r = 0; r < rank; ++r)
                    used |= selected[(size_t)row * fanout + r] == center;
                if (!used && !(upper[best] < lower[center]
                    || (upper[best] == lower[center] && best < center))) needs_check = 1;
            }
            selected[(size_t)row * fanout + rank] = best;
        }
        uncertain[row] = needs_check;
    }
}

extern "C" __global__ void sq4_primary_encode(
    const float* points, const float* centers, const unsigned* selected,
    const unsigned* signs, unsigned rows, unsigned dim, unsigned padded,
    unsigned fanout, unsigned repeats, float rotation_scale, signed char* codes,
    float* scales, float* norms, unsigned* invalid,
    unsigned capture, float* intermediate) {
    extern __shared__ float values[];
    __shared__ float scale;
    unsigned output = blockIdx.x;
    unsigned row = output / repeats, rank = output % repeats;
    if (row >= rows) return;
    unsigned center = selected[(size_t)row * fanout + rank];
    for (unsigned axis = threadIdx.x; axis < dim; axis += blockDim.x) {
        float r = __fsub_rn(points[(size_t)row * dim + axis], centers[(size_t)center * dim + axis]);
        values[axis] = r;
        if (capture) intermediate[(size_t)row * dim + axis] = r;
    }
    __syncthreads();
    for (unsigned round = 0; round < 4; ++round) {
        for (unsigned axis = threadIdx.x; axis < dim; axis += blockDim.x)
            values[axis] = __uint_as_float(__float_as_uint(values[axis]) ^ signs[(size_t)round * dim + axis]);
        __syncthreads();
        for (unsigned width = 1; width < dim; width *= 2) {
            for (unsigned pair = threadIdx.x; pair < dim / 2; pair += blockDim.x) {
                unsigned left = (pair / width) * (2 * width) + pair % width;
                float a = values[left], b = values[left + width];
                values[left] = __fadd_rn(a, b);
                values[left + width] = __fsub_rn(a, b);
            }
            __syncthreads();
        }
        for (unsigned axis = threadIdx.x; axis < dim; axis += blockDim.x)
            values[axis] = __fmul_rn(values[axis], rotation_scale);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Preserve the CPU's ascending-axis norm accumulation, not a tree sum.
        float maximum = 0.0f, norm = 0.0f;
        for (unsigned axis = 0; axis < dim; ++axis) {
            float x = values[axis];
            if (!isfinite(x)) atomicExch(invalid, 1u);
            maximum = fmaxf(maximum, fabsf(x));
            norm = __fadd_rn(norm, __fmul_rn(x, x));
        }
        scale = maximum == 0.0f ? 1.0f : __fdiv_rn(maximum, 7.0f);
        if (!isfinite(norm) || !isfinite(scale) || scale <= 0.0f) atomicExch(invalid, 1u);
        scales[output] = scale;
        norms[output] = norm;
    }
    __syncthreads();
    if (capture) {
        for (unsigned axis = threadIdx.x; axis < dim; axis += blockDim.x)
            intermediate[(size_t)(rows + row) * dim + axis] = values[axis];
    }
    // A single thread owns both nibbles of each byte, avoiding packed-write races.
    for (unsigned byte = threadIdx.x; byte < padded / 2; byte += blockDim.x) {
        unsigned packed = 0;
        for (unsigned half = 0; half < 2; ++half) {
            unsigned axis = 2 * byte + half;
            float value = axis < dim ? values[axis] : 0.0f;
            int q = (int)fminf(7.0f, fmaxf(-7.0f, roundf(__fdiv_rn(value, scale))));
            packed |= ((unsigned)q & 15u) << (4 * half);
        }
        codes[(size_t)output * (padded / 2) + byte] = (signed char)packed;
    }
}
