// Each half-warp owns a point; lanes own projections, never partial sums.
// Keep the CPU reference's ascending-axis FP32 multiply/add order.
extern "C" __global__ void hashprune_sketch_fp32(
    const float* __restrict__ points,
    const float* __restrict__ planes_axis_major,
    float* __restrict__ output,
    unsigned int rows, unsigned int dim, unsigned int width, unsigned int plane_stride) {
    unsigned int thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = thread / 16u;
    if (row >= rows) return;
    for (unsigned int projection = thread % 16u; projection < width; projection += 16u) {
        float sum = 0.0f;
        #pragma unroll 4
        for (unsigned int axis = 0; axis < dim; ++axis) {
            float x = points[(unsigned long long)row * dim + axis];
            float h = planes_axis_major[(unsigned long long)axis * plane_stride + projection];
            sum = __fadd_rn(sum, __fmul_rn(x, h));
        }
        output[(unsigned long long)row * width + projection] = sum;
    }
}
