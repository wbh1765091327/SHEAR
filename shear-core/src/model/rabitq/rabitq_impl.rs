//! RabitQ implementation in Rust
use crate::model::rabitq::ffi;

pub fn range_simd(q: &[f32], c: &[f32], b: usize) -> (f32, f32) {
    let mut vl = 0.0;
    let mut vr = 0.0;
    unsafe {
        ffi::rangeSIMD(q.as_ptr(), c.as_ptr(), &mut vl, &mut vr, b as i32);
    }
    (vl, vr)
}

pub fn range_common(q: &[f32], c: &[f32], b: usize) -> (f32, f32) {
    let mut vl = 0.0;
    let mut vr = 0.0;
    unsafe {
        ffi::rangeCommon(q.as_ptr(), c.as_ptr(), &mut vl, &mut vr, b as i32);
    }
    (vl, vr)
}

pub fn range(q: &[f32], c: &[f32], b: usize) -> (f32, f32) {
    match b {
        768 => range_simd(q, c, b),
        128 => range_common(q, c, b),
        _ => range_common(q, c, b),
    }
}

// Quantize functions
pub fn quantize_simd(
    result: &mut [u8],
    q: &[f32],
    c: &[f32],
    u: &[f32],
    vl: f32,
    width: f32,
    b: usize,
) -> u32 {
    let mut sum_q = 0u32;
    unsafe {
        ffi::quantizeSIMD(
            result.as_mut_ptr(),
            q.as_ptr(),
            c.as_ptr(),
            u.as_ptr(),
            vl,
            width,
            &mut sum_q,
            b as i32,
        );
    }
    sum_q
}

pub fn quantize(
    result: &mut [u8],
    q: &[f32],
    c: &[f32],
    u: &[f32],
    vl: f32,
    width: f32,
    b: usize,
) -> u32 {
    let mut sum_q = 0u32;
    unsafe {
        ffi::quantize(
            result.as_mut_ptr(),
            q.as_ptr(),
            c.as_ptr(),
            u.as_ptr(),
            vl,
            width,
            &mut sum_q,
            b as i32,
        );
    }
    sum_q
}

// Transpose binary - SIMD优化版本
pub fn transpose_bin(q: &[u8], b: usize) -> Vec<u32> {
    let mut tq = vec![0u32; 4 * b / 32];
    unsafe {
        ffi::transpose_bin(q.as_ptr(), tq.as_mut_ptr(), b as i32);
    }
    tq
}

// Inner product functions
#[inline]
pub fn ip_bin_bin(q: &[u32], d: &[u32], b: usize) -> u32 {
    #[cfg(feature = "ffi")]
    unsafe {
        ffi::ip_bin_bin(q.as_ptr(), d.as_ptr(), b as i32)
    }
    #[cfg(not(feature = "ffi"))]
    {
        use std::arch::x86_64::*;

        let len = b.div_ceil(32); // 计算需要处理的u32数量
        let mut ret = 0u32;

        // 处理256位倍数部分 (8个u32为一组)
        let len_256 = len / 8;
        for i in 0..len_256 {
            unsafe {
                let v01 = _mm256_loadu_si256(q.as_ptr().add(i * 8) as *const __m256i);
                let v02 = _mm256_loadu_si256(d.as_ptr().add(i * 8) as *const __m256i);
                let r0 = _mm256_and_si256(v01, v02);
                let r0 = _mm256_popcnt_epi64(r0);

                let sum = _mm256_add_epi64(r0, _mm256_setzero_si256());
                let u64_t: [u64; 4] = std::mem::transmute(sum);
                ret += (u64_t[0] + u64_t[1] + u64_t[2] + u64_t[3]) as u32;
            }
        }

        // 处理128位部分 (4个u32为一组)
        let remaining_after_256 = len - (len_256 * 8);
        let len_128 = remaining_after_256 / 4;
        for i in 0..len_128 {
            unsafe {
                let offset = len_256 * 8 + i * 4;
                let v01 = _mm_lddqu_si128(q.as_ptr().add(offset) as *const __m128i);
                let v02 = _mm_lddqu_si128(d.as_ptr().add(offset) as *const __m128i);
                let r0 = _mm_and_si128(v01, v02);
                let r0 = _mm_popcnt_epi64(r0);

                let sum = _mm_add_epi64(r0, _mm_setzero_si128());
                let u64_t: [u64; 2] = std::mem::transmute(sum);
                ret += (u64_t[0] + u64_t[1]) as u32;
            }
        }

        // 处理剩余部分
        let remaining_after_128 = remaining_after_256 - (len_128 * 4);
        for i in 0..remaining_after_128 {
            let offset = len_256 * 8 + len_128 * 4 + i;
            ret += (q[offset] & d[offset]).count_ones();
        }

        ret
    }
}

pub fn ip_768bin_bin(q: &[u32], d: &[u32]) -> u32 {
    unsafe { ffi::ip_768bin_bin(q.as_ptr(), d.as_ptr(), 768) }
}

#[inline]
pub fn ip_128bin_bin(q: &[u32], d: &[u32]) -> u32 {
    unsafe {
        use std::arch::x86_64::*;

        // Load 512 bits (16 u32s) from q.
        let vq = _mm512_loadu_si512(q.as_ptr() as *const _);

        // Load the first 128 bits (4 u32s) from d.
        let d_128 = _mm_loadu_si128(d.as_ptr() as *const _);

        // Broadcast the 128 bits from d to all four 128-bit lanes of a 512-bit register.
        let vd_512 = _mm512_broadcast_i32x4(d_128);

        // Bitwise AND of the two 512-bit vectors.
        let vand = _mm512_and_si512(vq, vd_512);

        // Calculate the population count for each 64-bit lane.
        let vpop = _mm512_popcnt_epi64(vand);

        // Transmute the AVX512 register to an array of u64 to sum up the popcounts.
        let popcounts: [u64; 8] = std::mem::transmute(vpop);

        // Sum popcounts for each original 128-bit chunk of q.
        let r1 = popcounts[0] + popcounts[1];
        let r2 = popcounts[2] + popcounts[3];
        let r3 = popcounts[4] + popcounts[5];
        let r4 = popcounts[6] + popcounts[7];

        (r1 + (r2 << 1) + (r3 << 2) + (r4 << 3)) as u32
    }
}

pub fn ip_byte_bin_256x(q: &[u32], d: &[u32], b: usize) -> u32 {
    unsafe { ffi::ip_byte_bin_256x(q.as_ptr(), d.as_ptr(), b as i32) }
}

pub fn ip_byte_bin_128x(q: &[u32], d: &[u32], b: usize) -> u32 {
    #[cfg(feature = "ffi")]
    {
        unsafe { ffi::ip_byte_bin_128x(q.as_ptr(), d.as_ptr(), b as i32) }
    }
    #[cfg(not(feature = "ffi"))]
    {
        unsafe {
            use std::arch::x86_64::*;
            let mut sum = _mm_setzero_si128();

            // 32*4=128位，每次处理4个u32
            for i in (0..b / 32).step_by(4) {
                if i + 3 < q.len() && i + 3 < d.len() {
                    let v01 = _mm_lddqu_si128(q.as_ptr().add(i) as *const __m128i);
                    let v02 = _mm_lddqu_si128(d.as_ptr().add(i) as *const __m128i);
                    let mut r0 = _mm_and_si128(v01, v02);

                    r0 = _mm_popcnt_epi64(r0);
                    sum = _mm_add_epi64(sum, r0);
                }
            }

            // 将sum中的两个64位值相加得到最终结果
            let u64_t: [u64; 2] = std::mem::transmute(sum);
            (u64_t[0] + u64_t[1]) as u32
        }
    }
}

#[inline]
#[cfg(not(feature = "ffi"))]
pub fn ip_byte_bin(q: &[u32], d: &[u32], b: usize, x_256: bool, x_128: bool) -> u32 {
    let mut ret = 0u32;

    if b == 768 {
        ret += ip_768bin_bin(q, d);
        ret += ip_768bin_bin(&q[b / 32..], d) << 1;
        ret += ip_768bin_bin(&q[2 * (b / 32)..], d) << 2;
        ret += ip_768bin_bin(&q[3 * (b / 32)..], d) << 3;
    } else if b == 128 {
        ret += ip_128bin_bin(q, d);
    } else if x_256 {
        ret += ip_byte_bin_256x(q, d, b);
        ret += ip_byte_bin_256x(&q[b / 32..], d, b) << 1;
        ret += ip_byte_bin_256x(&q[2 * (b / 32)..], d, b) << 2;
        ret += ip_byte_bin_256x(&q[3 * (b / 32)..], d, b) << 3;
    } else if x_128 {
        ret += ip_byte_bin_128x(q, d, b);
        ret += ip_byte_bin_128x(&q[b / 32..], d, b) << 1;
        ret += ip_byte_bin_128x(&q[2 * (b / 32)..], d, b) << 2;
        ret += ip_byte_bin_128x(&q[3 * (b / 32)..], d, b) << 3;
    } else {
        // Default path - B_QUERY assumed to be 4 based on the C++ code
        for i in 0..4 {
            ret += ip_bin_bin(&q[i * (b / 32)..], d, b) << i;
        }
    }

    ret
}

#[inline]
#[cfg(feature = "ffi")]
pub fn ip_byte_bin(q: &[u32], d: &[u32], b: usize, x_256: bool, x_128: bool) -> u32 {
    unsafe { ffi::ip_byte_bin(q.as_ptr(), d.as_ptr(), b as i32, x_256, x_128) }
}

// xb_qprime function
pub fn xb_qprime(query: &[f32], xb: &[u32], b: usize) -> f32 {
    #[cfg(feature = "ffi")]
    {
        unsafe { ffi::xb_qprime(query.as_ptr(), xb.as_ptr(), b) }
    }
    #[cfg(not(feature = "ffi"))]
    {
        let d = b.div_ceil(32);
        let mask = 0b11111111u8;
        let mut pos = 0;

        unsafe {
            use std::arch::x86_64::*;

            let mut sum = _mm256_setzero_ps();

            for &uxb in xb.iter().take(d) {
                let b0_7 = uxb as u8 & mask;
                let b8_15 = (uxb >> 8) as u8 & mask;
                let b16_23 = (uxb >> 16) as u8 & mask;
                let b24_31 = (uxb >> 24) as u8 & mask;

                let q0_7 = _mm256_loadu_ps(query.as_ptr().add(pos));
                let q8_15 = _mm256_loadu_ps(query.as_ptr().add(pos + 8));
                let q16_23 = _mm256_loadu_ps(query.as_ptr().add(pos + 16));
                let q24_31 = _mm256_loadu_ps(query.as_ptr().add(pos + 24));

                sum = _mm256_mask_add_ps(sum, b0_7, q0_7, sum);
                sum = _mm256_mask_add_ps(sum, b8_15, q8_15, sum);
                sum = _mm256_mask_add_ps(sum, b16_23, q16_23, sum);
                sum = _mm256_mask_add_ps(sum, b24_31, q24_31, sum);

                pos += 32;
            }

            let mut r = [0.0f32; 8];
            _mm256_storeu_ps(r.as_mut_ptr(), sum);

            r.iter().sum()
        }
    }
}

pub fn ip32_fxu4(query: &[f32], y: &[u8], d: usize) -> f32 {
    #[cfg(feature = "ffi")]
    {
        unsafe { ffi::IP32_fxu4(query.as_ptr(), y.as_ptr(), d) }
    }
    #[cfg(not(feature = "ffi"))]
    {
        unsafe {
            use std::arch::x86_64::*;

            let mask = _mm_set1_epi8(0b1111);
            let mut sum = _mm512_setzero_ps();

            for i in (0..d).step_by(32) {
                let a8 = _mm_loadu_si128(y.as_ptr().add(i / 2) as *const __m128i);
                let mut b8 = a8;
                let x1 = _mm512_loadu_ps(query.as_ptr().add(i));
                let x2 = _mm512_loadu_ps(query.as_ptr().add(i + 16));

                // get lower(0 to 15) and upper(16 to 31) 4 bits
                let a8 = _mm_and_si128(a8, mask);
                b8 = _mm_and_si128(_mm_srli_epi16(b8, 4), mask);

                let af = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a8));
                sum = _mm512_fmadd_ps(af, x1, sum);
                let bf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b8));
                sum = _mm512_fmadd_ps(bf, x2, sum);
            }

            _mm512_reduce_add_ps(sum)
        }
    }
}

// simd_residual_query function
pub fn simd_residual_query(
    processed_query: &[f32],
    random_cent_vec: &[f32],
    query_residual: &mut [f32],
    ndims: usize,
) -> (f32, f32) {
    {
        unsafe {
            use std::arch::x86_64::*;

            // 确保输入维度是16的倍数
            assert!(ndims.is_multiple_of(16), "ndims must be a multiple of 16");

            let mut l2dist_to_cent = _mm512_setzero_ps();
            let mut sum_residual = _mm512_setzero_ps();

            for d in (0..ndims).step_by(16) {
                let q_0 = _mm512_loadu_ps(processed_query.as_ptr().add(d));
                let c_0 = _mm512_loadu_ps(random_cent_vec.as_ptr().add(d));

                let r_0 = _mm512_sub_ps(q_0, c_0);

                l2dist_to_cent = _mm512_fmadd_ps(r_0, r_0, l2dist_to_cent);

                sum_residual = _mm512_add_ps(sum_residual, r_0);

                _mm512_storeu_ps(query_residual.as_mut_ptr().add(d), r_0);
            }

            let query_square_dist_to_centroid = _mm512_reduce_add_ps(l2dist_to_cent);
            let sum_longq = _mm512_reduce_add_ps(sum_residual);

            (query_square_dist_to_centroid, sum_longq)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{ffi, xb_qprime};

    #[test]
    fn test_xb_qprime_consistency() {
        // 测试数据
        let query: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,
        ];
        let xb: Vec<u32> = vec![
            0b10101010101010101010101010101010,
            0b01010101010101010101010101010101,
        ];
        let b = 64; // 位数

        // Rust实现结果
        let rust_result = xb_qprime(&query, &xb, b);

        // C++实现结果（通过FFI调用）
        let cpp_result = unsafe { ffi::xb_qprime(query.as_ptr(), xb.as_ptr(), b) };

        // 验证两者结果是否一致（考虑浮点数精度误差）
        assert!(
            (rust_result - cpp_result).abs() < 1e-6,
            "Rust result ({}) and C++ result ({}) should be equal",
            rust_result,
            cpp_result
        );
    }

    #[test]
    fn test_xb_qprime_with_different_data() {
        // 测试数据
        let query: Vec<f32> = vec![1.0; 32]; // 32个1.0
        let xb: Vec<u32> = vec![0xFFFFFFFF, 0x00000000, 0xAAAAAAAA, 0x55555555];
        let b = 128; // 位数

        // Rust实现结果
        let rust_result = xb_qprime(&query, &xb, b);

        // C++实现结果（通过FFI调用）
        let cpp_result = unsafe { ffi::xb_qprime(query.as_ptr(), xb.as_ptr(), b) };

        // 验证两者结果是否一致（考虑浮点数精度误差）
        assert!(
            (rust_result - cpp_result).abs() < 1e-6,
            "Rust result ({}) and C++ result ({}) should be equal",
            rust_result,
            cpp_result
        );
    }

    #[test]
    fn test_xb_qprime_edge_case() {
        // 测试边界情况
        let query: Vec<f32> = vec![0.0; 32]; // 32个0.0
        let xb: Vec<u32> = vec![0x00000000; 4]; // 全0
        let b = 128; // 位数

        // Rust实现结果
        let rust_result = xb_qprime(&query, &xb, b);

        // C++实现结果（通过FFI调用）
        let cpp_result = unsafe { ffi::xb_qprime(query.as_ptr(), xb.as_ptr(), b) };

        // 验证两者结果是否一致（考虑浮点数精度误差）
        assert!(
            (rust_result - cpp_result).abs() < 1e-6,
            "Rust result ({}) and C++ result ({}) should be equal",
            rust_result,
            cpp_result
        );
        // 都为0时结果应该为0
        assert!(
            rust_result.abs() < 1e-6,
            "Result should be 0 when query or xb is all zeros"
        );
    }
}
