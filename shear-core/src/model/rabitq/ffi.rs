use std::ffi::c_uchar;

use libc::{c_double, c_float, c_int, c_schar, c_uint, c_ulonglong, size_t};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphRabitq1Estimate {
    pub est_dist: c_float,
    pub low_dist: c_float,
}

// Range functions
unsafe extern "C" {
    pub fn rangeSIMD(
        q: *const c_float,
        c: *const c_float,
        vl: *mut c_float,
        vr: *mut c_float,
        b: c_int,
    );
    pub fn rangeCommon(
        q: *const c_float,
        c: *const c_float,
        vl: *mut c_float,
        vr: *mut c_float,
        b: c_int,
    );
    pub fn range(
        q: *const c_float,
        c: *const c_float,
        vl: *mut c_float,
        vr: *mut c_float,
        b: c_int,
    );
}

// Quantize functions
unsafe extern "C" {
    pub fn quantizeSIMD(
        result: *mut c_uchar,
        q: *const c_float,
        c: *const c_float,
        u: *const c_float,
        vl: c_float,
        width: c_float,
        sum_q: *mut c_uint,
        b: c_int,
    );
    pub fn quantize(
        result: *mut c_uchar,
        q: *const c_float,
        c: *const c_float,
        u: *const c_float,
        vl: c_float,
        width: c_float,
        sum_q: *mut c_uint,
        b: c_int,
    );
}

// Binary operations
unsafe extern "C" {
    pub fn transpose_bin(q: *const c_uchar, tq: *mut c_uint, b: c_int);
    pub fn ip_bin_bin(q: *const c_uint, d: *const c_uint, b: c_int) -> c_uint;
    pub fn ip_768bin_bin(q: *const c_uint, d: *const c_uint, b: c_int) -> c_uint;
    pub fn ip_128bin_bin(q: *const c_uint, d: *const c_uint, b: c_int) -> c_uint;
    pub fn ip_byte_bin_256x(q: *const c_uint, d: *const c_uint, b: c_int) -> c_uint;
    pub fn ip_byte_bin_128x(q: *const c_uint, d: *const c_uint, b: c_int) -> c_uint;
    pub fn ip_byte_bin(
        q: *const c_uint,
        d: *const c_uint,
        b: c_int,
        x_256: bool,
        x_128: bool,
    ) -> c_uint;

    // New functions
    pub fn fast_quantize(
        o_prime: *const c_float,
        code: *mut c_uchar,
        ip_norm: *mut c_float,
        max_bd: c_uint,
    );
    pub fn rabitq_plus_quantize_dense_fht_kac_simd(
        data: *const c_float,
        rows: size_t,
        dim: size_t,
        total_bits: c_int,
        t_const: c_double,
        seed: c_ulonglong,
        codes: *mut c_schar,
        deltas: *mut c_float,
        code_sums: *mut c_float,
    ) -> c_int;
    pub fn rabitq_plus_fast_t_const(dim: size_t, total_bits: c_int) -> c_double;
    pub fn rabitq_split_batch_data_bytes(padded_dim: size_t) -> size_t;
    pub fn rabitq_split_ex_data_bytes(padded_dim: size_t, ex_bits: c_int) -> size_t;
    pub fn rabitq_split_quantize_fht_kac_simd(
        data: *const c_float,
        rows: size_t,
        dim: size_t,
        total_bits: c_int,
        t_const: c_double,
        seed: c_ulonglong,
        batch_data: *mut c_uchar,
        ex_data: *mut c_uchar,
    ) -> c_int;
    pub fn rabitq_split_d0_assign_fht_kac_simd(
        batch_data: *const c_uchar,
        ex_data: *const c_uchar,
        rows: size_t,
        dim: size_t,
        total_bits: c_int,
        leaders: *const c_float,
        leader_count: size_t,
        seed: c_ulonglong,
        fanout: size_t,
        survivors: size_t,
        assignments: *mut u16,
        coarse_seconds: *mut c_double,
        boost_seconds: *mut c_double,
        boosted_pairs: *mut u64,
    ) -> c_int;
    pub fn graph_rabitq1_padded_dim(dim: size_t) -> size_t;
    pub fn graph_rabitq1_rotator_bytes(padded_dim: size_t) -> size_t;
    pub fn graph_rabitq1_bin_data_bytes(padded_dim: size_t) -> size_t;
    pub fn graph_rabitq1_generate_rotator(
        dim: size_t,
        seed: c_ulonglong,
        output: *mut c_uchar,
        output_bytes: size_t,
    ) -> c_int;
    pub fn graph_rabitq1_last_error() -> *const libc::c_char;
    pub fn graph_rabitq1_rotate_vectors(
        dim: size_t,
        rotator: *const c_uchar,
        rotator_bytes: size_t,
        input: *const c_float,
        rows: size_t,
        output: *mut c_float,
    ) -> c_int;
    pub fn graph_rabitq1_codec_create(
        dim: size_t,
        rotator: *const c_uchar,
        rotator_bytes: size_t,
        rotated_centers: *const c_float,
        center_count: size_t,
        output: *mut *mut libc::c_void,
    ) -> c_int;
    pub fn graph_rabitq1_codec_destroy(codec: *mut libc::c_void);
    pub fn graph_rabitq1_quantize(
        codec: *mut libc::c_void,
        input: *const c_float,
        rows: size_t,
        threads: size_t,
        center_ids: *mut c_uchar,
        bin_data: *mut c_uchar,
        bin_data_bytes: size_t,
    ) -> c_int;
    pub fn graph_rabitq1_query_create(
        codec: *mut libc::c_void,
        query: *const c_float,
        query_dim: size_t,
        output: *mut *mut libc::c_void,
    ) -> c_int;
    pub fn graph_rabitq1_query_destroy(query: *mut libc::c_void);
    pub fn graph_rabitq1_estimate_ids(
        query: *mut libc::c_void,
        center_ids: *const c_uchar,
        bin_data: *const c_uchar,
        rows: size_t,
        bin_stride: size_t,
        ids: *const c_uint,
        id_count: size_t,
        output: *mut GraphRabitq1Estimate,
    ) -> c_int;
    pub fn graph_rabitq1_reference_estimate(
        query: *mut libc::c_void,
        center_id: c_uchar,
        bin_data: *const c_uchar,
        bin_bytes: size_t,
        output: *mut GraphRabitq1Estimate,
    ) -> c_int;
    pub fn store_compacted_code(o_raw: *const c_uchar, o_compact: *mut c_uchar, d: size_t);

    // Additional functions
    pub fn IP32_fxu4(x: *const c_float, y: *const c_uchar, d: size_t) -> c_float;
    pub fn xb_qprime(query: *const c_float, xb: *const c_uint, b: size_t) -> c_float;

    // Matrix functions
    pub fn float_randMatrix(x: *mut c_float, n: size_t);
    pub fn matrix_qr(m: c_int, n: c_int, a: *mut c_float);

    // RabitQ functions
    pub fn doGetXP(
        data_file: *const libc::c_char,
        max_bd: size_t,
        matrix: *const c_float,
        num_centers: size_t,
        pivot_data: *const c_float,
        xp: *mut c_float,
        cp: *mut c_float,
    );
}
