use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::mem;
use std::path::Path;

use rand::distr::{Distribution, Uniform};

use super::CachedReader;
use crate::common::{AnnError, AnnResult};

/// streams data from the file, and samples each vector with probability p_val
/// and returns a matrix of size slice_size* ndims as floating point type.
/// the slice_size and ndims are set inside the function.
/// # Arguments
/// * `file_name` - filename where the data is
/// * `p_val` - possibility to sample data
/// * `sampled_vectors` - sampled vector chose by p_val possibility
/// * `slice_size` - how many sampled data return
/// * `dim` - each sample data dimension
pub fn gen_random_slice<T: Default + Copy + Into<f32>>(
    data_file: &Path,
    mut p_val: f64,
) -> AnnResult<(Vec<f32>, usize, usize)> {
    let read_blk_size = 64 * 1024 * 1024;
    let mut reader = CachedReader::new(data_file, read_blk_size)?;

    let npts = reader.read_u32()? as usize;
    let dim = reader.read_u32()? as usize;
    let mut sampled_vectors: Vec<f32> = Vec::new();
    let mut slice_size = 0;
    p_val = if p_val < 1f64 { p_val } else { 1f64 };

    let mut generator = rand::rng();
    let distribution =
        Uniform::new(0.0, 1.0).map_err(|e| AnnError::log_index_error(e.to_string()))?;

    for _ in 0..npts {
        let mut cur_vector_bytes = vec![0u8; dim * mem::size_of::<T>()];
        reader.read(&mut cur_vector_bytes)?;
        let random_value = distribution.sample(&mut generator);
        if random_value < p_val {
            let ptr = cur_vector_bytes.as_ptr() as *const T;
            let cur_vector_t = unsafe { std::slice::from_raw_parts(ptr, dim) };
            sampled_vectors.extend(cur_vector_t.iter().map(|&t| t.into()));
            slice_size += 1;
        }
    }

    Ok((sampled_vectors, slice_size, dim))
}

/// Generate random sample data and write into output_file
pub fn gen_sample_data<T>(
    data_file: &Path,
    output_file: &Path,
    sampling_rate: f64,
) -> AnnResult<()> {
    let read_blk_size = 64 * 1024 * 1024;
    let mut reader = CachedReader::new(Path::new(data_file), read_blk_size)?;

    let sample_data_path = output_file.join("_data.bin");
    let sample_ids_path = output_file.join("_ids.bin");
    let mut sample_data_writer = File::create(Path::new(&sample_data_path))?;
    let mut sample_id_writer = File::create(Path::new(&sample_ids_path))?;

    let mut num_sampled_pts = 0u32;
    let one_const = 1u32;
    let mut generator = rand::rng();
    let distribution = Uniform::new(0.0, 1.0).unwrap();

    let npts_u32 = reader.read_u32()?;
    let dim_u32 = reader.read_u32()?;
    let dim = dim_u32 as usize;
    sample_data_writer.write_all(&num_sampled_pts.to_le_bytes())?;
    sample_data_writer.write_all(&dim_u32.to_le_bytes())?;
    sample_id_writer.write_all(&num_sampled_pts.to_le_bytes())?;
    sample_id_writer.write_all(&one_const.to_le_bytes())?;

    for id in 0..npts_u32 {
        let mut cur_row_bytes = vec![0u8; dim * mem::size_of::<T>()];
        reader.read(&mut cur_row_bytes)?;
        let random_value = distribution.sample(&mut generator);
        if random_value < sampling_rate {
            sample_data_writer.write_all(&cur_row_bytes)?;
            sample_id_writer.write_all(&id.to_le_bytes())?;
            num_sampled_pts += 1;
        }
    }

    sample_data_writer.seek(SeekFrom::Start(0))?;
    sample_data_writer.write_all(&num_sampled_pts.to_le_bytes())?;
    sample_id_writer.seek(SeekFrom::Start(0))?;
    sample_id_writer.write_all(&num_sampled_pts.to_le_bytes())?;
    println!(
        "Wrote {} points to sample file: {:?}",
        num_sampled_pts, sample_data_path
    );

    Ok(())
}

#[cfg(test)]
mod partition_test {
    use std::fs;
    use std::io::Read;

    use byteorder::{LittleEndian, ReadBytesExt};

    use super::*;
    use crate::utils::file_exists;

    #[test]
    fn gen_sample_data_test() {
        let file_name = "gen_sample_data_test.bin";
        // npoints=2, dim=8
        let data: [u8; 72] = [
            2, 0, 0, 0, 8, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40,
            0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00,
            0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41,
            0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41,
        ];
        std::fs::write(file_name, data).expect("Failed to write sample file");

        let sample_file_prefix = file_name.to_string() + "_sample";
        gen_sample_data::<f32>(
            Path::new(file_name),
            Path::new(sample_file_prefix.as_str()),
            1f64,
        )
        .unwrap();

        let sample_data_path = format!("{}_data.bin", sample_file_prefix);
        let sample_ids_path = format!("{}_ids.bin", sample_file_prefix);
        assert!(file_exists(Path::new(sample_data_path.as_str())));
        assert!(file_exists(Path::new(sample_ids_path.as_str())));

        let mut data_file_reader = File::open(sample_data_path.as_str()).unwrap();
        let mut ids_file_reader = File::open(sample_ids_path.as_str()).unwrap();

        let mut num_sampled_pts = data_file_reader.read_u32::<LittleEndian>().unwrap();
        assert_eq!(num_sampled_pts, 2);
        num_sampled_pts = ids_file_reader.read_u32::<LittleEndian>().unwrap();
        assert_eq!(num_sampled_pts, 2);

        let dim = data_file_reader.read_u32::<LittleEndian>().unwrap() as usize;
        assert_eq!(dim, 8);
        assert_eq!(ids_file_reader.read_u32::<LittleEndian>().unwrap(), 1);

        let mut start = 8;
        for i in 0..num_sampled_pts {
            let mut data_bytes = vec![0u8; dim * 4];
            data_file_reader.read_exact(&mut data_bytes).unwrap();
            assert_eq!(data_bytes, data[start..start + dim * 4]);

            let id = ids_file_reader.read_u32::<LittleEndian>().unwrap();
            assert_eq!(id, i);

            start += dim * 4;
        }

        fs::remove_file(file_name).expect("Failed to delete file");
        fs::remove_file(sample_data_path.as_str()).expect("Failed to delete file");
        fs::remove_file(sample_ids_path.as_str()).expect("Failed to delete file");
    }
}

/// Generate random slice from data file
/// # Arguments
/// * `data_file` - filename where the data is
/// * `p_val` - possibility to sample data
/// * `sampled_vectors` - sampled vector chose by p_val possibility
/// * `ndims` - dimension of data
pub fn gen_fixed_random_slice<T: Default + Copy + Into<f32> + Sized>(
    data_file: &Path,
    p_val: f64,
    sampled_vectors: &mut Vec<f32>,
    ndims: &mut usize,
) -> AnnResult<usize> {
    // amount to read in one shot
    let read_blk_size = 64 * 1024 * 1024;
    // create cached reader
    let mut base_reader = CachedReader::new(data_file, read_blk_size)?;

    // metadata: npts, ndims
    let npts32 = base_reader.read_u32()?;
    let ndims32 = base_reader.read_u32()?;
    let npts = npts32 as usize;
    *ndims = ndims32 as usize;

    // const MAX_RABITQ_TRAINING_SET_SIZE: u32 = 256000;
    // const MAX_RABITQ_TRAINING_SET_SIZE: u32 = 1000000;
    // let p_val = 1.0.min(MAX_RABITQ_TRAINING_SET_SIZE as f64 / (npts32 as f64));

    // Create distribution for random sampling
    let mut generator = rand::rng();
    let distribution =
        Uniform::new(0.0, 1.0).map_err(|e| AnnError::log_index_error(e.to_string()))?;

    // Buffer for reading vector data
    let mut cur_vector_bytes = vec![0u8; *ndims * mem::size_of::<T>()];

    for _i in 0..npts {
        // Read vector data
        base_reader.read(&mut cur_vector_bytes)?;

        // Generate random value for sampling
        let rnd_val = distribution.sample(&mut generator);

        if rnd_val < p_val {
            // Convert bytes to T type slice
            let ptr = cur_vector_bytes.as_ptr() as *const T;
            let cur_vector_t = unsafe { std::slice::from_raw_parts(ptr, *ndims) };

            // Add sampled vector to output
            sampled_vectors.extend(cur_vector_t.iter().map(|&t| t.into()));
        }
    }

    Ok(npts)
}
