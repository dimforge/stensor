use nalgebra::{Matrix3, Vector3};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::Shader;

#[derive(Copy, Clone, Debug)]
/// Result of a GPU SVD3 computation.
pub struct GpuSvd3 {
    /// First orthogonal matrix of the SVD.
    pub u: Matrix3<f32>,
    /// Singular values (descending order).
    pub s: Vector3<f32>,
    /// Transposed right singular vectors.
    pub vt: Matrix3<f32>,
}

#[derive(Shader)]
#[shader(module = "stensor::geometry::svd_stable::test_svd3")]
/// Test shader for SVD3.
pub struct Svd3Shader<B: Backend> {
    /// The compute function for testing SVD3.
    pub test_svd3: GpuFunction<B>,
}

#[cfg(test)]
mod test {
    use super::GpuSvd3;
    use crate::tensor::GpuTensor;
    use minislang::SlangCompiler;
    use nalgebra::Matrix3;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use slang_hal::{BufferUsages, ShaderArgs, Shader};

    #[derive(ShaderArgs)]
    struct Svd3Args<'a, B: Backend> {
        inputs: &'a B::Buffer<f32>,
        outputs: &'a B::Buffer<f32>,
    }

    // -----------------------------------------------------------------------
    // Helpers (ported from minisvd tests)
    // -----------------------------------------------------------------------

    fn approx_eq_mat3_rel(a: &Matrix3<f32>, b: &Matrix3<f32>, eps: f32) -> bool {
        let scale_a = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale_b = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = scale_a.max(scale_b).max(1e-10);
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() < eps * scale)
    }

    fn assert_valid_svd(m: &Matrix3<f32>, svd: &GpuSvd3, recon_rel_eps: f32, ortho_eps: f32, label: &str) {
        // Singular values non-negative and descending.
        assert!(
            svd.s.x >= -1e-6,
            "{label}: s0 negative: {}", svd.s.x
        );
        assert!(
            svd.s.y >= -1e-6,
            "{label}: s1 negative: {}", svd.s.y
        );
        assert!(
            svd.s.z >= -1e-6,
            "{label}: s2 negative: {}", svd.s.z
        );
        assert!(
            svd.s.x + 1e-6 >= svd.s.y,
            "{label}: s0 < s1: {} < {}", svd.s.x, svd.s.y
        );
        assert!(
            svd.s.y + 1e-6 >= svd.s.z,
            "{label}: s1 < s2: {} < {}", svd.s.y, svd.s.z
        );

        // Reconstruction.
        let reconstructed = svd.u * Matrix3::from_diagonal(&svd.s) * svd.vt;
        assert!(
            approx_eq_mat3_rel(m, &reconstructed, recon_rel_eps),
            "{label}: reconstruction failed\n  original:      {m:?}\n  reconstructed: {reconstructed:?}"
        );

        // U orthogonality.
        let utu = svd.u.transpose() * svd.u;
        assert!(
            approx_eq_mat3_rel(&utu, &Matrix3::identity(), ortho_eps),
            "{label}: U not orthogonal\n  U^T*U: {utu:?}"
        );

        // V orthogonality.
        let vtv = svd.vt * svd.vt.transpose();
        assert!(
            approx_eq_mat3_rel(&vtv, &Matrix3::identity(), ortho_eps),
            "{label}: V not orthogonal\n  V^T*V: {vtv:?}"
        );
    }

    fn assert_matches_nalgebra(m: &Matrix3<f32>, svd: &GpuSvd3, rel_eps: f32, label: &str) {
        let na_svd = m.svd(true, true);
        let na_sv = na_svd.singular_values;
        let scale = na_sv[0].max(1e-10);

        assert!(
            (svd.s.x - na_sv[0]).abs() < rel_eps * scale,
            "{label}: s0 mismatch: {} vs {} (scale={scale})", svd.s.x, na_sv[0]
        );
        assert!(
            (svd.s.y - na_sv[1]).abs() < rel_eps * scale,
            "{label}: s1 mismatch: {} vs {} (scale={scale})", svd.s.y, na_sv[1]
        );
        assert!(
            (svd.s.z - na_sv[2]).abs() < rel_eps * scale,
            "{label}: s2 mismatch: {} vs {} (scale={scale})", svd.s.z, na_sv[2]
        );
    }

    /// Minimal xorshift32 PRNG.
    struct Rng(u32);

    impl Rng {
        fn new(seed: u32) -> Self {
            Self(seed)
        }

        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }

        fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next_u32() as f64) / (u32::MAX as f64);
            lo + (hi - lo) * t as f32
        }

        /// Random 3x3 matrix (column-major) with elements in [lo, hi).
        fn next_mat3(&mut self, lo: f32, hi: f32) -> Matrix3<f32> {
            let mut arr = [0.0f32; 9];
            for v in &mut arr {
                *v = self.next_f32_range(lo, hi);
            }
            Matrix3::from_column_slice(&arr)
        }
    }

    // -----------------------------------------------------------------------
    // GPU dispatch helpers
    // -----------------------------------------------------------------------

    /// Packs matrices into a flat f32 buffer (9 floats per matrix, column-major).
    fn pack_matrices(matrices: &[Matrix3<f32>]) -> Vec<f32> {
        let mut data = Vec::with_capacity(matrices.len() * 9);
        for m in matrices {
            for &v in m.as_slice() {
                data.push(v);
            }
        }
        data
    }

    /// Unpacks SVD results from a flat f32 buffer (21 floats per result).
    fn unpack_svd_results(data: &[f32], count: usize) -> Vec<GpuSvd3> {
        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            let base = i * 21;
            let u = Matrix3::from_column_slice(&data[base..base + 9]);
            let s = nalgebra::Vector3::new(data[base + 9], data[base + 10], data[base + 11]);
            let vt = Matrix3::from_column_slice(&data[base + 12..base + 21]);
            results.push(GpuSvd3 { u, s, vt });
        }
        results
    }

    /// Runs SVD3 on the GPU for the given matrices and returns the results.
    async fn run_gpu_svd3(backend: &impl Backend, matrices: &[Matrix3<f32>]) -> Vec<GpuSvd3> {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        let shader = super::Svd3Shader::from_backend(backend, &compiler).unwrap();

        let input_data = pack_matrices(matrices);
        let count = matrices.len();

        let gpu_inputs =
            GpuTensor::vector(backend, &input_data, BufferUsages::STORAGE).unwrap();
        let output_len = count * 21;
        let gpu_outputs = GpuTensor::<f32, _>::vector(
            backend,
            &vec![0.0f32; output_len],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test_svd3", None);
        let args = Svd3Args {
            inputs: gpu_inputs.buffer(),
            outputs: gpu_outputs.buffer(),
        };
        shader
            .test_svd3
            .launch(backend, &mut pass, &args, [count as u32, 1, 1])
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut output_data = vec![0.0f32; output_len];
        backend
            .slow_read_buffer(gpu_outputs.buffer(), &mut output_data)
            .await
            .unwrap();

        unpack_svd_results(&output_data, count)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_svd3_webgpu() {
        let backend = WebGpu::default().await.unwrap();
        gpu_svd3_all(&backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    async fn gpu_svd3_cuda() {
        let backend = slang_hal::cuda::Cuda::new().unwrap();
        gpu_svd3_all(&backend).await;
    }

    async fn gpu_svd3_all(backend: &impl Backend) {
        gpu_svd3_deterministic(backend).await;
        gpu_svd3_random_unit_range(backend).await;
        gpu_svd3_random_large_range(backend).await;
        gpu_svd3_random_small_values(backend).await;
        gpu_svd3_random_positive_definite(backend).await;
        gpu_svd3_random_symmetric(backend).await;
        gpu_svd3_random_rank_deficient(backend).await;
        gpu_svd3_random_near_singular(backend).await;
        gpu_svd3_random_negative_determinant(backend).await;
    }

    async fn gpu_svd3_deterministic(backend: &impl Backend) {
        use nalgebra::Vector3;

        let identity = Matrix3::identity();
        let diagonal = Matrix3::from_columns(&[
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ]);
        let symmetric = Matrix3::from_columns(&[
            Vector3::new(2.0, 1.0, 0.0),
            Vector3::new(1.0, 3.0, 1.0),
            Vector3::new(0.0, 1.0, 2.0),
        ]);
        let general = Matrix3::from_columns(&[
            Vector3::new(1.0, 4.0, 7.0),
            Vector3::new(2.0, 5.0, 8.0),
            Vector3::new(3.0, 6.0, 10.0),
        ]);
        let zero = Matrix3::zeros();

        let v = Vector3::new(1.0, 2.0, 3.0);
        let w = Vector3::new(4.0, 5.0, 6.0);
        let rank_one = Matrix3::from_columns(&[v * w.x, v * w.y, v * w.z]);

        let mixed_sign = Matrix3::from_columns(&[
            Vector3::new(0.5, -1.2, 3.7),
            Vector3::new(2.1, 0.3, -0.8),
            Vector3::new(-1.0, 4.5, 2.2),
        ]);
        let large_values = Matrix3::from_columns(&[
            Vector3::new(100.0, 0.01, -50.0),
            Vector3::new(0.01, 200.0, 30.0),
            Vector3::new(-50.0, 30.0, 150.0),
        ]);

        let matrices = vec![identity, diagonal, symmetric, general, zero, rank_one, mixed_sign, large_values];
        let labels = ["identity", "diagonal", "symmetric", "general", "zero", "rank_one", "mixed_sign", "large_values"];

        let results = run_gpu_svd3(backend, &matrices).await;

        // Identity: singular values should all be 1.
        assert!((results[0].s.x - 1.0).abs() < 1e-6, "identity s0: {}", results[0].s.x);
        assert!((results[0].s.y - 1.0).abs() < 1e-6, "identity s1: {}", results[0].s.y);
        assert!((results[0].s.z - 1.0).abs() < 1e-6, "identity s2: {}", results[0].s.z);

        // Diagonal: singular values should be 3, 2, 1.
        assert!((results[1].s.x - 3.0).abs() < 1e-6, "diagonal s0: {}", results[1].s.x);
        assert!((results[1].s.y - 2.0).abs() < 1e-6, "diagonal s1: {}", results[1].s.y);
        assert!((results[1].s.z - 1.0).abs() < 1e-6, "diagonal s2: {}", results[1].s.z);

        // Zero: singular values should all be 0.
        assert!(results[4].s.x.abs() < 1e-6, "zero s0: {}", results[4].s.x);
        assert!(results[4].s.y.abs() < 1e-6, "zero s1: {}", results[4].s.y);
        assert!(results[4].s.z.abs() < 1e-6, "zero s2: {}", results[4].s.z);

        // Rank one: only first singular value should be large.
        assert!(results[5].s.x > 1e-3, "rank_one s0: {}", results[5].s.x);
        assert!(results[5].s.y < 1e-4, "rank_one s1: {}", results[5].s.y);
        assert!(results[5].s.z < 1e-4, "rank_one s2: {}", results[5].s.z);

        // All except zero: validate SVD properties.
        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            if labels[i] == "zero" {
                continue;
            }
            assert_valid_svd(m, svd, 1e-4, 1e-4, labels[i]);
        }

        // Cross-validate with nalgebra.
        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            if labels[i] == "zero" {
                continue;
            }
            assert_matches_nalgebra(m, svd, 1e-4, labels[i]);
        }
    }

    async fn gpu_svd3_random_unit_range(backend: &impl Backend) {
        let mut rng = Rng::new(0xDEAD_BEEF);
        let matrices: Vec<_> = (0..1000).map(|_| rng.next_mat3(-1.0, 1.0)).collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_unit_{i}");
            assert_valid_svd(m, svd, 1e-4, 1e-4, &label);
            assert_matches_nalgebra(m, svd, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_large_range(backend: &impl Backend) {
        let mut rng = Rng::new(0xCAFE_1234);
        let matrices: Vec<_> = (0..1000).map(|_| rng.next_mat3(-100.0, 100.0)).collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_large_{i}");
            assert_valid_svd(m, svd, 1e-4, 1e-4, &label);
            assert_matches_nalgebra(m, svd, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_small_values(backend: &impl Backend) {
        let mut rng = Rng::new(0xBAAD_F00D);
        let matrices: Vec<_> = (0..1000).map(|_| rng.next_mat3(-1e-3, 1e-3)).collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_small_{i}");
            assert_valid_svd(m, svd, 1e-4, 1e-4, &label);
            assert_matches_nalgebra(m, svd, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_positive_definite(backend: &impl Backend) {
        let mut rng = Rng::new(0x1234_ABCD);
        let matrices: Vec<_> = (0..500)
            .map(|_| {
                let a = rng.next_mat3(-5.0, 5.0);
                a.transpose() * a
            })
            .collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_posdef_{i}");
            assert_valid_svd(m, svd, 1e-3, 1e-4, &label);
            assert_matches_nalgebra(m, svd, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_symmetric(backend: &impl Backend) {
        let mut rng = Rng::new(0xFEED_FACE);
        let matrices: Vec<_> = (0..500)
            .map(|_| {
                let a = rng.next_mat3(-10.0, 10.0);
                (a + a.transpose()) * 0.5
            })
            .collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_symmetric_{i}");
            assert_valid_svd(m, svd, 1e-4, 1e-4, &label);
            assert_matches_nalgebra(m, svd, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_rank_deficient(backend: &impl Backend) {
        let mut rng = Rng::new(0xABCD_EF01);
        let matrices: Vec<_> = (0..500)
            .map(|_| {
                let c0 = nalgebra::Vector3::new(
                    rng.next_f32_range(-5.0, 5.0),
                    rng.next_f32_range(-5.0, 5.0),
                    rng.next_f32_range(-5.0, 5.0),
                );
                let c1 = nalgebra::Vector3::new(
                    rng.next_f32_range(-5.0, 5.0),
                    rng.next_f32_range(-5.0, 5.0),
                    rng.next_f32_range(-5.0, 5.0),
                );
                let alpha = rng.next_f32_range(-3.0, 3.0);
                let beta = rng.next_f32_range(-3.0, 3.0);
                let c2 = c0 * alpha + c1 * beta;
                Matrix3::from_columns(&[c0, c1, c2])
            })
            .collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_rank2_{i}");
            assert!(
                svd.s.z < 1e-3,
                "{label}: expected near-zero s2, got {}", svd.s.z
            );
            assert_valid_svd(m, svd, 1e-3, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_near_singular(backend: &impl Backend) {
        let mut rng = Rng::new(0x1111_2222);
        let matrices: Vec<_> = (0..500)
            .map(|_| {
                let mut arr = [0.0f32; 9];
                arr[0] = 1.0;
                arr[4] = 1.0;
                arr[8] = 1.0;
                for v in &mut arr {
                    *v += rng.next_f32_range(-0.01, 0.01);
                }
                arr[6] = arr[0] * rng.next_f32_range(-0.001, 0.001)
                    + arr[3] * rng.next_f32_range(-0.001, 0.001);
                arr[7] = arr[1] * rng.next_f32_range(-0.001, 0.001)
                    + arr[4] * rng.next_f32_range(-0.001, 0.001);
                arr[8] = arr[2] * rng.next_f32_range(-0.001, 0.001)
                    + arr[5] * rng.next_f32_range(-0.001, 0.001);
                Matrix3::from_column_slice(&arr)
            })
            .collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_near_singular_{i}");
            assert_valid_svd(m, svd, 1e-4, 1e-4, &label);
        }
    }

    async fn gpu_svd3_random_negative_determinant(backend: &impl Backend) {
        let mut rng = Rng::new(0x5555_AAAA);
        let matrices: Vec<_> = (0..500)
            .map(|_| {
                let mut m = rng.next_mat3(-5.0, 5.0);
                m.set_column(0, &(-m.column(0)));
                m
            })
            .collect();
        let results = run_gpu_svd3(backend, &matrices).await;

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            let label = format!("random_negdet_{i}");
            assert_valid_svd(m, svd, 1e-4, 1e-4, &label);
            assert_matches_nalgebra(m, svd, 1e-4, &label);
        }
    }
}
