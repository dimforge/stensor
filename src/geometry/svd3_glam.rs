use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::Shader;

#[derive(Shader)]
#[shader(module = "stensor::geometry::svd_glam::test_svd3")]
/// Test shader for the glamx-ported SVD3.
pub struct Svd3GlamShader<B: Backend> {
    /// The compute function for testing the glamx-ported SVD3.
    pub test_svd3: GpuFunction<B>,
}

#[cfg(test)]
mod test {
    use crate::tensor::GpuTensor;
    use minislang::SlangCompiler;
    use nalgebra::{Matrix3, Vector3};
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use slang_hal::{BufferUsages, Shader, ShaderArgs};

    #[derive(Copy, Clone, Debug)]
    struct GpuSvd3 {
        u: Matrix3<f32>,
        s: Vector3<f32>,
        vt: Matrix3<f32>,
    }

    #[derive(ShaderArgs)]
    struct Svd3Args<'a, B: Backend> {
        inputs: &'a B::Buffer<f32>,
        outputs: &'a B::Buffer<f32>,
    }

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
        assert!(svd.s.x >= -1e-6, "{label}: s0 negative: {}", svd.s.x);
        assert!(svd.s.y >= -1e-6, "{label}: s1 negative: {}", svd.s.y);
        assert!(svd.s.z >= -1e-6, "{label}: s2 negative: {}", svd.s.z);
        assert!(svd.s.x + 1e-6 >= svd.s.y, "{label}: s0 < s1: {} < {}", svd.s.x, svd.s.y);
        assert!(svd.s.y + 1e-6 >= svd.s.z, "{label}: s1 < s2: {} < {}", svd.s.y, svd.s.z);

        // Reconstruction: U * diag(S) * Vt == M.
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
            "{label}: V not orthogonal\n  V*V^T: {vtv:?}"
        );
    }

    // Invariants that hold for *every* matrix regardless of conditioning: U and Vt are
    // orthogonal and the singular values are non-negative and sorted descending.
    fn assert_svd_invariants(svd: &GpuSvd3, ortho_eps: f32, label: &str) {
        assert!(svd.s.x >= -1e-6, "{label}: s0 negative: {}", svd.s.x);
        assert!(svd.s.y >= -1e-6, "{label}: s1 negative: {}", svd.s.y);
        assert!(svd.s.z >= -1e-6, "{label}: s2 negative: {}", svd.s.z);
        assert!(svd.s.x + 1e-6 >= svd.s.y, "{label}: s0 < s1: {} < {}", svd.s.x, svd.s.y);
        assert!(svd.s.y + 1e-6 >= svd.s.z, "{label}: s1 < s2: {} < {}", svd.s.y, svd.s.z);

        let utu = svd.u.transpose() * svd.u;
        assert!(
            approx_eq_mat3_rel(&utu, &Matrix3::identity(), ortho_eps),
            "{label}: U not orthogonal\n  U^T*U: {utu:?}"
        );
        let vtv = svd.vt * svd.vt.transpose();
        assert!(
            approx_eq_mat3_rel(&vtv, &Matrix3::identity(), ortho_eps),
            "{label}: V not orthogonal\n  V*V^T: {vtv:?}"
        );
    }

    fn assert_matches_nalgebra(m: &Matrix3<f32>, svd: &GpuSvd3, rel_eps: f32, label: &str) {
        let na_svd = m.svd(true, true);
        let na_sv = na_svd.singular_values;
        let scale = na_sv[0].max(1e-10);
        assert!((svd.s.x - na_sv[0]).abs() < rel_eps * scale, "{label}: s0 {} vs {}", svd.s.x, na_sv[0]);
        assert!((svd.s.y - na_sv[1]).abs() < rel_eps * scale, "{label}: s1 {} vs {}", svd.s.y, na_sv[1]);
        assert!((svd.s.z - na_sv[2]).abs() < rel_eps * scale, "{label}: s2 {} vs {}", svd.s.z, na_sv[2]);
    }

    /// Minimal xorshift32 PRNG.
    struct Rng(u32);
    impl Rng {
        fn new(seed: u32) -> Self { Self(seed) }
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
        fn next_mat3(&mut self, lo: f32, hi: f32) -> Matrix3<f32> {
            let mut arr = [0.0f32; 9];
            for v in &mut arr { *v = self.next_f32_range(lo, hi); }
            Matrix3::from_column_slice(&arr)
        }
    }

    fn pack_matrices(matrices: &[Matrix3<f32>]) -> Vec<f32> {
        let mut data = Vec::with_capacity(matrices.len() * 9);
        for m in matrices {
            for &v in m.as_slice() { data.push(v); }
        }
        data
    }

    fn unpack_svd_results(data: &[f32], count: usize) -> Vec<GpuSvd3> {
        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            let base = i * 21;
            let u = Matrix3::from_column_slice(&data[base..base + 9]);
            let s = Vector3::new(data[base + 9], data[base + 10], data[base + 11]);
            let vt = Matrix3::from_column_slice(&data[base + 12..base + 21]);
            results.push(GpuSvd3 { u, s, vt });
        }
        results
    }

    async fn run_gpu_svd3(backend: &impl Backend, matrices: &[Matrix3<f32>]) -> Vec<GpuSvd3> {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);
        let shader = super::Svd3GlamShader::from_backend(backend, &compiler).unwrap();

        let input_data = pack_matrices(matrices);
        let count = matrices.len();
        let gpu_inputs = GpuTensor::vector(backend, &input_data, BufferUsages::STORAGE).unwrap();
        let output_len = count * 21;
        let gpu_outputs = GpuTensor::<f32, _>::vector(
            backend,
            &vec![0.0f32; output_len],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass("test_svd3_glam", None);
        let args = Svd3Args { inputs: gpu_inputs.buffer(), outputs: gpu_outputs.buffer() };
        shader
            .test_svd3
            .launch(backend, &mut pass, &args, [count as u32, 1, 1])
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        let mut output_data = vec![0.0f32; output_len];
        backend.slow_read_buffer(gpu_outputs.buffer(), &mut output_data).await.unwrap();
        unpack_svd_results(&output_data, count)
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_svd3_glam_webgpu() {
        let backend = WebGpu::default().await.unwrap();

        // Deterministic cases.
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
        let mixed_sign = Matrix3::from_columns(&[
            Vector3::new(0.5, -1.2, 3.7),
            Vector3::new(2.1, 0.3, -0.8),
            Vector3::new(-1.0, 4.5, 2.2),
        ]);
        // Rank-deficient (third singular value ~ 0).
        let rank2 = Matrix3::from_columns(&[
            Vector3::new(1.0, 4.0, 7.0),
            Vector3::new(2.0, 5.0, 8.0),
            Vector3::new(3.0, 6.0, 9.0),
        ]);
        // Negative determinant.
        let mut neg_det = general;
        neg_det.set_column(0, &(-general.column(0)));

        // Full-rank cases: tight reconstruction + cross-check against nalgebra.
        let matrices = vec![identity, diagonal, symmetric, general, mixed_sign, neg_det];
        let labels = ["identity", "diagonal", "symmetric", "general", "mixed_sign", "neg_det"];
        let results = run_gpu_svd3(&backend, &matrices).await;

        assert!((results[0].s.x - 1.0).abs() < 1e-6);
        assert!((results[1].s.x - 3.0).abs() < 1e-6);
        assert!((results[1].s.z - 1.0).abs() < 1e-6);

        for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
            assert_valid_svd(m, svd, 1e-3, 1e-4, labels[i]);
            // glamx's A^T*A method loses ~half the f32 mantissa on the smallest singular value,
            // so the direct comparison is only good to a few e-3 (recompose above is far tighter).
            assert_matches_nalgebra(m, svd, 3e-3, labels[i]);
        }

        // Rank-deficient case: glamx's A^T*A method squares the condition number, so the
        // smallest singular value and reconstruction are only loosely accurate (this matches
        // glamx's own `svd_3x3_rank_deficient` test: s.z < 0.1, recompose eps 0.02).
        let rank2_res = run_gpu_svd3(&backend, &[rank2]).await;
        assert!(rank2_res[0].s.z < 0.1, "rank2 s2 should be small: {}", rank2_res[0].s.z);
        assert_valid_svd(&rank2, &rank2_res[0], 0.02, 1e-4, "rank2");

        // Random battery. Scales are kept at/above unit magnitude: glamx's eigensolver uses an
        // absolute `d_max < 1e-20` degeneracy threshold on A^T*A, so very small-magnitude inputs
        // (A^T*A near that threshold) fall outside the algorithm's valid domain. Deformation
        // gradients in the physics solver are ~O(1), so this is not a concern for the drop-in use.
        for (seed, lo, hi) in [
            (0xDEAD_BEEFu32, -1.0f32, 1.0f32),
            (0xCAFE_1234, -100.0, 100.0),
        ] {
            let mut rng = Rng::new(seed);
            let matrices: Vec<_> = (0..500).map(|_| rng.next_mat3(lo, hi)).collect();
            let results = run_gpu_svd3(&backend, &matrices).await;
            for (i, (m, svd)) in matrices.iter().zip(results.iter()).enumerate() {
                let label = format!("random_{seed:x}_{i}");
                // Orthogonality + descending singular values hold for every matrix.
                assert_svd_invariants(svd, 1e-3, &label);
                // Reconstruction is only checked for well-conditioned draws: glamx's A^T*A method
                // degrades on near-singular matrices (it squares the condition number), so an
                // occasional ill-conditioned random matrix legitimately reconstructs poorly.
                let na_sv = m.svd(false, false).singular_values;
                if na_sv[2] > 1e-2 * na_sv[0] {
                    let recon = svd.u * Matrix3::from_diagonal(&svd.s) * svd.vt;
                    assert!(
                        approx_eq_mat3_rel(m, &recon, 1e-2),
                        "{label}: reconstruction failed (cond {})\n  original: {m:?}\n  recon:    {recon:?}",
                        na_sv[0] / na_sv[2]
                    );
                }
            }
        }
    }
}
