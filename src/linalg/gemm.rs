use crate::shapes::{ViewShape, ViewShapeBuffers};
use crate::tensor::GpuTensorView;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};

#[derive(Shader)]
#[shader(module = "stensor::linalg::gemm")]
/// Shader for computing the product of two matrices.
pub struct Gemm<B: Backend> {
    /// The compute pipeline for `matrix1 * matrix2`.
    pub gemm: GpuFunction<B>,
    /// A compute pipeline for `matrix1 * matrix2` leveraging workgroup reduction.
    pub gemm_fast: GpuFunction<B>,
    /// The compute pipeline for `transpose(matrix1) * matrix2`.
    pub gemm_tr: GpuFunction<B>,
    /// A compute pipeline for `transpose(matrix1) * matrix2` leveraging workgroup reduction.
    pub gemm_tr_fast: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct GemmArgs<'a, B: Backend> {
    m1: B::BufferSlice<'a, f32>,
    m2: B::BufferSlice<'a, f32>,
    out: B::BufferSlice<'a, f32>,
    shape_m1: &'a B::Buffer<ViewShape>,
    shape_m2: &'a B::Buffer<ViewShape>,
    shape_out: &'a B::Buffer<ViewShape>,
}

/// Variants used to select the specific kernel to dispatch from the [`Gemm`] shader.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GemmVariant {
    /// The compute pipeline for `matrix1 * matrix2`.
    Gemm,
    /// A compute pipeline for `matrix1 * matrix2` leveraging workgroup reduction.
    GemmFast,
    /// The compute pipeline for `transpose(matrix1) * matrix2`.
    GemmTr,
    /// A compute pipeline for `transpose(matrix1) * matrix2` leveraging workgroup reduction.
    GemmTrFast,
}

impl<B: Backend> Gemm<B> {
    /// Dispatch this shader to compute `out = m1 * m2`.
    pub fn dispatch<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        m1: impl Into<GpuTensorView<'a, f32, B>>,
        m2: impl Into<GpuTensorView<'a, f32, B>>,
    ) -> Result<(), B::Error> {
        self.dispatch_generic(backend, shapes, pass, out, m1, m2, GemmVariant::Gemm)
    }

    /// Dispatch this shader to compute `out = tr(m1) * m2`.
    pub fn dispatch_tr<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        m1: impl Into<GpuTensorView<'a, f32, B>>,
        m2: impl Into<GpuTensorView<'a, f32, B>>,
    ) -> Result<(), B::Error> {
        self.dispatch_generic(backend, shapes, pass, out, m1, m2, GemmVariant::GemmTr)
    }

    /// Dispatches the matrix-vector multiplication variant indicated by the given [`GemmVariant`].
    pub fn dispatch_generic<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        m1: impl Into<GpuTensorView<'a, f32, B>>,
        m2: impl Into<GpuTensorView<'a, f32, B>>,
        variant: GemmVariant,
    ) -> Result<(), B::Error> {
        let out = out.into();
        let m1 = m1.into();
        let m2 = m2.into();
        let [out_rows, out_cols, out_mats, _out_cubes] = out.shape().size;

        // Check dimensions.
        let m_rows;
        let m_cols;
        {
            match variant {
                GemmVariant::Gemm | GemmVariant::GemmFast => {
                    m_rows = m1.shape().size[0];
                    m_cols = m1.shape().size[1];
                }
                GemmVariant::GemmTr | GemmVariant::GemmTrFast => {
                    m_rows = m1.shape().size[1];
                    m_cols = m1.shape().size[0];
                }
            };

            assert_eq!(m_cols, m2.shape().size[0], "Gemm: dimension mismatch.");
            assert_eq!(m_rows, out_rows, "Gemm: dimension mismatch.");
            assert_eq!(out_cols, m2.shape().size[1], "Gemm: dimension mismatch.");
            assert_eq!(out_mats, m1.shape().size[2], "Gemm: dimension mismatch.");
            assert_eq!(out_mats, m2.shape().size[2], "Gemm: dimension mismatch.");
        }

        let aligned_shape_out = out.shape().f32_to_vec4();
        let aligned_shape_m1 = m1.shape().f32_to_vec4();
        let aligned_shape_m2 = m2.shape().f32_to_vec4();

        shapes.insert(backend, aligned_shape_out)?;
        shapes.insert(backend, aligned_shape_m1)?;
        shapes.insert(backend, aligned_shape_m2)?;
        let shape_out = shapes
            .get(aligned_shape_out)
            .unwrap_or_else(|| unreachable!());
        let shape_m1 = shapes
            .get(aligned_shape_m1)
            .unwrap_or_else(|| unreachable!());
        let shape_m2 = shapes
            .get(aligned_shape_m2)
            .unwrap_or_else(|| unreachable!());

        let pipeline = match variant {
            GemmVariant::Gemm => &self.gemm,
            GemmVariant::GemmFast => &self.gemm_fast,
            GemmVariant::GemmTr => &self.gemm_tr,
            GemmVariant::GemmTrFast => &self.gemm_tr_fast,
        };

        let n = match variant {
            // Each thread handles 4 rows of the matrix, there is no special
            // consideration of workgroup threads.
            GemmVariant::Gemm | GemmVariant::GemmTr => out_rows.div_ceil(64),
            // Each workgroup handles 4 entire rows of the matrix.
            GemmVariant::GemmFast | GemmVariant::GemmTrFast => out_rows.div_ceil(4),
        };

        #[cfg(all(feature = "cuda", feature = "cublas"))]
        if out.is_entire_tensor().is_some()
            && m1.is_entire_tensor().is_some()
            && m2.is_entire_tensor().is_some()
            && let Some(cuda) = backend.as_cuda()
        {
            if cuda.cublas_enabled {
                use cudarc::cublas::safe::{Gemm, GemmConfig};
                use cudarc::cublas::sys::cublasOperation_t;
                use cudarc::driver::CudaSlice;
                use cudarc::driver::CudaView;

                // Call cublas
                let transa = match variant {
                    GemmVariant::Gemm | GemmVariant::GemmFast => cublasOperation_t::CUBLAS_OP_N,
                    GemmVariant::GemmTr | GemmVariant::GemmTrFast => cublasOperation_t::CUBLAS_OP_T,
                };

                let gemm_config = GemmConfig {
                    transa,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: m_rows as i32,
                    n: out_cols as i32,
                    k: m_cols as i32,
                    alpha: 1.0,
                    lda: m1.shape().size[0] as i32,
                    ldb: m2.shape().size[0] as i32,
                    beta: 0.0,
                    ldc: out.shape().size[0] as i32,
                };

                unsafe {
                    let m1: &CudaSlice<f32> = std::mem::transmute(m1.raw_buffer());
                    let m2: &CudaSlice<f32> = std::mem::transmute(m2.raw_buffer());
                    let out: &CudaSlice<f32> = std::mem::transmute(out.raw_buffer());
                    // FIXME SAFETY the out buffer should be a mutable ref
                    #[allow(mutable_transmutes)]
                    let out: &mut cudarc::driver::CudaSlice<f32> = std::mem::transmute(out);
                    cuda.cublas.gemm(gemm_config, m1, m2, out).unwrap();
                }
                return Ok(());
            }
        }

        let args = GemmArgs {
            m1: m1.buffer(),
            m2: m2.buffer(),
            out: out.buffer(),
            shape_m1,
            shape_m2,
            shape_out,
        };
        pipeline.launch(backend, pass, &args, [n, 1, 1])
    }
}

#[cfg(test)]
mod test {
    use crate::GemmVariant;
    use crate::shapes::ViewShapeBuffers;
    use crate::tensor::GpuTensor;
    use approx::relative_eq;
    use minislang::SlangCompiler;
    use nalgebra::DMatrix;
    use slang_hal::Shader;
    use slang_hal::backend::{Backend, Encoder, WebGpu};
    use wgpu::{BufferUsages, Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    async fn gpu_gemm_cuda() {
        let mut backend = slang_hal::cuda::Cuda::new().unwrap();
        #[cfg(feature = "cublas")]
        {
            backend.cublas_enabled = false;
        }
        gpu_gemm_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cublas")]
    async fn gpu_gemm_cublas() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = true;
        gpu_gemm_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_gemm_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_gemm_generic(backend).await;
    }

    async fn gpu_gemm_generic(backend: impl Backend) {
        let compiler = SlangCompiler::new(vec!["../../crates/stensor/shaders".into()]);
        let gemm = super::Gemm::from_backend(&backend, &compiler).unwrap();

        let mut shapes = ViewShapeBuffers::new(&backend);

        const NROWS: u32 = 256;
        const NCOLS: u32 = 256;

        let m1_cpu = DMatrix::<f32>::new_random(NROWS as usize, NCOLS as usize);
        let m2_cpu = DMatrix::<f32>::new_random(NCOLS as usize, NROWS as usize);
        let lhs_cpu = DMatrix::<f32>::zeros(NROWS as usize, NROWS as usize);
        let mut gpu_result = DMatrix::<f32>::zeros(NROWS as usize, NROWS as usize);

        let m1 = GpuTensor::matrix(&backend, &m1_cpu, BufferUsages::STORAGE).unwrap();
        let m2 = GpuTensor::matrix(&backend, &m2_cpu, BufferUsages::STORAGE).unwrap();
        let result = GpuTensor::matrix(
            &backend,
            &lhs_cpu,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        for variant in [
            GemmVariant::Gemm,
            GemmVariant::GemmTr,
            GemmVariant::GemmFast,
            GemmVariant::GemmTrFast,
        ] {
            println!("Checking variant: {:?}", variant);
            let t0 = std::time::Instant::now();
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass();
            gemm.dispatch_generic(&backend, &mut shapes, &mut pass, &result, &m1, &m2, variant)
                .unwrap();
            drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

            backend.submit(encoder).unwrap();
            backend.synchronize().unwrap();
            println!("GEMM before read: {}", t0.elapsed().as_secs_f32());
            backend
                .slow_read_buffer(result.buffer(), gpu_result.as_mut_slice())
                .await
                .unwrap();
            println!("GEMM time: {}", t0.elapsed().as_secs_f32());

            let cpu_result = match variant {
                GemmVariant::Gemm | GemmVariant::GemmFast => &m1_cpu * &m2_cpu,
                GemmVariant::GemmTr | GemmVariant::GemmTrFast => m1_cpu.tr_mul(&m2_cpu),
            };

            // NOTE: don't use assert_relative_eq so it doesn't print out the whole matrices
            //       when it fails (it tends to break rustrover tests integration).
            if !relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-3) {
                println!("{:?}", &gpu_result.as_slice()[..10]);
                println!("{:?}", &cpu_result.as_slice()[..10]);
            }
            assert!(relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-3));
        }
    }
}
