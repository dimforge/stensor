use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use crate::shapes::{MatrixOrdering, ViewShape, ViewShapeBuffers};
use crate::tensor::GpuTensorView;
use slang_hal::{Shader, ShaderArgs};

/// Indicates if a matrix needs to be considered as-is or as its transpose when running a matrix
/// multiplication operation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MatrixMode {
    /// The matrix multiplication operation will operate with the normal matrix value (not transposed).
    Normal,
    /// The matrix multiplication operation will operate with the matrix’s transpose.
    Transposed,
}

impl MatrixMode {
    /// Flips between transposed and non-transposed mode.
    pub fn transpose(&mut self) {
        match self {
            Self::Normal => *self = Self::Transposed,
            Self::Transposed => *self = Self::Normal,
        }
    }
}

/// Alternate name for `MatrixMode::Transposed` for conciseness when calling matrix multiplication.
pub const N: MatrixMode = MatrixMode::Normal;
/// Alternate name for `MatrixMode::Transposed` for conciseness when calling matrix multiplication.
pub const T: MatrixMode = MatrixMode::Transposed;

#[derive(Shader)]
#[shader(module = "gla::linalg::gemv")]
/// Shader for computing the product of a matrix and a vector.
pub struct Gemv<B: Backend> {
    /// The compute pipeline for `matrix * vector`.
    pub gemv: GpuFunction<B>,
    /// The compute pipeline for `matrix * vector` (naive implementation).
    pub gemv_naive: GpuFunction<B>,
    /// A compute pipeline for `matrix * vector` leveraging workgroup reduction.
    pub gemv_fast: GpuFunction<B>,
    /// The compute pipeline for `transpose(matrix) * vector`.
    pub gemv_tr: GpuFunction<B>,
    /// A compute pipeline for `transpose(matrix) * vector` leveraging workgroup reduction.
    pub gemv_tr_fast: GpuFunction<B>,
    /// The compute pipeline for `transpose(matrix) * vector` (naive implementation).
    pub gemv_tr_naive: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct GemvArgs<'a, B: Backend> {
    m: B::BufferSlice<'a, f32>,
    v: B::BufferSlice<'a, f32>,
    out: B::BufferSlice<'a, f32>,
    shape_m: &'a B::Buffer<ViewShape>,
    shape_v: &'a B::Buffer<ViewShape>,
    shape_out: &'a B::Buffer<ViewShape>,
}

/// Variants used to select the specific kernel to dispatch from the [`Gemv`] shader.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GemvVariant {
    /// Multiplication of a vector by a matrix.
    Gemv,
    /// An optimized version for multiplication of a vector by a matrix.
    GemvFast, // This is actually currently much **slower** than gemv.
    /// Multiplication of a vector by a matrix (most naive implementation).
    GemvNaive,
    /// Multiplication of a vector by a transposed matrix.
    GemvTr,
    /// An optimized version for multiplication of a vector by a transposed matrix.
    GemvTrFast, // This is actually much faster than GemvTr
    /// Multiplication of a vector by a transposed matrix (most native implementation).
    GemvTrNaive,
}

impl<B: Backend> Gemv<B> {
    /// Dispatches this shader to compute `out = m * v`.
    pub fn dispatch<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        m: impl Into<GpuTensorView<'a, f32, B>>,
        v: impl Into<GpuTensorView<'a, f32, B>>,
    ) -> Result<(), B::Error> {
        self.dispatch_generic(backend, shapes, pass, out, m, v, N, N)
    }

    /// Dispatches this shader to compute `out = tr(m) * v`.
    pub fn dispatch_tr<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        m: impl Into<GpuTensorView<'a, f32, B>>,
        v: impl Into<GpuTensorView<'a, f32, B>>,
    ) -> Result<(), B::Error> {
        self.dispatch_generic(backend, shapes, pass, out, m, v, T, N)
    }

    /// Dispatches the matrix-vector multiplication variant indicated by the given [`GemvVariant`].
    pub fn dispatch_generic<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        m: impl Into<GpuTensorView<'a, f32, B>>,
        v: impl Into<GpuTensorView<'a, f32, B>>,
        // Indicates arguments that need to be interpreted as transposed.
        mut m_mode: MatrixMode,
        mut v_mode: MatrixMode,
    ) -> Result<(), B::Error> {
        let out = out.into();
        let mut m = m.into();
        let mut v = v.into();

        // Shapes of the mathematical operatioon being executed, independently from the potential artificial transpose
        // we’d apply for switching to a column-major equivalent of `m` and `v`.
        let math_shape_out = out.shape();
        let math_shape_m = m.shape().maybe_transpose(m_mode == T);
        let math_shape_v = v.shape().maybe_transpose(v_mode == T);

        let ordering_out = out
            .shape()
            .ordering()
            .expect("matmul: input doesn’t have a contiguous dimension < 2");
        let mut ordering_m = m
            .shape()
            .ordering()
            .expect("matmul: input doesn’t have a contiguous dimension < 2");
        let mut ordering_v = v
            .shape()
            .ordering()
            .expect("matmul: input doesn’t have a contiguous dimension < 2");

        // Since our kernels assume a column-major output, swap the arguments and transpose everything
        // if the provided output is row-major.
        if ordering_out == MatrixOrdering::RowMajor {
            std::mem::swap(&mut ordering_m, &mut ordering_v);
            std::mem::swap(&mut m, &mut v);
            std::mem::swap(&mut m_mode, &mut v_mode);
            m_mode.transpose();
            v_mode.transpose();
        }

        if ordering_m == MatrixOrdering::RowMajor {
            m_mode.transpose();
        }
        if ordering_v == MatrixOrdering::RowMajor {
            v_mode.transpose();
        }

        if v_mode == MatrixMode::Transposed {
            panic!("matmul: not implemented yet")
        }

        // Shape of the matrices in column-major order.
        let col_maj_shape_out = out
            .shape()
            .maybe_transpose(ordering_out == MatrixOrdering::RowMajor);
        let col_maj_shape_m = m
            .shape()
            .maybe_transpose(ordering_m == MatrixOrdering::RowMajor);
        let col_maj_shape_v = v
            .shape()
            .maybe_transpose(ordering_v == MatrixOrdering::RowMajor);

        // Check dimensions.
        {
            assert_eq!(
                math_shape_m.size[1], math_shape_v.size[0],
                "matmul: dimensions mismatch"
            );
            assert_eq!(
                math_shape_out.size[0], math_shape_m.size[0],
                "matmul: dimensions mismatch"
            );
            assert_eq!(
                math_shape_out.size[1], math_shape_v.size[1],
                "matmul: dimensions mismatch"
            );
            // assert_eq!(
            //     math_shape_out.size[2], math_shape_v.size[2],
            //     "matmul: dimensions mismatch"
            // );
            // assert_eq!(
            //     math_shape_out.size[3], math_shape_v.size[3],
            //     "matmul: dimensions mismatch"
            // );
            // assert_eq!(
            //    math_shape_m.size[2], math_shape_v.size[2],
            //     "matmul: dimensions currently unsupported"
            // );
            // assert_eq!(
            //    math_shape_m.size[3], math_shape_v.size[3],
            //     "matmul: dimensions currently unsupported"
            // );
        }

        // Check contiguity and alignment.
        {
            // TODO: implement kernels that support non-contiguous inputs.
            assert_eq!(col_maj_shape_v.stride[0], 1, "matmul: non-contiguous input");
            assert_eq!(col_maj_shape_m.stride[0], 1, "matmul: non-contiguous input");
            assert_eq!(
                col_maj_shape_out.stride[0], 1,
                "matmul: non-contiguous input"
            );
        }

        let use_float4 = !(col_maj_shape_out.size[0] == 256 && col_maj_shape_out.size[1] == 256)
            && col_maj_shape_v.size[0] % 4 == 0
            && col_maj_shape_m.size[0] % 4 == 0
            && col_maj_shape_out.size[0] % 4 == 0;

        // Cache shape buffers.
        let aligned_shape_out = if use_float4 {
            col_maj_shape_out.f32_to_vec4()
        } else {
            col_maj_shape_out
        };
        let aligned_shape_m = if use_float4 {
            col_maj_shape_m.f32_to_vec4()
        } else {
            col_maj_shape_m
        };
        let aligned_shape_v = if use_float4 {
            col_maj_shape_v.f32_to_vec4()
        } else {
            col_maj_shape_v
        };

        shapes.insert(backend, aligned_shape_out)?;
        shapes.insert(backend, aligned_shape_m)?;
        shapes.insert(backend, aligned_shape_v)?;
        let shape_out = shapes
            .get(aligned_shape_out)
            .unwrap_or_else(|| unreachable!());
        let shape_m = shapes
            .get(aligned_shape_m)
            .unwrap_or_else(|| unreachable!());
        let shape_v = shapes
            .get(aligned_shape_v)
            .unwrap_or_else(|| unreachable!());

        // Select kernel.
        const WORKGROUP_SIZE: u32 = 32;

        let variant = match m_mode {
            MatrixMode::Transposed => {
                if use_float4 && col_maj_shape_m.size[0] % (WORKGROUP_SIZE * 4) == 0 {
                    GemvVariant::GemvTrFast
                } else if use_float4 {
                    GemvVariant::GemvTr
                } else {
                    GemvVariant::GemvTrNaive
                }
            }
            MatrixMode::Normal => {
                if use_float4 {
                    GemvVariant::Gemv
                } else {
                    GemvVariant::GemvNaive
                }
            }
        };

        let pipeline = match variant {
            GemvVariant::Gemv => &self.gemv,
            GemvVariant::GemvFast => &self.gemv_fast,
            GemvVariant::GemvNaive => &self.gemv_naive,
            GemvVariant::GemvTr => &self.gemv_tr,
            GemvVariant::GemvTrFast => &self.gemv_tr_fast,
            GemvVariant::GemvTrNaive => &self.gemv_tr_naive,
        };

        let n = match variant {
            // Each thread handles a row of the matrix.
            GemvVariant::Gemv
            | GemvVariant::GemvTr
            | GemvVariant::GemvNaive
            | GemvVariant::GemvTrNaive => aligned_shape_out.size[0].div_ceil(WORKGROUP_SIZE),
            // Each workgroup handles a row of the matrix.
            GemvVariant::GemvFast | GemvVariant::GemvTrFast => aligned_shape_out.size[0],
        };

        let args = GemvArgs {
            m: m.buffer(),
            v: v.buffer(),
            out: out.buffer(),
            shape_m,
            shape_v,
            shape_out,
        };
        pipeline.launch_grid(
            backend,
            pass,
            &args,
            [n, col_maj_shape_out.size[1], col_maj_shape_out.size[2]],
        )
    }
}

#[cfg(test)]
mod test {
    use crate::GemvVariant;
    use approx::assert_relative_eq;
    use minislang::SlangCompiler;
    use nalgebra::{DMatrix, DVector};
    use slang_hal::Shader;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use crate::shapes::ViewShapeBuffers;
    use crate::tensor::GpuTensor;
    use wgpu::{BufferUsages, Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    async fn gpu_gemv_cuda() {
        let mut backend = slang_hal::cuda::Cuda::new().unwrap();
        #[cfg(feature = "cublas")]
        {
            backend.cublas_enabled = false;
        }
        gpu_gemv_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cublas")]
    async fn gpu_gemv_cublas() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = true;
        gpu_gemv_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_gemv_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_gemv_generic(backend).await;
    }

    async fn gpu_gemv_generic(backend: impl Backend) {
        let compiler = SlangCompiler::new(vec!["../../crates/gla/shaders".into()]);
        let gemv = super::Gemv::from_backend(&backend, &compiler).unwrap();

        let mut shapes = ViewShapeBuffers::new(&backend);

        const NROWS: u32 = 256;
        const NCOLS: u32 = 256;

        let m_cpu = DMatrix::<f32>::new_random(NROWS as usize, NCOLS as usize);
        let v_cpu = DVector::<f32>::new_random(NCOLS as usize);
        let lhs_cpu = DVector::<f32>::zeros(NROWS as usize);
        let mut gpu_result = DVector::<f32>::zeros(NROWS as usize);

        let m = GpuTensor::matrix(&backend, &m_cpu, BufferUsages::STORAGE).unwrap();
        let v = GpuTensor::matrix(&backend, &v_cpu, BufferUsages::STORAGE).unwrap();
        let result = GpuTensor::matrix(
            &backend,
            &lhs_cpu,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        for variant in [
            GemvVariant::Gemv,
            GemvVariant::GemvTr,
            GemvVariant::GemvFast,
            GemvVariant::GemvTrFast,
        ] {
            println!("Checking variant: {:?}", variant);
            let t0 = std::time::Instant::now();
            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass();
            let modes = match variant {
                GemvVariant::GemvFast | GemvVariant::Gemv | GemvVariant::GemvNaive => {
                    (super::N, super::N)
                }
                GemvVariant::GemvTrFast | GemvVariant::GemvTr | GemvVariant::GemvTrNaive => {
                    (super::T, super::N)
                }
            };
            gemv.dispatch_generic(
                &backend,
                &mut shapes,
                &mut pass,
                &result,
                &m,
                &v,
                modes.0,
                modes.1,
            )
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
                GemvVariant::Gemv | GemvVariant::GemvFast | GemvVariant::GemvNaive => {
                    &m_cpu * &v_cpu
                }
                GemvVariant::GemvTr | GemvVariant::GemvTrFast | GemvVariant::GemvTrNaive => {
                    m_cpu.tr_mul(&v_cpu)
                }
            };

            // NOTE: don't use assert_relative_eq so it doesn't print out the whole matrices
            //       when it fails (it tends to break rustrover tests integration).
            // assert!(relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-3));
            assert_relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-3);
        }
    }
}
