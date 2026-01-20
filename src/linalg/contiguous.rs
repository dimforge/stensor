use crate::shapes::{MatrixOrdering, ViewShape, ViewShapeBuffers};
use crate::tensor::GpuTensorView;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};

#[derive(Shader)]
#[shader(module = "stensor::linalg::contiguous")]
/// Slang module for conversion from a non-contiguous tensor into a contiguous tensor.
pub struct Contiguous<B: Backend> {
    /// Shader for copying a non-contiguous tensor into a row-major contiguous tensor.
    pub contiguous_row_major: GpuFunction<B>,
    /// Shader for copying a non-contiguous tensor into a column-major contiguous tensor.
    pub contiguous_col_major: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct ContiguousArgs<'a, B: Backend> {
    tensor: B::BufferSlice<'a, f32>,
    out: B::BufferSlice<'a, f32>,
    shape: &'a B::Buffer<ViewShape>,
}

impl<B: Backend> Contiguous<B> {
    /// Launch the kernel that copies the content of a `tensor` with non-contiguous layout into
    /// the contiguous tensor `out`.
    pub fn launch<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        out: impl Into<GpuTensorView<'a, f32, B>>,
        tensor: impl Into<GpuTensorView<'a, f32, B>>,
    ) -> Result<(), B::Error> {
        let out = out.into();
        let tensor = tensor.into();
        let tensor_shape = tensor.shape();
        let out_shape = out.shape();
        assert_eq!(tensor_shape.size, out_shape.size);

        let function = match out.is_contiguous() {
            Some(MatrixOrdering::ColumnMajor) => &self.contiguous_col_major,
            Some(MatrixOrdering::RowMajor) => &self.contiguous_row_major,
            None => panic!("Output tensor must be contiguous."),
        };

        shapes.insert(backend, tensor_shape)?;
        let shape = shapes.get(tensor_shape).unwrap_or_else(|| unreachable!());
        let args = ContiguousArgs {
            tensor: tensor.buffer(),
            out: out.buffer(),
            shape,
        };

        function.launch_capped(backend, pass, &args, tensor_shape.len() as u32)
    }
}

#[cfg(test)]
mod test {
    use crate::shapes::ViewShapeBuffers;
    use crate::tensor::GpuTensor;
    use minislang::SlangCompiler;
    use nalgebra::DMatrix;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use slang_hal::{BufferUsages, Shader};
    use wgpu::{Features, Limits};

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    async fn gpu_contiguous_cuda() {
        let mut backend = slang_hal::cuda::Cuda::new().unwrap();
        gpu_contiguous_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_contiguous_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_contiguous_generic(backend).await;
    }

    async fn gpu_contiguous_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);
        let contiguous = super::Contiguous::from_backend(&backend, &compiler).unwrap();

        let mut shapes = ViewShapeBuffers::new(&backend);

        const NROWS: u32 = 256;
        const NCOLS: u32 = 128;

        let tensor = DMatrix::<f32>::new_random(NROWS as usize, NCOLS as usize);
        let mut output = DMatrix::<f32>::new_random(NCOLS as usize, NROWS as usize);

        let gpu_tensor = GpuTensor::matrix(&backend, &tensor, BufferUsages::STORAGE).unwrap();
        let gpu_output = GpuTensor::matrix(
            &backend,
            &output,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        contiguous
            .launch(
                &backend,
                &mut shapes,
                &mut pass,
                &gpu_output,
                gpu_tensor.as_view().transposed(),
            )
            .unwrap();
        drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();
        backend
            .slow_read_buffer(gpu_output.buffer(), output.as_mut_slice())
            .await
            .unwrap();

        // NOTE: don't use assert_relative_eq so it doesn't print out the whole matrices
        //       when it fails (it tends to break rustrover tests integration).
        assert!(output == tensor.transpose());
    }
}
