use slang_hal::Shader;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;

#[cfg(test)]
use nalgebra::DVector;

/// The desired operation for the [`Reduce`] kernel.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum ReduceVariant {
    /// Minimum: `result = min(input[0], min(input[1], ...))`
    Min,
    /// Maximum: `result = max(input[0], max(input[1], ...))`
    Max,
    /// Sum: `result = input[0] + input[1] ...`
    Sum,
    /// Product: `result = input[0] * input[1] ...`
    Prod,
    /// Squared norm: `result = input[0] * input[0] + input[1] * input[1] ...`
    SqNorm,
}

impl ReduceVariant {
    #[cfg(test)]
    fn eval(self, val: &DVector<f32>) -> f32 {
        match self {
            ReduceVariant::Min => val.min(),
            ReduceVariant::Max => val.max(),
            ReduceVariant::Prod => val.product(),
            ReduceVariant::Sum => val.sum(),
            ReduceVariant::SqNorm => val.norm_squared(),
        }
    }
}

/// A GPU kernel for performing the operation described by [`ReduceVariant`].
#[derive(Shader)]
#[shader(module = "stensor::linalg::reduce")]
pub struct Reduce<B: Backend> {
    /// Kernel for computing the sum of every element of a tensor.
    pub reduce_sum: GpuFunction<B>,
    /// Kernel for computing the product of every element of a tensor.
    pub reduce_product: GpuFunction<B>,
    /// Kernel for computing the minimum element of a tensor.
    pub reduce_min: GpuFunction<B>,
    /// Kernel for computing the maximum element of a tensor.
    pub reduce_max: GpuFunction<B>,
    /// Kernel for computing the squared norm of a tensor.
    pub reduce_sqnorm: GpuFunction<B>,
}

#[cfg(test)]
mod test {
    use super::ReduceVariant;
    use minislang::SlangCompiler;
    use nalgebra::DVector;
    use slang_hal::ShaderArgs;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use slang_hal::shader::Shader;
    use crate::shapes::{ViewShape, ViewShapeBuffers};
    use crate::tensor::GpuTensor;
    use wgpu::BufferUsages;

    #[derive(ShaderArgs)]
    pub struct ReduceArgs<'a, B: Backend> {
        pub shape: &'a B::Buffer<ViewShape>,
        pub input: &'a B::Buffer<f32>,
        pub output: &'a B::Buffer<f32>,
    }

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    async fn gpu_reduce_cuda() {
        let backend = slang_hal::cuda::Cuda::new().unwrap();
        gpu_reduce_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_reduce_webgpu() {
        let backend = WebGpu::default().await.unwrap();
        gpu_reduce_generic(backend).await;
    }

    async fn gpu_reduce_generic(backend: impl Backend) {
        let ops = [
            ReduceVariant::Min,
            ReduceVariant::Max,
            ReduceVariant::Sum,
            ReduceVariant::Prod,
            ReduceVariant::SqNorm,
        ];
        let compiler = SlangCompiler::new(vec!["../../crates/stensor/shaders".into()]);

        let reduce = super::Reduce::from_backend(&backend, &compiler).unwrap();

        for op in ops {
            println!("Testing: {:?}", op);

            let function = match op {
                ReduceVariant::Min => &reduce.reduce_min,
                ReduceVariant::Max => &reduce.reduce_max,
                ReduceVariant::Sum => &reduce.reduce_sum,
                ReduceVariant::Prod => &reduce.reduce_product,
                ReduceVariant::SqNorm => &reduce.reduce_sqnorm,
            };
            let mut shapes = ViewShapeBuffers::new(&backend);
            let mut encoder = backend.begin_encoding();

            const LEN: u32 = 345;

            let v = DVector::new_random(LEN as usize);
            let mut gpu_result = [1.0];
            let gpu_v = GpuTensor::vector(&backend, &v, BufferUsages::STORAGE).unwrap();
            let gpu_out = GpuTensor::scalar(
                &backend,
                0.0,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap();

            shapes.insert(&backend, gpu_v.as_view().shape()).unwrap();

            let shape = shapes.get(gpu_v.as_view().shape()).unwrap();

            let mut pass = encoder.begin_pass();
            let binop_args = ReduceArgs {
                shape,
                input: gpu_v.buffer(),
                output: gpu_out.buffer(),
            };
            function
                .launch(&backend, &mut pass, &binop_args, [1, 1, 1])
                .unwrap();
            drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

            backend.submit(encoder).unwrap();
            backend
                .slow_read_buffer(gpu_out.buffer(), &mut gpu_result)
                .await
                .unwrap();

            let cpu_result = op.eval(&v);

            approx::assert_relative_eq!(gpu_result[0], cpu_result, epsilon = 1.0e-3);
        }
    }
}
