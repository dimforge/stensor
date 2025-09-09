use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use crate::shapes::{ViewShape, ViewShapeBuffers};
use crate::tensor::GpuTensorView;
use slang_hal::{Shader, ShaderArgs};

/// Slang module for replicating the content of a source tensor as many times as possible to fill
/// a destination tensor.
#[derive(Shader)]
#[shader(module = "stensor::linalg::repeat")]
pub struct Repeat<B: Backend> {
    /// Kernel for replicating the content of a source tensor as many times as possible to fill
    /// a destination tensor.
    ///
    /// The shape of the destination tensor needs to be an integer multiple of the source tensor’s
    /// shape.
    pub repeat: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct RepeatArgs<'a, B: Backend> {
    source: B::BufferSlice<'a, f32>,
    result: B::BufferSlice<'a, f32>,
    shape_source: &'a B::Buffer<ViewShape>,
    shape_result: &'a B::Buffer<ViewShape>,
}

impl<B: Backend> Repeat<B> {
    /// Launches the kernel that repeats the content of `source` into `destination` as many times
    /// as needed to fill `destination`.
    ///
    /// The shape of `destination` must be an integer multiple of the shape of `source`.
    pub fn launch<'a>(
        &self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        destination: impl Into<GpuTensorView<'a, f32, B>>,
        source: impl Into<GpuTensorView<'a, f32, B>>,
    ) -> Result<(), B::Error> {
        let result = destination.into();
        let result_shape = result.shape();

        let source = source.into();
        let source_shape = source.shape();

        assert!(result_shape.is_multiple_of(source_shape));

        shapes.insert(backend, source_shape)?;
        shapes.insert(backend, result_shape)?;
        let shape_source = shapes.get(source_shape).unwrap_or_else(|| unreachable!());
        let shape_result = shapes.get(result_shape).unwrap_or_else(|| unreachable!());

        let args = RepeatArgs {
            source: source.buffer(),
            result: result.buffer(),
            shape_source,
            shape_result,
        };
        self.repeat
            .launch(backend, pass, &args, [result.len() as u32, 1, 1])
    }
}
