use crate::shapes::{ViewShape, ViewShapeBuffers};
use crate::tensor::GpuTensorView;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// The desired operation for the [`OpAssign`] kernel.
pub enum OpAssignVariant {
    /// Sum: `a += b`
    Add,
    /// Subtraction: `a -= b`
    Sub,
    /// Product: `a *= b`
    Mul,
    /// Division: `a /= b`
    Div,
    /// Copy: `a = b`
    Copy,
}

/// Slang modules for various in-place binary operations.
#[derive(Shader)]
#[shader(module = "stensor::linalg::op_assign")]
pub struct OpAssign<B: Backend> {
    /// Kernel for computing in-place the sum of two tensors.
    pub add: GpuFunction<B>,
    /// Kernel for computing in-place the subtraction of two tensors.
    pub sub: GpuFunction<B>,
    /// Kernel for computing in-place the product of two tensors.
    pub mul: GpuFunction<B>,
    /// Kernel for computing in-place the division of two tensors.
    pub div: GpuFunction<B>,
    /// Kernel for copying a tensor into another.
    pub copy: GpuFunction<B>,
    /// Kernel for copying a tensor into another, using a custom offset where to start reading
    /// the source tensor.
    pub copy_with_offsets: GpuFunction<B>,
}

/// Offsets given to the offseted tensor copy operation.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BinOpOffsets {
    /// Index of the first element of the destination tensor.
    pub a: u32,
    /// Index of the first element of the source tensor.
    pub b: u32,
    /// Padding used to match the GPU layout. Its value is irrelevant.
    pub padding: [u32; 2],
}

#[derive(ShaderArgs)]
pub struct BinOpArgs<'a, B: Backend> {
    pub shape_a: &'a B::Buffer<ViewShape>,
    pub shape_b: &'a B::Buffer<ViewShape>,
    pub a: B::BufferSlice<'a, f32>,
    pub b: B::BufferSlice<'a, f32>,
    pub offsets: Option<B::BufferSlice<'a, BinOpOffsets>>,
}

impl<B: Backend> OpAssign<B> {
    /// Launches the kernel for a binary operation `variant` where the first operand
    /// `a` being read & written to, and `b` is only being read from (e.g. `a += b`).
    pub fn launch<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        variant: OpAssignVariant,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let a = a.into();
        let b = b.into();
        let pipeline = match variant {
            OpAssignVariant::Add => &self.add,
            OpAssignVariant::Copy => &self.copy,
            OpAssignVariant::Div => &self.div,
            OpAssignVariant::Mul => &self.mul,
            OpAssignVariant::Sub => &self.sub,
        };

        let shape_a = a.shape();
        let shape_b = b.shape();
        assert!(
            shape_a.is_multiple_of(shape_b),
            "shape_A: {:?} must be a multiple of shape_b: {:?}",
            shape_a.size,
            shape_b.size
        );

        shapes.insert(backend, shape_a)?;
        shapes.insert(backend, shape_b)?;
        let shape_a = shapes.get(shape_a).unwrap();
        let shape_b = shapes.get(shape_b).unwrap();

        let binop_args = BinOpArgs {
            shape_a,
            shape_b,
            a: a.buffer(),
            b: b.buffer(),
            offsets: None,
        };

        pipeline.launch_capped(backend, pass, &binop_args, a.len() as u32)
    }

    // FIXME: this only exists because we needed a quick fix to work arround the limitation on
    //        buffer offset alignment when targetting WebGpu (in our case the buffer offset was 32
    //        but the hardware needed an alignment of 256).
    //        We should figure out a more general way of handling this.
    /// Launches the GPU kernel for copying the content of `b[offsets.b..]` into
    /// `a[offsets.a..]`.
    ///
    /// While this is similar to calling `launch` with an already offset tensor view,
    /// this is useful for cases where the desired offset is smaller than what’s supported
    /// by the backend (for example WebGpu).
    pub fn launch_copy_with_offsets<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        offsets: impl Into<GpuTensorView<'b, BinOpOffsets, B>>,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let offsets = offsets.into();
        let a = a.into();
        let b = b.into();
        let pipeline = &self.copy_with_offsets;

        let shape_a = a.shape();
        let shape_b = b.shape();
        assert!(shape_a.is_multiple_of(shape_b));

        shapes.insert(backend, shape_a)?;
        shapes.insert(backend, shape_b)?;
        let shape_a = shapes.get(shape_a).unwrap();
        let shape_b = shapes.get(shape_b).unwrap();

        let binop_args = BinOpArgs {
            shape_a,
            shape_b,
            a: a.buffer(),
            b: b.buffer(),
            offsets: Some(offsets.buffer()),
        };

        pipeline.launch_capped(backend, pass, &binop_args, a.len() as u32)
    }
}

#[cfg(test)]
mod test {
    use super::{BinOpArgs, OpAssignVariant};
    use crate::shapes::ViewShapeBuffers;
    use crate::tensor::GpuTensor;
    use minislang::SlangCompiler;
    use nalgebra::DVector;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Buffer, Encoder};
    use slang_hal::BufferUsages;
    use slang_hal::shader::Shader;

    #[futures_test::test]
    #[serial_test::serial]
    #[cfg(feature = "cuda")]
    async fn gpu_op_assign_cuda() {
        let backend = slang_hal::cuda::Cuda::new().unwrap();
        gpu_op_assign_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_op_assign_webgpu() {
        let backend = WebGpu::default().await.unwrap();
        gpu_op_assign_generic(backend).await;
    }

    async fn gpu_op_assign_generic(backend: impl Backend) {
        let ops = [
            OpAssignVariant::Add,
            OpAssignVariant::Sub,
            OpAssignVariant::Mul,
            OpAssignVariant::Div,
            OpAssignVariant::Copy,
        ];
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        let op_assign = super::OpAssign::from_backend(&backend, &compiler).unwrap();

        for op in ops {
            println!("Testing: {:?}", op);

            let function = match op {
                OpAssignVariant::Add => &op_assign.add,
                OpAssignVariant::Sub => &op_assign.sub,
                OpAssignVariant::Mul => &op_assign.mul,
                OpAssignVariant::Div => &op_assign.div,
                OpAssignVariant::Copy => &op_assign.copy,
            };
            let mut shapes = ViewShapeBuffers::new(&backend);
            let mut encoder = backend.begin_encoding();

            const LEN: u32 = 1757;

            let v0 = DVector::from_fn(LEN as usize, |i, _| i as f32 + 0.1);
            let v1 = DVector::from_fn(LEN as usize, |i, _| i as f32 * 10.0 + 0.1);
            let mut gpu_result = DVector::zeros(LEN as usize);
            let gpu_v0 = GpuTensor::vector(
                &backend,
                &v0,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap();
            let gpu_v1 = GpuTensor::vector(&backend, &v1, BufferUsages::STORAGE).unwrap();

            shapes.insert(&backend, gpu_v0.as_view().shape()).unwrap();
            shapes.insert(&backend, gpu_v1.as_view().shape()).unwrap();

            let shape_a = shapes.get(gpu_v0.as_view().shape()).unwrap();
            let shape_b = shapes.get(gpu_v1.as_view().shape()).unwrap();

            let mut pass = encoder.begin_pass();
            let binop_args = BinOpArgs {
                shape_a,
                shape_b,
                a: gpu_v0.buffer().as_slice(),
                b: gpu_v1.buffer().as_slice(),
                offsets: None,
            };
            function
                .launch(&backend, &mut pass, &binop_args, [LEN, 1, 1])
                .unwrap();
            drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

            backend.submit(encoder).unwrap();
            backend
                .slow_read_buffer(gpu_v0.buffer(), gpu_result.as_mut_slice())
                .await
                .unwrap();

            let cpu_result = match op {
                OpAssignVariant::Add => v0 + v1,
                OpAssignVariant::Sub => v0 - v1,
                OpAssignVariant::Mul => v0.component_mul(&v1),
                OpAssignVariant::Div => v0.component_div(&v1),
                OpAssignVariant::Copy => v1.clone(),
            };

            println!("Testing: {:?}", gpu_result);
            approx::assert_relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-7);
        }
    }
}
