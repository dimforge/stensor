use nalgebra::{Matrix4x3, Vector4};

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// A 3D SVD as represented on the gpu, with padding (every fourth rows
/// can be ignored).
// TODO: switch to encase?
pub struct GpuSvd3 {
    /// First orthogonal matrix of the SVD.
    u: Matrix4x3<f32>,
    /// Singular values.
    s: Vector4<f32>,
    /// Second orthogonal matrix of the SVD.
    vt: Matrix4x3<f32>,
}
