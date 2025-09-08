use nalgebra::{Matrix2, Vector2};

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// GPU representation of a 2x2 matrix SVD.
pub struct GpuSvd2 {
    /// First orthogonal matrix of the SVD.
    pub u: Matrix2<f32>,
    /// Singular values.
    pub s: Vector2<f32>,
    /// Second orthogonal matrix of the SVD.
    pub vt: Matrix2<f32>,
}
