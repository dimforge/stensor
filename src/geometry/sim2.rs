use nalgebra::{Isometry2, Similarity2};

#[derive(Copy, Clone, PartialEq, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// A GPU-compatible 2D similarity (uniform scale + rotation + translation).
pub struct GpuSim2 {
    /// The similarity value.
    pub similarity: Similarity2<f32>,
    /// Padding to match the layout on gpu. Its value is irrelevant.
    pub padding: f32,
}

impl From<Similarity2<f32>> for GpuSim2 {
    fn from(value: Similarity2<f32>) -> Self {
        Self {
            similarity: value,
            padding: 0.0,
        }
    }
}

impl From<Isometry2<f32>> for GpuSim2 {
    fn from(value: Isometry2<f32>) -> Self {
        Self {
            similarity: Similarity2::from_isometry(value, 1.0),
            padding: 0.0,
        }
    }
}

impl GpuSim2 {
    /// The identity similarity (scale = 1, rotation = identity, translation = 0).
    pub fn identity() -> Self {
        Similarity2::identity().into()
    }
}
