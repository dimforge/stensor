use nalgebra::Similarity3;

/// A GPU-compatible 3d similarity (uniform scale + rotation + translation).
pub type GpuSim3 = Similarity3<f32>;
