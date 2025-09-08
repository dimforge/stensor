//! Fundamental linear-algebra matrix/vector operations.

mod contiguous;
mod gemm;
mod gemv;
mod op_assign;
mod reduce;
mod repeat;

pub use contiguous::Contiguous;
pub use gemm::{Gemm, GemmVariant};
pub use gemv::{Gemv, GemvVariant, MatrixMode, N, T};
pub use op_assign::{BinOpOffsets, OpAssign, OpAssignVariant};
pub use reduce::{Reduce, ReduceVariant};
pub use repeat::Repeat;
