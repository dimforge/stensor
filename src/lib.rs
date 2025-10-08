#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_large_err)]
#![warn(missing_docs)]

pub use geometry::*;
pub use linalg::*;

use minislang::SlangCompiler;

pub mod geometry;
pub mod linalg;
pub mod shapes;
pub mod tensor;

// pub mod utils;

const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/shaders");

/// Register all the shaders from this crate (and its dependencies) as modules accessible to the
/// `compiler`.
///
/// This function must be called before compiling any shader. This is the main way of handling
/// cross-crates shader dependencies without tricky include path handling.
pub fn register_shaders(compiler: &mut SlangCompiler) {
    compiler.add_dir(SLANG_SRC_DIR.clone());
}
