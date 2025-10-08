use minislang::{SlangCompiler, shader_slang::CompileTarget};
use std::path::PathBuf;
use std::str::FromStr;

pub fn main() {
    let slang = SlangCompiler::new(vec![PathBuf::from_str("./shaders").unwrap()]);

    let targets = [
        CompileTarget::Wgsl,
        #[cfg(feature = "cuda")]
        CompileTarget::CudaSource,
    ];

    for target in targets {
        slang.compile_all(target, "../shaders", "./src/autogen", &[]);
    }
}
