use slang_hal::Shader;

#[derive(Shader)]
#[shader(src = "inv.wgsl")]
/// Shader exposing small matrix inverses.
pub struct WgInv;

slang_hal::test_shader_compilation!(WgInv);
