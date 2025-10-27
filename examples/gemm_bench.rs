use indexmap::IndexMap;
use minislang::SlangCompiler;
use nalgebra::DMatrix;
use slang_hal::Shader;
use slang_hal::backend::WebGpu;
use slang_hal::backend::{Backend, Encoder};
use stensor::linalg::{Gemm, GemmVariant};
use stensor::shapes::ViewShapeBuffers;
use stensor::tensor::GpuTensor;
use wgpu::{BufferUsages, Features, Limits};

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    let webgpu = WebGpu::new(Features::default(), Limits::default()).await?;
    #[cfg(feature = "cuda")]
    let mut cuda = slang_hal::cuda::Cuda::new()?;
    let mut entries = vec![];
    let compiler = SlangCompiler::new(vec!["crates/stensor/shaders".into()]);

    let variants = ["Gemm", "GemmTr", "GemmFast", "GemmTrFast"];

    for dim in (128..4800).step_by(100) {
        println!("Running {dim}");
        let timing = run_gemm(&webgpu, &compiler, dim).await?;
        for k in 0..4 {
            entries.push(GemmBenchEntry {
                matrix: (dim, dim),
                backend: format!("webgpu [{}]", variants[k]),
                timing: timing[k],
            });
        }

        #[cfg(feature = "cuda")]
        {
            #[cfg(feature = "cublas")]
            {
                cuda.cublas_enabled = false;
            }

            let timing = run_gemm(&cuda, &compiler, dim).await?;
            for k in 0..4 {
                entries.push(GemmBenchEntry {
                    matrix: (dim, dim),
                    backend: format!("cuda [{}]", variants[k]),
                    timing: timing[k],
                });
            }

            #[cfg(feature = "cublas")]
            {
                cuda.cublas_enabled = true;
            }

            let timing = run_gemm(&cuda, &compiler, dim).await?;
            for k in 0..4 {
                entries.push(GemmBenchEntry {
                    matrix: (dim, dim),
                    backend: format!("cublas [{}]", variants[k]),
                    timing: timing[k],
                });
            }
        }
    }

    plot_timings(&entries);
    Ok(())
}

async fn run_gemm<B: Backend>(
    backend: &B,
    compiler: &SlangCompiler,
    dims: u32,
) -> anyhow::Result<[f32; 4]> {
    let gemm = Gemm::from_backend(backend, compiler)?;
    let mut shapes = ViewShapeBuffers::new(backend);

    let m1_cpu = DMatrix::<f32>::new_random(dims as usize, dims as usize);
    let m2_cpu = DMatrix::<f32>::new_random(dims as usize, dims as usize);
    let lhs_cpu = DMatrix::<f32>::zeros(dims as usize, dims as usize);
    let mut gpu_result = DMatrix::<f32>::zeros(dims as usize, dims as usize);

    let m1 = GpuTensor::matrix(backend, &m1_cpu, BufferUsages::STORAGE)?;
    let m2 = GpuTensor::matrix(backend, &m2_cpu, BufferUsages::STORAGE)?;
    let result = GpuTensor::matrix(
        backend,
        &lhs_cpu,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;
    let mut timing = [0.0; 4];

    for (i, variant) in [
        GemmVariant::Gemm,
        GemmVariant::GemmTr,
        GemmVariant::GemmFast,
        GemmVariant::GemmTrFast,
    ]
    .into_iter()
    .enumerate()
    {
        let t0 = std::time::Instant::now();
        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        gemm.dispatch_generic(backend, &mut shapes, &mut pass, &result, &m1, &m2, variant)?;
        drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

        backend.submit(encoder)?;
        backend.synchronize()?;
        timing[i] = t0.elapsed().as_secs_f32();
        backend
            .slow_read_buffer(result.buffer(), gpu_result.as_mut_slice())
            .await?;

        // let cpu_result = match variant {
        //     GemmVariant::Gemm | GemmVariant::GemmFast => &m1_cpu * &m2_cpu,
        //     GemmVariant::GemmTr | GemmVariant::GemmTrFast => m1_cpu.tr_mul(&m2_cpu),
        // };
        //
        // assert_relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-3);
    }

    Ok(timing)
}

#[derive(Debug, Clone)]
struct GemmBenchEntry {
    matrix: (u32, u32),
    backend: String,
    timing: f32,
}

fn plot_timings(gemm: &[GemmBenchEntry]) {
    use plotly::{
        Plot, Scatter,
        common::{Line, Mode},
        layout::Layout,
    };

    let layout = Layout::new()
        .title("slml matmul benches")
        .width(2000)
        .height(500);
    let mut plot = Plot::new();

    let mut gemm_entries: IndexMap<_, (Vec<_>, Vec<_>)> = IndexMap::new();
    for entry in gemm {
        let (x, y) = gemm_entries
            .entry(entry.backend.clone())
            .or_insert((Vec::new(), Vec::new()));
        x.push(entry.matrix.0);
        y.push(entry.timing);
    }

    println!("{:?}", gemm_entries);

    for (key, values) in gemm_entries {
        let trace = Scatter::new(values.0, values.1)
            .mode(Mode::LinesMarkersText)
            .name(format!("{:?}-f32", key))
            .line(Line::new().width(3.0));
        plot.add_trace(trace);
    }

    plot.set_layout(layout);
    plot.write_html("gemm_plot.html");
    plot.show();
}
