import shape;
import stensor.utils.limits;

static const int WORKGROUP_SIZE = 128;
// TODO: slang doesn’t have a way to get the total number of workgroups?
//       The max here is based on webgpu’s limitations.
static const uint MAX_NUM_THREADS = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void contiguous_col_major(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    RWStructuredBuffer<float> out,
    StructuredBuffer<float> tensor,
) {
    for (var thread_id = invocation_id.x; thread_id < shape.len(); thread_id += MAX_NUM_THREADS) {
        // Compute the corresponding (i, j, k, l) indices for the out tensor.
        // A contiguous column-major tensor has stride:
        // [1, nrows, nrows * ncols, nrows * ncols * nmats]
        let col_stride = shape.nrows;
        let mat_stride = shape.nrows * shape.ncols;
        let cube_stride = shape.nrows * shape.ncols * shape.nmats;

        let l = thread_id / cube_stride;
        let k = (thread_id % cube_stride) / mat_stride;
        let j = (thread_id % mat_stride) / col_stride;
        let i = thread_id % col_stride;

        out[thread_id] = tensor[shape.it(i, j, k, l)];
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void contiguous_row_major(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    RWStructuredBuffer<float> out,
    StructuredBuffer<float> tensor,
) {
    for (var thread_id = invocation_id.x; thread_id < shape.len(); thread_id += MAX_NUM_THREADS) {
        // Compute the corresponding (i, j, k, l) indices for the out tensor.
        // A contiguous row-major tensor has stride:
        // [ncols, 1, nrows * ncols, nrows * ncols * nmats]
        let row_stride = shape.ncols;
        let mat_stride = shape.nrows * shape.ncols;
        let cube_stride = shape.nrows * shape.ncols * shape.nmats;

        let l = thread_id / cube_stride;
        let k = (thread_id % cube_stride) / mat_stride;
        let i = (thread_id % mat_stride) / row_stride;
        let j = thread_id % row_stride;

        out[thread_id] = tensor[shape.it(i, j, k, l)];
    }
}