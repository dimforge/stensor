import shape;

static const int WORKGROUP_SIZE = 128;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func repeat(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_result,
    ConstantBuffer<Shape> shape_source,
    RWStructuredBuffer<float> result,
    StructuredBuffer<float> source,
) {
    if (invocation_id.x >= shape_result.len()) {
        return;
    }

    let id = shape_result.decompose(invocation_id.x);

    // TODO PERF: ensure we are doing coalesced reads
    let ia = shape_result.it(id);
    let ib = shape_source.it_wrapping(id);
    result[ia] = source[ib];
}
