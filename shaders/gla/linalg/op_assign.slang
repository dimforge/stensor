import shape;
import gla.utils.limits;

static const int WORKGROUP_SIZE = 256;
// TODO: slang doesn’t have a way to get the total number of workgroups?
//       The max here is based on webgpu’s limitations.
static const uint MAX_NUM_THREADS = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;

interface IBinOp {
    [ForceInline]
    static func binop(a: float, b: float) -> float;
}

struct Add: IBinOp {
    [ForceInline]
    static func binop(a: float, b: float) -> float {
        return a + b;
    }
}

struct Sub: IBinOp {
    [ForceInline]
    static func binop(a: float, b: float) -> float {
        return a - b;
    }
}

struct Mul: IBinOp {
    [ForceInline]
    static func binop(a: float, b: float) -> float {
        return a * b;
    }
}

struct Div: IBinOp {
    [ForceInline]
    static func binop(a: float, b: float) -> float {
        return a / b;
    }
}

struct Copy: IBinOp {
    [ForceInline]
    static func binop(a: float, b: float) -> float {
        return b;
    }
}

[ForceInline]
func main<Op: IBinOp>(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_a.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it(id);
        let ib = shape_b.it_wrapping(id);
        a[ia] = Op.binop(a[ia], b[ib]);
    }
}

// NOTE: keep it this way or do link-time specialization?
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func add(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    main<Add>(invocation_id, shape_a, shape_b, a, b);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sub(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    main<Sub>(invocation_id, shape_a, shape_b, a, b);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func mul(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    main<Mul>(invocation_id, shape_a, shape_b, a, b);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func div(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    main<Div>(invocation_id, shape_a, shape_b, a, b);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func copy(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_a.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it(id);
        let ib = shape_b.it_wrapping(id);
        a[ia] = b[ib];
    }
}

struct BinOpOffsets {
    uint a;
    uint b;
    uint pad0;
    uint pad1;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func copy_with_offsets(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<BinOpOffsets> offsets,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> a,
    StructuredBuffer<float> b,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_a.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_a.decompose(thread_id);
        let ia = shape_a.it(id);
        let ib = shape_b.it_wrapping(id);
        a[offsets.a + ia] = b[offsets.b + ib];
    }
}
