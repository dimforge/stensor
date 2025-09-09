import shape;

static const int WORKGROUP_SIZE = 128;
static const float MAX_FLT = 3.402823466e+38F;
static const float MIN_FLT = -3.402823466e+38F;

interface IReduction {
    static func init() -> float;
    static func reduce_buffer(a: float, b: float) -> float;
    static func reduce_workspace(a: float, b: float) -> float;
}

struct Sum: IReduction {
    static func init() -> float {
        return 0.0;
    }
    static func reduce_buffer(a: float, b: float) -> float {
        return a + b;
    }
    static func reduce_workspace(a: float, b: float) -> float {
        return a + b;
    }
}

struct Prod: IReduction {
    static func init() -> float {
        return 1.0;
    }
    static func reduce_buffer(a: float, b: float) -> float {
        return a * b;
    }
    static func reduce_workspace(a: float, b: float) -> float {
        return a * b;
    }
}

struct Min: IReduction {
    static func init() -> float {
        return MAX_FLT;
    }
    static func reduce_buffer(a: float, b: float) -> float {
        return min(a, b);
    }
    static func reduce_workspace(a: float, b: float) -> float {
        return min(a, b);
    }
}

struct Max: IReduction {
    static func init() -> float {
        return MIN_FLT;
    }
    static func reduce_buffer(a: float, b: float) -> float {
        return max(a, b);
    }
    static func reduce_workspace(a: float, b: float) -> float {
        return max(a, b);
    }
}

struct SqNorm: IReduction {
    static func init() -> float {
        return 0.0;
    }
    static func reduce_buffer(a: float, b: float) -> float {
        return a + b * b;
    }
    static func reduce_workspace(a: float, b: float) -> float {
        return a + b;
    }
}

groupshared float workspace[WORKGROUP_SIZE];

func reduce<R: IReduction>(thread_id: uint, stride: uint) {
    if (thread_id < stride) {
        workspace[thread_id] = R.reduce_workspace(workspace[thread_id], workspace[thread_id + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
}

func run_reduction<R: IReduction>(
    thread_id: uint,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
) -> float {
    workspace[thread_id] = R.init();

    for (var i = thread_id; i < shape.nrows; i += WORKGROUP_SIZE) {
        let val_i = input[shape.iv(i)];
        workspace[thread_id] = R.reduce_buffer(workspace[thread_id], val_i);
    }

    GroupMemoryBarrierWithGroupSync();

    reduce<R>(thread_id, 64u);
    reduce<R>(thread_id, 32u);
    reduce<R>(thread_id, 16u);
    reduce<R>(thread_id, 8u);
    reduce<R>(thread_id, 4u);
    reduce<R>(thread_id, 2u);
    reduce<R>(thread_id, 1u);

    return workspace[0];
}

void main<R: IReduction>(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
) {
    let result = run_reduction<R>(invocation_id.x, shape, input);

    if (invocation_id.x == 0) {
        output[0] = result;
    }
}

// TODO: keep it this way or do link-time specialization?
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void reduce_sum(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
 ) {
    main<Sum>(invocation_id, shape, input, output);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void reduce_product(uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
 ) {
    main<Prod>(invocation_id, shape, input, output);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void reduce_min(uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
 ) {
    main<Min>(invocation_id, shape, input, output);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void reduce_max(uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
 ) {
    main<Max>(invocation_id, shape, input, output);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void reduce_sqnorm(uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
 ) {
    main<SqNorm>(invocation_id, shape, input, output);
}
