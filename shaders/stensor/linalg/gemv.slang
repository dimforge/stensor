import shape;

static const int WORKGROUP_SIZE = 32;
static const int MAX_WORKGROUP_SIZE = 65535;

groupshared float4 sketch[WORKGROUP_SIZE];

func reduce_sum(index: uint, stride: uint) {
    if (index < stride) {
        sketch[index] += sketch[index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(1, WORKGROUP_SIZE, 1)]
void gemv_fast(
    uint3 workgroup_id: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m,
    StructuredBuffer<float4> v,
) {
    let local_id = local_id.y;


    for (var l = 0u; l < shape_out.ncubes; l += 1) {
        var sum = float4(0.0);

        for (var j = 0u; j < shape_m.ncols; j += 4u * WORKGROUP_SIZE) {
            var ia = shape_m.it_wrapping(workgroup_id.x, j + local_id * 4u, workgroup_id.z, l);
            let ib = ia + shape_m.col_stride;
            let ic = ib + shape_m.col_stride;
            let id = ic + shape_m.col_stride;
            let submat = float4x4(m[ia], m[ib], m[ic], m[id]);

            let iv = shape_v.it_wrapping(j / 4u + local_id, workgroup_id.y, workgroup_id.z, l);
            sum += mul(v[iv], submat);
        }

        sketch[local_id] = sum;

        GroupMemoryBarrierWithGroupSync();

    //    reduce_sum(local_id, 32u);
        reduce_sum(local_id, 16u);
        reduce_sum(local_id, 8u);
        reduce_sum(local_id, 4u);
        reduce_sum(local_id, 2u);
        reduce_sum(local_id, 1u);

        if (local_id == 0u) {
            let i_out = shape_out.it(workgroup_id.x, workgroup_id.y, workgroup_id.z, l);
            out[i_out] = sketch[0];
        }
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv(
    uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_out,
        ConstantBuffer<Shape> shape_m,
        ConstantBuffer<Shape> shape_v,
        RWStructuredBuffer<float4> out,
        StructuredBuffer<float4> m,
        StructuredBuffer<float4> v,
    ) {
    if (invocation_id.x < shape_m.nrows) {
        for (var l = 0u; l < shape_out.ncubes; l += 1) {
            var sum = float4(0.0);

            for (var j = 0u; j < shape_m.ncols; j += 4u) {
                var ia = shape_m.it_wrapping(invocation_id.x, j, invocation_id.z, l);
                let ib = ia + shape_m.col_stride;
                let ic = ib + shape_m.col_stride;
                let id = ic + shape_m.col_stride;
                let submat = float4x4(m[ia], m[ib], m[ic], m[id]);

                let iv = shape_v.it_wrapping(j / 4u, invocation_id.y, invocation_id.z, l);
                sum += mul(v[iv], submat);
            }

            let i_out = shape_out.it(invocation_id.x, invocation_id.y, invocation_id.z, l);
            out[i_out] = sum;
        }
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv_tr(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m,
    StructuredBuffer<float4> v,
    ) {
    if (invocation_id.x < (shape_m.ncols + 3u) / 4) {
        for (var l = 0u; l < shape_out.ncubes; l += 1) {
            var sum = float4(0.0);

            for (var j = 0u; j < shape_m.nrows; j++) {
                var ia = shape_m.it_wrapping(j, invocation_id.x * 4u, invocation_id.z, l);
                let ib = ia + shape_m.col_stride;
                let ic = ib + shape_m.col_stride;
                let id = ic + shape_m.col_stride;
                let submat = float4x4(m[ia], m[ib], m[ic], m[id]);

                let iv = shape_v.it_wrapping(j, invocation_id.y, invocation_id.z, l);
                sum += mul(submat, v[iv]);
            }

            let i_out = shape_out.it(invocation_id.x, invocation_id.y, invocation_id.z, l);
            out[i_out] = sum;
        }
    }
}

[shader("compute")]
[numthreads(1, WORKGROUP_SIZE, 1)]
void gemv_tr_fast(
    uint3 workgroup_id: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m,
    StructuredBuffer<float4> v,
) {
    let local_id = local_id.y;

    for (var l = 0u; l < shape_out.ncubes; l += 1) {
        var sum = float4(0.0);

        for (var j = 0u; j < shape_m.nrows; j += WORKGROUP_SIZE) {
            var ia = shape_m.it_wrapping(j + local_id, workgroup_id.x * 4u, workgroup_id.z, l);
            let ib = ia + shape_m.col_stride;
            let ic = ib + shape_m.col_stride;
            let id = ic + shape_m.col_stride;
            let submat = float4x4(m[ia], m[ib], m[ic], m[id]);

            let iv = shape_v.it_wrapping(j + local_id, workgroup_id.y, workgroup_id.z, l);
            sum += mul(submat, v[iv]);
        }

        sketch[local_id] = sum;

        GroupMemoryBarrierWithGroupSync();

    //    reduce_sum(local_id, 64u);
    //    reduce_sum(local_id, 32u);
        reduce_sum(local_id, 16u);
        reduce_sum(local_id, 8u);
        reduce_sum(local_id, 4u);
        reduce_sum(local_id, 2u);
        reduce_sum(local_id, 1u);

        if (local_id == 0u) {
            let i_out = shape_out.it(workgroup_id.x, workgroup_id.y, workgroup_id.z, l);
            out[i_out] = sketch[0];
        }
    }
}


/*
 * Below are naive implementations.
 * Note that they expect float instead of float4.
 */
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv_naive(
    uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_out,
        ConstantBuffer<Shape> shape_m,
        ConstantBuffer<Shape> shape_v,
        RWStructuredBuffer<float> out,
        StructuredBuffer<float> m,
        StructuredBuffer<float> v,
    ) {
    if (invocation_id.x < shape_m.nrows) {
        for (var l = 0u; l < shape_out.ncubes; l += 1) {
            var sum = 0.0;

            for (var j = 0u; j < shape_m.ncols; j += 1u) {
                var ia = shape_m.it_wrapping(invocation_id.x, j, invocation_id.z, l);
                let iv = shape_v.it_wrapping(j, invocation_id.y, invocation_id.z, l);
                sum += m[ia] * v[iv];
            }

            let i_out = shape_out.it(invocation_id.x, invocation_id.y, invocation_id.z, l);
            out[i_out] = sum;
        }
    }
}


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv_tr_naive(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float> out,
    StructuredBuffer<float> m,
    StructuredBuffer<float> v,
) {
    if (invocation_id.x < shape_out.nrows) {
        for (var l = 0u; l < shape_out.ncubes; l += 1) {
            var sum = 0.0;

            for (var j = 0u; j < shape_m.nrows; j++) {
                var ia = shape_m.it_wrapping(j, invocation_id.x, invocation_id.z, l);
                let iv = shape_v.it_wrapping(j, invocation_id.y, invocation_id.z, l);
                sum += m[ia] * v[iv];
            }

            let i_out = shape_out.it(invocation_id.x, invocation_id.y, invocation_id.z, l);
            out[i_out] = sum;
        }
    }
}