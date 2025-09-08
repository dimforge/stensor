import shape;

static const int WORKGROUP_SIZE = 64;

groupshared float4x4 sketch[WORKGROUP_SIZE];


func reduce_sum(index: uint, stride: uint) {
    if (index < stride) {
        sketch[index] += sketch[index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(1, WORKGROUP_SIZE, 1)]
void gemm_fast(
    uint3 workgroup_id: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m1,
    ConstantBuffer<Shape> shape_m2,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m1,
    StructuredBuffer<float4> m2,
) {
    let local_id = local_id.y;

    out[0] = 1.0;

    for (var k = 0u; k < shape_m2.ncols; k += 4u) {
        var sum = float4x4(0.0);

        for (var j = 0u; j < shape_m1.ncols; j += 4u * WORKGROUP_SIZE) {
            var ia1 = shape_m1.it(workgroup_id.x, j + local_id * 4u, workgroup_id.y);
            let ib1 = ia1 + shape_m1.col_stride;
            let ic1 = ib1 + shape_m1.col_stride;
            let id1 = ic1 + shape_m1.col_stride;
            let submat1 = float4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

            let ia2 = shape_m2.it(j / 4u + local_id, k, workgroup_id.y);
            let ib2 = ia2 + shape_m2.col_stride;
            let ic2 = ib2 + shape_m2.col_stride;
            let id2 = ic2 + shape_m2.col_stride;
            let submat2 = float4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

            sum += mul(submat2, submat1);
        }

        sketch[local_id] = sum;

        GroupMemoryBarrierWithGroupSync();

        reduce_sum(local_id, 32u);
        reduce_sum(local_id, 16u);
        reduce_sum(local_id, 8u);
        reduce_sum(local_id, 4u);
        reduce_sum(local_id, 2u);
        reduce_sum(local_id, 1u);

        if (local_id == 0u) {
            let i_out = shape_out.it(workgroup_id.x, k, workgroup_id.y);
            let mat = sketch[0];
            out[i_out] = mat[0];
            out[i_out + shape_out.col_stride] = mat[1];
            out[i_out + shape_out.col_stride * 2] = mat[2];
            out[i_out + shape_out.col_stride * 3] = mat[3];
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemm(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m1,
    ConstantBuffer<Shape> shape_m2,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m1,
    StructuredBuffer<float4> m2,

) {
    if (invocation_id.x < shape_m1.nrows) {
        for (var k = 0u; k < shape_m2.ncols; k += 4u) {
            var sum = float4x4(0.0);

            for (var j = 0u; j < shape_m1.ncols; j += 4u) {
                let ia1 = shape_m1.it(invocation_id.x, j, invocation_id.y);
                let ib1 = ia1 + shape_m1.col_stride;
                let ic1 = ib1 + shape_m1.col_stride;
                let id1 = ic1 + shape_m1.col_stride;
                let submat1 = float4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

                let ia2 = shape_m2.it(j / 4u, k, invocation_id.y);
                let ib2 = ia2 + shape_m2.col_stride;
                let ic2 = ib2 + shape_m2.col_stride;
                let id2 = ic2 + shape_m2.col_stride;
                let submat2 = float4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

                sum += mul(submat2, submat1);
            }

            let i_out = shape_out.it(invocation_id.x, k, invocation_id.y);
            out[i_out] = sum[0];
            out[i_out + shape_out.col_stride] = sum[1];
            out[i_out + shape_out.col_stride * 2] = sum[2];
            out[i_out + shape_out.col_stride * 3] = sum[3];
        }
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemm_tr(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m1,
    ConstantBuffer<Shape> shape_m2,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m1,
    StructuredBuffer<float4> m2,
) {
    if (invocation_id.x < (shape_m1.ncols + 3u) / 4) {
        for (var k = 0u; k < shape_m2.ncols; k += 4u) {
            var sum = float4x4(0.0);

            for (var j = 0u; j < shape_m1.nrows; j++) {
                var ia1 = shape_m1.it(j, invocation_id.x * 4u, invocation_id.y);
                let ib1 = ia1 + shape_m1.col_stride;
                let ic1 = ib1 + shape_m1.col_stride;
                let id1 = ic1 + shape_m1.col_stride;
                let submat1 = float4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

                let ia2 = shape_m2.it(j, k, invocation_id.y);
                let ib2 = ia2 + shape_m2.col_stride;
                let ic2 = ib2 + shape_m2.col_stride;
                let id2 = ic2 + shape_m2.col_stride;
                let submat2 = float4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

                sum += mul(submat2, transpose(submat1));
            }

            let i_out = shape_out.it(invocation_id.x, k, invocation_id.y);
            out[i_out] = sum[0];
            out[i_out + shape_out.col_stride] = sum[1];
            out[i_out + shape_out.col_stride * 2] = sum[2];
            out[i_out + shape_out.col_stride * 3] = sum[3];
        }
    }
}

[shader("compute")]
[numthreads(1, WORKGROUP_SIZE, 1)]
void gemm_tr_fast(
    uint3 workgroup_id: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m1,
    ConstantBuffer<Shape> shape_m2,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<float4> m1,
    StructuredBuffer<float4> m2,
) {
    let local_id = local_id.y;

    for (var k = 0u; k < shape_m2.ncols; k += 4u) {
        var sum = float4x4(0.0);

        for (var j = 0u; j < shape_m1.nrows; j += WORKGROUP_SIZE) {
            var ia1 = shape_m1.it(j + local_id, workgroup_id.x * 4u, workgroup_id.y);
            let ib1 = ia1 + shape_m1.col_stride;
            let ic1 = ib1 + shape_m1.col_stride;
            let id1 = ic1 + shape_m1.col_stride;
            let submat1 = float4x4(m1[ia1], m1[ib1], m1[ic1], m1[id1]);

            let ia2 = shape_m2.it(j + local_id, k, workgroup_id.y);
            let ib2 = ia2 + shape_m2.col_stride;
            let ic2 = ib2 + shape_m2.col_stride;
            let id2 = ic2 + shape_m2.col_stride;
            let submat2 = float4x4(m2[ia2], m2[ib2], m2[ic2], m2[id2]);

            sum += mul(submat2, transpose(submat1));
        }

        sketch[local_id] = sum;

        GroupMemoryBarrierWithGroupSync();

        reduce_sum(local_id, 32u);
        reduce_sum(local_id, 16u);
        reduce_sum(local_id, 8u);
        reduce_sum(local_id, 4u);
        reduce_sum(local_id, 2u);
        reduce_sum(local_id, 1u);

        if (local_id == 0u) {
            let i_out = shape_out.it(workgroup_id.x, k, workgroup_id.y);
            let mat = sketch[0];
            out[i_out] = mat[0];
            out[i_out + shape_out.col_stride] = mat[1];
            out[i_out + shape_out.col_stride * 2] = mat[2];
            out[i_out + shape_out.col_stride * 3] = mat[3];
        }

        GroupMemoryBarrierWithGroupSync();
    }
}