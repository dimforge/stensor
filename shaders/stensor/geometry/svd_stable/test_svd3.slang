module test_svd3;

import stensor.geometry.svd_stable.svd3;

[shader("compute")]
[numthreads(1, 1, 1)]
void test_svd3(
    uint3 id: SV_DispatchThreadID,
    RWStructuredBuffer<float> inputs,
    RWStructuredBuffer<float> outputs,
) {
    let i = id.x;
    let base_in = i * 9;

    // Read 3x3 matrix from flat buffer (column-major) into row-major float3x3.
    // m[row] = (buf[0*3+row], buf[1*3+row], buf[2*3+row])
    float3x3 m = float3x3(
        float3(inputs[base_in + 0], inputs[base_in + 3], inputs[base_in + 6]),
        float3(inputs[base_in + 1], inputs[base_in + 4], inputs[base_in + 7]),
        float3(inputs[base_in + 2], inputs[base_in + 5], inputs[base_in + 8])
    );

    let result = Svd3(m);

    // Write results: U (9 floats) + S (3 floats) + Vt (9 floats) = 21 floats per matrix.
    let base_out = i * 21;

    // U (row-major float3x3 → column-major buffer): buf[col*3+row] = U[row][col]
    outputs[base_out + 0] = result.U[0].x;
    outputs[base_out + 1] = result.U[1].x;
    outputs[base_out + 2] = result.U[2].x;
    outputs[base_out + 3] = result.U[0].y;
    outputs[base_out + 4] = result.U[1].y;
    outputs[base_out + 5] = result.U[2].y;
    outputs[base_out + 6] = result.U[0].z;
    outputs[base_out + 7] = result.U[1].z;
    outputs[base_out + 8] = result.U[2].z;

    // S
    outputs[base_out + 9] = result.S.x;
    outputs[base_out + 10] = result.S.y;
    outputs[base_out + 11] = result.S.z;

    // Vt (row-major float3x3 → column-major buffer): buf[col*3+row] = Vt[row][col]
    outputs[base_out + 12] = result.Vt[0].x;
    outputs[base_out + 13] = result.Vt[1].x;
    outputs[base_out + 14] = result.Vt[2].x;
    outputs[base_out + 15] = result.Vt[0].y;
    outputs[base_out + 16] = result.Vt[1].y;
    outputs[base_out + 17] = result.Vt[2].y;
    outputs[base_out + 18] = result.Vt[0].z;
    outputs[base_out + 19] = result.Vt[1].z;
    outputs[base_out + 20] = result.Vt[2].z;
}
