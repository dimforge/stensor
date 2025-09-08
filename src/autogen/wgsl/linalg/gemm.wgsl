struct Shape_std140_0
{
    @align(16) nrows_0 : u32,
    @align(4) ncols_0 : u32,
    @align(8) nmats_0 : u32,
    @align(4) ncubes_0 : u32,
    @align(16) row_stride_0 : u32,
    @align(4) col_stride_0 : u32,
    @align(8) mat_stride_0 : u32,
    @align(4) cube_stride_0 : u32,
};

@binding(0) @group(0) var<uniform> entryPointParams_shape_out_0 : Shape_std140_0;
@binding(1) @group(0) var<uniform> entryPointParams_shape_m1_0 : Shape_std140_0;
@binding(2) @group(0) var<uniform> entryPointParams_shape_m2_0 : Shape_std140_0;
@binding(3) @group(0) var<storage, read_write> entryPointParams_out_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> entryPointParams_m1_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> entryPointParams_m2_0 : array<vec4<f32>>;

@binding(6) @group(0) var<uniform> entryPointParams_shape_out_1 : Shape_std140_0;
@binding(7) @group(0) var<uniform> entryPointParams_shape_m1_1 : Shape_std140_0;
@binding(8) @group(0) var<uniform> entryPointParams_shape_m2_1 : Shape_std140_0;
@binding(9) @group(0) var<storage, read_write> entryPointParams_out_1 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> entryPointParams_m1_1 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read> entryPointParams_m2_1 : array<vec4<f32>>;

@binding(12) @group(0) var<uniform> entryPointParams_shape_out_2 : Shape_std140_0;
@binding(13) @group(0) var<uniform> entryPointParams_shape_m1_2 : Shape_std140_0;
@binding(14) @group(0) var<uniform> entryPointParams_shape_m2_2 : Shape_std140_0;
@binding(15) @group(0) var<storage, read_write> entryPointParams_out_2 : array<vec4<f32>>;

@binding(16) @group(0) var<storage, read> entryPointParams_m1_2 : array<vec4<f32>>;

@binding(17) @group(0) var<storage, read> entryPointParams_m2_2 : array<vec4<f32>>;

@binding(18) @group(0) var<uniform> entryPointParams_shape_out_3 : Shape_std140_0;
@binding(19) @group(0) var<uniform> entryPointParams_shape_m1_3 : Shape_std140_0;
@binding(20) @group(0) var<uniform> entryPointParams_shape_m2_3 : Shape_std140_0;
@binding(21) @group(0) var<storage, read_write> entryPointParams_out_3 : array<vec4<f32>>;

@binding(22) @group(0) var<storage, read> entryPointParams_m1_3 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read> entryPointParams_m2_3 : array<vec4<f32>>;

var<workgroup> sketch_0 : array<mat4x4<f32>, i32(64)>;

fn reduce_sum_0( index_0 : u32,  stride_0 : u32)
{
    if(index_0 < stride_0)
    {
        sketch_0[index_0] = sketch_0[index_0] + sketch_0[index_0 + stride_0];
    }
    workgroupBarrier();
    return;
}

fn Shape_it_0( _S1 : u32,  _S2 : u32,  _S3 : u32) -> u32
{
    return _S1 * entryPointParams_shape_m1_0.row_stride_0 + _S2 * entryPointParams_shape_m1_0.col_stride_0 + _S3 * entryPointParams_shape_m1_0.mat_stride_0;
}

fn Shape_it_1( _S4 : u32,  _S5 : u32,  _S6 : u32) -> u32
{
    return _S4 * entryPointParams_shape_m2_0.row_stride_0 + _S5 * entryPointParams_shape_m2_0.col_stride_0 + _S6 * entryPointParams_shape_m2_0.mat_stride_0;
}

fn Shape_it_2( _S7 : u32,  _S8 : u32,  _S9 : u32) -> u32
{
    return _S7 * entryPointParams_shape_out_0.row_stride_0 + _S8 * entryPointParams_shape_out_0.col_stride_0 + _S9 * entryPointParams_shape_out_0.mat_stride_0;
}

@compute
@workgroup_size(1, 64, 1)
fn gemm_fast(@builtin(workgroup_id) workgroup_id_0 : vec3<u32>, @builtin(local_invocation_id) local_id_0 : vec3<u32>)
{
    var _S10 : u32 = local_id_0.y;
    entryPointParams_out_0[i32(0)] = vec4<f32>(1.0f);
    var k_0 : u32 = u32(0);
    for(;;)
    {
        if(k_0 < (entryPointParams_shape_m2_0.ncols_0))
        {
        }
        else
        {
            break;
        }
        var _S11 : mat4x4<f32> = mat4x4<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        var j_0 : u32 = u32(0);
        var sum_0 : mat4x4<f32> = _S11;
        for(;;)
        {
            if(j_0 < (entryPointParams_shape_m1_0.ncols_0))
            {
            }
            else
            {
                break;
            }
            var _S12 : u32 = workgroup_id_0.y;
            var _S13 : u32 = Shape_it_0(workgroup_id_0.x, j_0 + _S10 * u32(4), _S12);
            var _S14 : u32 = _S13 + entryPointParams_shape_m1_0.col_stride_0;
            var _S15 : u32 = _S14 + entryPointParams_shape_m1_0.col_stride_0;
            var _S16 : u32 = Shape_it_1(j_0 / u32(4) + _S10, k_0, _S12);
            var _S17 : u32 = _S16 + entryPointParams_shape_m2_0.col_stride_0;
            var _S18 : u32 = _S17 + entryPointParams_shape_m2_0.col_stride_0;
            var sum_1 : mat4x4<f32> = sum_0 + (((mat4x4<f32>(entryPointParams_m1_0[_S13], entryPointParams_m1_0[_S14], entryPointParams_m1_0[_S15], entryPointParams_m1_0[_S15 + entryPointParams_shape_m1_0.col_stride_0])) * (mat4x4<f32>(entryPointParams_m2_0[_S16], entryPointParams_m2_0[_S17], entryPointParams_m2_0[_S18], entryPointParams_m2_0[_S18 + entryPointParams_shape_m2_0.col_stride_0]))));
            j_0 = j_0 + u32(256);
            sum_0 = sum_1;
        }
        sketch_0[_S10] = sum_0;
        workgroupBarrier();
        reduce_sum_0(_S10, u32(32));
        reduce_sum_0(_S10, u32(16));
        reduce_sum_0(_S10, u32(8));
        reduce_sum_0(_S10, u32(4));
        reduce_sum_0(_S10, u32(2));
        reduce_sum_0(_S10, u32(1));
        if(_S10 == u32(0))
        {
            var _S19 : u32 = Shape_it_2(workgroup_id_0.x, k_0, workgroup_id_0.y);
            var _S20 : mat4x4<f32> = sketch_0[i32(0)];
            entryPointParams_out_0[_S19] = sketch_0[i32(0)][i32(0)];
            entryPointParams_out_0[_S19 + entryPointParams_shape_out_0.col_stride_0] = _S20[i32(1)];
            entryPointParams_out_0[_S19 + entryPointParams_shape_out_0.col_stride_0 * u32(2)] = _S20[i32(2)];
            entryPointParams_out_0[_S19 + entryPointParams_shape_out_0.col_stride_0 * u32(3)] = _S20[i32(3)];
        }
        workgroupBarrier();
        k_0 = k_0 + u32(4);
    }
    return;
}

fn Shape_it_3( _S21 : u32,  _S22 : u32,  _S23 : u32) -> u32
{
    return _S21 * entryPointParams_shape_m1_1.row_stride_0 + _S22 * entryPointParams_shape_m1_1.col_stride_0 + _S23 * entryPointParams_shape_m1_1.mat_stride_0;
}

fn Shape_it_4( _S24 : u32,  _S25 : u32,  _S26 : u32) -> u32
{
    return _S24 * entryPointParams_shape_m2_1.row_stride_0 + _S25 * entryPointParams_shape_m2_1.col_stride_0 + _S26 * entryPointParams_shape_m2_1.mat_stride_0;
}

fn Shape_it_5( _S27 : u32,  _S28 : u32,  _S29 : u32) -> u32
{
    return _S27 * entryPointParams_shape_out_1.row_stride_0 + _S28 * entryPointParams_shape_out_1.col_stride_0 + _S29 * entryPointParams_shape_out_1.mat_stride_0;
}

@compute
@workgroup_size(64, 1, 1)
fn gemm(@builtin(global_invocation_id) invocation_id_0 : vec3<u32>)
{
    var _S30 : u32 = invocation_id_0.x;
    if(_S30 < (entryPointParams_shape_m1_1.nrows_0))
    {
        var k_1 : u32 = u32(0);
        for(;;)
        {
            if(k_1 < (entryPointParams_shape_m2_1.ncols_0))
            {
            }
            else
            {
                break;
            }
            var _S31 : mat4x4<f32> = mat4x4<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            var j_1 : u32 = u32(0);
            var sum_2 : mat4x4<f32> = _S31;
            for(;;)
            {
                if(j_1 < (entryPointParams_shape_m1_1.ncols_0))
                {
                }
                else
                {
                    break;
                }
                var _S32 : u32 = invocation_id_0.y;
                var _S33 : u32 = Shape_it_3(_S30, j_1, _S32);
                var _S34 : u32 = _S33 + entryPointParams_shape_m1_1.col_stride_0;
                var _S35 : u32 = _S34 + entryPointParams_shape_m1_1.col_stride_0;
                var _S36 : u32 = Shape_it_4(j_1 / u32(4), k_1, _S32);
                var _S37 : u32 = _S36 + entryPointParams_shape_m2_1.col_stride_0;
                var _S38 : u32 = _S37 + entryPointParams_shape_m2_1.col_stride_0;
                var sum_3 : mat4x4<f32> = sum_2 + (((mat4x4<f32>(entryPointParams_m1_1[_S33], entryPointParams_m1_1[_S34], entryPointParams_m1_1[_S35], entryPointParams_m1_1[_S35 + entryPointParams_shape_m1_1.col_stride_0])) * (mat4x4<f32>(entryPointParams_m2_1[_S36], entryPointParams_m2_1[_S37], entryPointParams_m2_1[_S38], entryPointParams_m2_1[_S38 + entryPointParams_shape_m2_1.col_stride_0]))));
                j_1 = j_1 + u32(4);
                sum_2 = sum_3;
            }
            var _S39 : u32 = Shape_it_5(_S30, k_1, invocation_id_0.y);
            entryPointParams_out_1[_S39] = sum_2[i32(0)];
            entryPointParams_out_1[_S39 + entryPointParams_shape_out_1.col_stride_0] = sum_2[i32(1)];
            entryPointParams_out_1[_S39 + entryPointParams_shape_out_1.col_stride_0 * u32(2)] = sum_2[i32(2)];
            entryPointParams_out_1[_S39 + entryPointParams_shape_out_1.col_stride_0 * u32(3)] = sum_2[i32(3)];
            k_1 = k_1 + u32(4);
        }
    }
    return;
}

fn Shape_it_6( _S40 : u32,  _S41 : u32,  _S42 : u32) -> u32
{
    return _S40 * entryPointParams_shape_m1_2.row_stride_0 + _S41 * entryPointParams_shape_m1_2.col_stride_0 + _S42 * entryPointParams_shape_m1_2.mat_stride_0;
}

fn Shape_it_7( _S43 : u32,  _S44 : u32,  _S45 : u32) -> u32
{
    return _S43 * entryPointParams_shape_m2_2.row_stride_0 + _S44 * entryPointParams_shape_m2_2.col_stride_0 + _S45 * entryPointParams_shape_m2_2.mat_stride_0;
}

fn Shape_it_8( _S46 : u32,  _S47 : u32,  _S48 : u32) -> u32
{
    return _S46 * entryPointParams_shape_out_2.row_stride_0 + _S47 * entryPointParams_shape_out_2.col_stride_0 + _S48 * entryPointParams_shape_out_2.mat_stride_0;
}

@compute
@workgroup_size(64, 1, 1)
fn gemm_tr(@builtin(global_invocation_id) invocation_id_1 : vec3<u32>)
{
    var _S49 : u32 = invocation_id_1.x;
    if(_S49 < ((entryPointParams_shape_m1_2.ncols_0 + u32(3)) / u32(4)))
    {
        var k_2 : u32 = u32(0);
        for(;;)
        {
            if(k_2 < (entryPointParams_shape_m2_2.ncols_0))
            {
            }
            else
            {
                break;
            }
            var _S50 : mat4x4<f32> = mat4x4<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            var j_2 : u32 = u32(0);
            var sum_4 : mat4x4<f32> = _S50;
            for(;;)
            {
                if(j_2 < (entryPointParams_shape_m1_2.nrows_0))
                {
                }
                else
                {
                    break;
                }
                var _S51 : u32 = invocation_id_1.y;
                var _S52 : u32 = Shape_it_6(j_2, _S49 * u32(4), _S51);
                var _S53 : u32 = _S52 + entryPointParams_shape_m1_2.col_stride_0;
                var _S54 : u32 = _S53 + entryPointParams_shape_m1_2.col_stride_0;
                var _S55 : u32 = Shape_it_7(j_2, k_2, _S51);
                var _S56 : u32 = _S55 + entryPointParams_shape_m2_2.col_stride_0;
                var _S57 : u32 = _S56 + entryPointParams_shape_m2_2.col_stride_0;
                var sum_5 : mat4x4<f32> = sum_4 + (((transpose(mat4x4<f32>(entryPointParams_m1_2[_S52], entryPointParams_m1_2[_S53], entryPointParams_m1_2[_S54], entryPointParams_m1_2[_S54 + entryPointParams_shape_m1_2.col_stride_0]))) * (mat4x4<f32>(entryPointParams_m2_2[_S55], entryPointParams_m2_2[_S56], entryPointParams_m2_2[_S57], entryPointParams_m2_2[_S57 + entryPointParams_shape_m2_2.col_stride_0]))));
                j_2 = j_2 + u32(1);
                sum_4 = sum_5;
            }
            var _S58 : u32 = Shape_it_8(_S49, k_2, invocation_id_1.y);
            entryPointParams_out_2[_S58] = sum_4[i32(0)];
            entryPointParams_out_2[_S58 + entryPointParams_shape_out_2.col_stride_0] = sum_4[i32(1)];
            entryPointParams_out_2[_S58 + entryPointParams_shape_out_2.col_stride_0 * u32(2)] = sum_4[i32(2)];
            entryPointParams_out_2[_S58 + entryPointParams_shape_out_2.col_stride_0 * u32(3)] = sum_4[i32(3)];
            k_2 = k_2 + u32(4);
        }
    }
    return;
}

fn Shape_it_9( _S59 : u32,  _S60 : u32,  _S61 : u32) -> u32
{
    return _S59 * entryPointParams_shape_m1_3.row_stride_0 + _S60 * entryPointParams_shape_m1_3.col_stride_0 + _S61 * entryPointParams_shape_m1_3.mat_stride_0;
}

fn Shape_it_10( _S62 : u32,  _S63 : u32,  _S64 : u32) -> u32
{
    return _S62 * entryPointParams_shape_m2_3.row_stride_0 + _S63 * entryPointParams_shape_m2_3.col_stride_0 + _S64 * entryPointParams_shape_m2_3.mat_stride_0;
}

fn Shape_it_11( _S65 : u32,  _S66 : u32,  _S67 : u32) -> u32
{
    return _S65 * entryPointParams_shape_out_3.row_stride_0 + _S66 * entryPointParams_shape_out_3.col_stride_0 + _S67 * entryPointParams_shape_out_3.mat_stride_0;
}

@compute
@workgroup_size(1, 64, 1)
fn gemm_tr_fast(@builtin(workgroup_id) workgroup_id_1 : vec3<u32>, @builtin(local_invocation_id) local_id_1 : vec3<u32>)
{
    var _S68 : u32 = local_id_1.y;
    var k_3 : u32 = u32(0);
    for(;;)
    {
        if(k_3 < (entryPointParams_shape_m2_3.ncols_0))
        {
        }
        else
        {
            break;
        }
        var _S69 : mat4x4<f32> = mat4x4<f32>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        var j_3 : u32 = u32(0);
        var sum_6 : mat4x4<f32> = _S69;
        for(;;)
        {
            if(j_3 < (entryPointParams_shape_m1_3.nrows_0))
            {
            }
            else
            {
                break;
            }
            var _S70 : u32 = j_3 + _S68;
            var _S71 : u32 = workgroup_id_1.y;
            var _S72 : u32 = Shape_it_9(_S70, workgroup_id_1.x * u32(4), _S71);
            var _S73 : u32 = _S72 + entryPointParams_shape_m1_3.col_stride_0;
            var _S74 : u32 = _S73 + entryPointParams_shape_m1_3.col_stride_0;
            var _S75 : u32 = Shape_it_10(_S70, k_3, _S71);
            var _S76 : u32 = _S75 + entryPointParams_shape_m2_3.col_stride_0;
            var _S77 : u32 = _S76 + entryPointParams_shape_m2_3.col_stride_0;
            var sum_7 : mat4x4<f32> = sum_6 + (((transpose(mat4x4<f32>(entryPointParams_m1_3[_S72], entryPointParams_m1_3[_S73], entryPointParams_m1_3[_S74], entryPointParams_m1_3[_S74 + entryPointParams_shape_m1_3.col_stride_0]))) * (mat4x4<f32>(entryPointParams_m2_3[_S75], entryPointParams_m2_3[_S76], entryPointParams_m2_3[_S77], entryPointParams_m2_3[_S77 + entryPointParams_shape_m2_3.col_stride_0]))));
            j_3 = j_3 + u32(64);
            sum_6 = sum_7;
        }
        sketch_0[_S68] = sum_6;
        workgroupBarrier();
        reduce_sum_0(_S68, u32(32));
        reduce_sum_0(_S68, u32(16));
        reduce_sum_0(_S68, u32(8));
        reduce_sum_0(_S68, u32(4));
        reduce_sum_0(_S68, u32(2));
        reduce_sum_0(_S68, u32(1));
        if(_S68 == u32(0))
        {
            var _S78 : u32 = Shape_it_11(workgroup_id_1.x, k_3, workgroup_id_1.y);
            var _S79 : mat4x4<f32> = sketch_0[i32(0)];
            entryPointParams_out_3[_S78] = sketch_0[i32(0)][i32(0)];
            entryPointParams_out_3[_S78 + entryPointParams_shape_out_3.col_stride_0] = _S79[i32(1)];
            entryPointParams_out_3[_S78 + entryPointParams_shape_out_3.col_stride_0 * u32(2)] = _S79[i32(2)];
            entryPointParams_out_3[_S78 + entryPointParams_shape_out_3.col_stride_0 * u32(3)] = _S79[i32(3)];
        }
        workgroupBarrier();
        k_3 = k_3 + u32(4);
    }
    return;
}

