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

@binding(0) @group(0) var<uniform> entryPointParams_shape_a_0 : Shape_std140_0;
@binding(1) @group(0) var<uniform> entryPointParams_shape_b_0 : Shape_std140_0;
@binding(2) @group(0) var<storage, read_write> entryPointParams_a_0 : array<f32>;

@binding(3) @group(0) var<storage, read> entryPointParams_b_0 : array<f32>;

@binding(4) @group(0) var<uniform> entryPointParams_shape_a_1 : Shape_std140_0;
@binding(5) @group(0) var<uniform> entryPointParams_shape_b_1 : Shape_std140_0;
@binding(6) @group(0) var<storage, read_write> entryPointParams_a_1 : array<f32>;

@binding(7) @group(0) var<storage, read> entryPointParams_b_1 : array<f32>;

@binding(8) @group(0) var<uniform> entryPointParams_shape_a_2 : Shape_std140_0;
@binding(9) @group(0) var<uniform> entryPointParams_shape_b_2 : Shape_std140_0;
@binding(10) @group(0) var<storage, read_write> entryPointParams_a_2 : array<f32>;

@binding(11) @group(0) var<storage, read> entryPointParams_b_2 : array<f32>;

@binding(12) @group(0) var<uniform> entryPointParams_shape_a_3 : Shape_std140_0;
@binding(13) @group(0) var<uniform> entryPointParams_shape_b_3 : Shape_std140_0;
@binding(14) @group(0) var<storage, read_write> entryPointParams_a_3 : array<f32>;

@binding(15) @group(0) var<storage, read> entryPointParams_b_3 : array<f32>;

@binding(16) @group(0) var<uniform> entryPointParams_shape_a_4 : Shape_std140_0;
@binding(17) @group(0) var<uniform> entryPointParams_shape_b_4 : Shape_std140_0;
@binding(18) @group(0) var<storage, read_write> entryPointParams_a_4 : array<f32>;

@binding(19) @group(0) var<storage, read> entryPointParams_b_4 : array<f32>;

struct BinOpOffsets_std140_0
{
    @align(16) a_0 : u32,
    @align(4) b_0 : u32,
    @align(8) pad0_0 : u32,
    @align(4) pad1_0 : u32,
};

@binding(20) @group(0) var<uniform> entryPointParams_offsets_0 : BinOpOffsets_std140_0;
@binding(21) @group(0) var<uniform> entryPointParams_shape_a_5 : Shape_std140_0;
@binding(22) @group(0) var<uniform> entryPointParams_shape_b_5 : Shape_std140_0;
@binding(23) @group(0) var<storage, read_write> entryPointParams_a_5 : array<f32>;

@binding(24) @group(0) var<storage, read> entryPointParams_b_5 : array<f32>;

fn Shape_len_0() -> u32
{
    return entryPointParams_shape_a_0.nrows_0 * entryPointParams_shape_a_0.ncols_0 * entryPointParams_shape_a_0.nmats_0 * entryPointParams_shape_a_0.ncubes_0;
}

fn Shape_decompose_0( _S1 : u32) -> vec4<u32>
{
    var _S2 : u32 = entryPointParams_shape_a_0.nrows_0;
    var _S3 : u32 = entryPointParams_shape_a_0.ncols_0;
    var _S4 : u32 = entryPointParams_shape_a_0.nmats_0;
    var _S5 : u32 = entryPointParams_shape_a_0.nrows_0 * entryPointParams_shape_a_0.ncols_0;
    var _S6 : u32 = _S1 / (_S5 * entryPointParams_shape_a_0.nmats_0);
    var _S7 : u32 = _S1 - _S6 * (_S4 * _S3 * _S2);
    var _S8 : u32 = _S7 / _S5;
    var _S9 : u32 = _S7 - _S8 * _S5;
    var _S10 : u32 = _S9 / _S2;
    return vec4<u32>(_S9 - _S10 * _S2, _S10, _S8, _S6);
}

fn Shape_it_0( _S11 : vec4<u32>) -> u32
{
    return _S11.x * entryPointParams_shape_a_0.row_stride_0 + _S11.y * entryPointParams_shape_a_0.col_stride_0 + _S11.z * entryPointParams_shape_a_0.mat_stride_0 + _S11.w * entryPointParams_shape_a_0.cube_stride_0;
}

fn Shape_it_1( _S12 : vec4<u32>) -> u32
{
    return _S12.x * entryPointParams_shape_b_0.row_stride_0 + _S12.y * entryPointParams_shape_b_0.col_stride_0 + _S12.z * entryPointParams_shape_b_0.mat_stride_0 + _S12.w * entryPointParams_shape_b_0.cube_stride_0;
}

fn Shape_it_wrapping_0( _S13 : vec4<u32>) -> u32
{
    var _S14 : vec4<u32> = _S13 % vec4<u32>(entryPointParams_shape_b_0.nrows_0, entryPointParams_shape_b_0.ncols_0, entryPointParams_shape_b_0.nmats_0, entryPointParams_shape_b_0.ncubes_0);
    return Shape_it_1(_S14);
}

@compute
@workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) invocation_id_0 : vec3<u32>)
{
    for(;;)
    {
        var thread_id_0 : u32 = invocation_id_0.x;
        for(;;)
        {
            if(thread_id_0 < (Shape_len_0()))
            {
            }
            else
            {
                break;
            }
            var _S15 : vec4<u32> = Shape_decompose_0(thread_id_0);
            var _S16 : u32 = Shape_it_0(_S15);
            var _S17 : u32 = Shape_it_wrapping_0(_S15);
            entryPointParams_a_0[_S16] = entryPointParams_a_0[_S16] + entryPointParams_b_0[_S17];
            thread_id_0 = thread_id_0 + u32(16776960);
        }
        break;
    }
    return;
}

fn Shape_len_1() -> u32
{
    return entryPointParams_shape_a_1.nrows_0 * entryPointParams_shape_a_1.ncols_0 * entryPointParams_shape_a_1.nmats_0 * entryPointParams_shape_a_1.ncubes_0;
}

fn Shape_decompose_1( _S18 : u32) -> vec4<u32>
{
    var _S19 : u32 = entryPointParams_shape_a_1.nrows_0;
    var _S20 : u32 = entryPointParams_shape_a_1.ncols_0;
    var _S21 : u32 = entryPointParams_shape_a_1.nmats_0;
    var _S22 : u32 = entryPointParams_shape_a_1.nrows_0 * entryPointParams_shape_a_1.ncols_0;
    var _S23 : u32 = _S18 / (_S22 * entryPointParams_shape_a_1.nmats_0);
    var _S24 : u32 = _S18 - _S23 * (_S21 * _S20 * _S19);
    var _S25 : u32 = _S24 / _S22;
    var _S26 : u32 = _S24 - _S25 * _S22;
    var _S27 : u32 = _S26 / _S19;
    return vec4<u32>(_S26 - _S27 * _S19, _S27, _S25, _S23);
}

fn Shape_it_2( _S28 : vec4<u32>) -> u32
{
    return _S28.x * entryPointParams_shape_a_1.row_stride_0 + _S28.y * entryPointParams_shape_a_1.col_stride_0 + _S28.z * entryPointParams_shape_a_1.mat_stride_0 + _S28.w * entryPointParams_shape_a_1.cube_stride_0;
}

fn Shape_it_3( _S29 : vec4<u32>) -> u32
{
    return _S29.x * entryPointParams_shape_b_1.row_stride_0 + _S29.y * entryPointParams_shape_b_1.col_stride_0 + _S29.z * entryPointParams_shape_b_1.mat_stride_0 + _S29.w * entryPointParams_shape_b_1.cube_stride_0;
}

fn Shape_it_wrapping_1( _S30 : vec4<u32>) -> u32
{
    var _S31 : vec4<u32> = _S30 % vec4<u32>(entryPointParams_shape_b_1.nrows_0, entryPointParams_shape_b_1.ncols_0, entryPointParams_shape_b_1.nmats_0, entryPointParams_shape_b_1.ncubes_0);
    return Shape_it_3(_S31);
}

@compute
@workgroup_size(256, 1, 1)
fn sub(@builtin(global_invocation_id) invocation_id_1 : vec3<u32>)
{
    for(;;)
    {
        var thread_id_1 : u32 = invocation_id_1.x;
        for(;;)
        {
            if(thread_id_1 < (Shape_len_1()))
            {
            }
            else
            {
                break;
            }
            var _S32 : vec4<u32> = Shape_decompose_1(thread_id_1);
            var _S33 : u32 = Shape_it_2(_S32);
            var _S34 : u32 = Shape_it_wrapping_1(_S32);
            entryPointParams_a_1[_S33] = entryPointParams_a_1[_S33] - entryPointParams_b_1[_S34];
            thread_id_1 = thread_id_1 + u32(16776960);
        }
        break;
    }
    return;
}

fn Shape_len_2() -> u32
{
    return entryPointParams_shape_a_2.nrows_0 * entryPointParams_shape_a_2.ncols_0 * entryPointParams_shape_a_2.nmats_0 * entryPointParams_shape_a_2.ncubes_0;
}

fn Shape_decompose_2( _S35 : u32) -> vec4<u32>
{
    var _S36 : u32 = entryPointParams_shape_a_2.nrows_0;
    var _S37 : u32 = entryPointParams_shape_a_2.ncols_0;
    var _S38 : u32 = entryPointParams_shape_a_2.nmats_0;
    var _S39 : u32 = entryPointParams_shape_a_2.nrows_0 * entryPointParams_shape_a_2.ncols_0;
    var _S40 : u32 = _S35 / (_S39 * entryPointParams_shape_a_2.nmats_0);
    var _S41 : u32 = _S35 - _S40 * (_S38 * _S37 * _S36);
    var _S42 : u32 = _S41 / _S39;
    var _S43 : u32 = _S41 - _S42 * _S39;
    var _S44 : u32 = _S43 / _S36;
    return vec4<u32>(_S43 - _S44 * _S36, _S44, _S42, _S40);
}

fn Shape_it_4( _S45 : vec4<u32>) -> u32
{
    return _S45.x * entryPointParams_shape_a_2.row_stride_0 + _S45.y * entryPointParams_shape_a_2.col_stride_0 + _S45.z * entryPointParams_shape_a_2.mat_stride_0 + _S45.w * entryPointParams_shape_a_2.cube_stride_0;
}

fn Shape_it_5( _S46 : vec4<u32>) -> u32
{
    return _S46.x * entryPointParams_shape_b_2.row_stride_0 + _S46.y * entryPointParams_shape_b_2.col_stride_0 + _S46.z * entryPointParams_shape_b_2.mat_stride_0 + _S46.w * entryPointParams_shape_b_2.cube_stride_0;
}

fn Shape_it_wrapping_2( _S47 : vec4<u32>) -> u32
{
    var _S48 : vec4<u32> = _S47 % vec4<u32>(entryPointParams_shape_b_2.nrows_0, entryPointParams_shape_b_2.ncols_0, entryPointParams_shape_b_2.nmats_0, entryPointParams_shape_b_2.ncubes_0);
    return Shape_it_5(_S48);
}

@compute
@workgroup_size(256, 1, 1)
fn mul(@builtin(global_invocation_id) invocation_id_2 : vec3<u32>)
{
    for(;;)
    {
        var thread_id_2 : u32 = invocation_id_2.x;
        for(;;)
        {
            if(thread_id_2 < (Shape_len_2()))
            {
            }
            else
            {
                break;
            }
            var _S49 : vec4<u32> = Shape_decompose_2(thread_id_2);
            var _S50 : u32 = Shape_it_4(_S49);
            var _S51 : u32 = Shape_it_wrapping_2(_S49);
            entryPointParams_a_2[_S50] = entryPointParams_a_2[_S50] * entryPointParams_b_2[_S51];
            thread_id_2 = thread_id_2 + u32(16776960);
        }
        break;
    }
    return;
}

fn Shape_len_3() -> u32
{
    return entryPointParams_shape_a_3.nrows_0 * entryPointParams_shape_a_3.ncols_0 * entryPointParams_shape_a_3.nmats_0 * entryPointParams_shape_a_3.ncubes_0;
}

fn Shape_decompose_3( _S52 : u32) -> vec4<u32>
{
    var _S53 : u32 = entryPointParams_shape_a_3.nrows_0;
    var _S54 : u32 = entryPointParams_shape_a_3.ncols_0;
    var _S55 : u32 = entryPointParams_shape_a_3.nmats_0;
    var _S56 : u32 = entryPointParams_shape_a_3.nrows_0 * entryPointParams_shape_a_3.ncols_0;
    var _S57 : u32 = _S52 / (_S56 * entryPointParams_shape_a_3.nmats_0);
    var _S58 : u32 = _S52 - _S57 * (_S55 * _S54 * _S53);
    var _S59 : u32 = _S58 / _S56;
    var _S60 : u32 = _S58 - _S59 * _S56;
    var _S61 : u32 = _S60 / _S53;
    return vec4<u32>(_S60 - _S61 * _S53, _S61, _S59, _S57);
}

fn Shape_it_6( _S62 : vec4<u32>) -> u32
{
    return _S62.x * entryPointParams_shape_a_3.row_stride_0 + _S62.y * entryPointParams_shape_a_3.col_stride_0 + _S62.z * entryPointParams_shape_a_3.mat_stride_0 + _S62.w * entryPointParams_shape_a_3.cube_stride_0;
}

fn Shape_it_7( _S63 : vec4<u32>) -> u32
{
    return _S63.x * entryPointParams_shape_b_3.row_stride_0 + _S63.y * entryPointParams_shape_b_3.col_stride_0 + _S63.z * entryPointParams_shape_b_3.mat_stride_0 + _S63.w * entryPointParams_shape_b_3.cube_stride_0;
}

fn Shape_it_wrapping_3( _S64 : vec4<u32>) -> u32
{
    var _S65 : vec4<u32> = _S64 % vec4<u32>(entryPointParams_shape_b_3.nrows_0, entryPointParams_shape_b_3.ncols_0, entryPointParams_shape_b_3.nmats_0, entryPointParams_shape_b_3.ncubes_0);
    return Shape_it_7(_S65);
}

@compute
@workgroup_size(256, 1, 1)
fn div(@builtin(global_invocation_id) invocation_id_3 : vec3<u32>)
{
    for(;;)
    {
        var thread_id_3 : u32 = invocation_id_3.x;
        for(;;)
        {
            if(thread_id_3 < (Shape_len_3()))
            {
            }
            else
            {
                break;
            }
            var _S66 : vec4<u32> = Shape_decompose_3(thread_id_3);
            var _S67 : u32 = Shape_it_6(_S66);
            var _S68 : u32 = Shape_it_wrapping_3(_S66);
            entryPointParams_a_3[_S67] = entryPointParams_a_3[_S67] / entryPointParams_b_3[_S68];
            thread_id_3 = thread_id_3 + u32(16776960);
        }
        break;
    }
    return;
}

fn Shape_len_4() -> u32
{
    return entryPointParams_shape_a_4.nrows_0 * entryPointParams_shape_a_4.ncols_0 * entryPointParams_shape_a_4.nmats_0 * entryPointParams_shape_a_4.ncubes_0;
}

fn Shape_decompose_4( _S69 : u32) -> vec4<u32>
{
    var _S70 : u32 = entryPointParams_shape_a_4.nrows_0;
    var _S71 : u32 = entryPointParams_shape_a_4.ncols_0;
    var _S72 : u32 = entryPointParams_shape_a_4.nmats_0;
    var _S73 : u32 = entryPointParams_shape_a_4.nrows_0 * entryPointParams_shape_a_4.ncols_0;
    var _S74 : u32 = _S69 / (_S73 * entryPointParams_shape_a_4.nmats_0);
    var _S75 : u32 = _S69 - _S74 * (_S72 * _S71 * _S70);
    var _S76 : u32 = _S75 / _S73;
    var _S77 : u32 = _S75 - _S76 * _S73;
    var _S78 : u32 = _S77 / _S70;
    return vec4<u32>(_S77 - _S78 * _S70, _S78, _S76, _S74);
}

fn Shape_it_8( _S79 : vec4<u32>) -> u32
{
    return _S79.x * entryPointParams_shape_a_4.row_stride_0 + _S79.y * entryPointParams_shape_a_4.col_stride_0 + _S79.z * entryPointParams_shape_a_4.mat_stride_0 + _S79.w * entryPointParams_shape_a_4.cube_stride_0;
}

fn Shape_it_9( _S80 : vec4<u32>) -> u32
{
    return _S80.x * entryPointParams_shape_b_4.row_stride_0 + _S80.y * entryPointParams_shape_b_4.col_stride_0 + _S80.z * entryPointParams_shape_b_4.mat_stride_0 + _S80.w * entryPointParams_shape_b_4.cube_stride_0;
}

fn Shape_it_wrapping_4( _S81 : vec4<u32>) -> u32
{
    var _S82 : vec4<u32> = _S81 % vec4<u32>(entryPointParams_shape_b_4.nrows_0, entryPointParams_shape_b_4.ncols_0, entryPointParams_shape_b_4.nmats_0, entryPointParams_shape_b_4.ncubes_0);
    return Shape_it_9(_S82);
}

@compute
@workgroup_size(256, 1, 1)
fn copy(@builtin(global_invocation_id) invocation_id_4 : vec3<u32>)
{
    var thread_id_4 : u32 = invocation_id_4.x;
    for(;;)
    {
        if(thread_id_4 < (Shape_len_4()))
        {
        }
        else
        {
            break;
        }
        var _S83 : vec4<u32> = Shape_decompose_4(thread_id_4);
        var _S84 : u32 = Shape_it_8(_S83);
        var _S85 : u32 = Shape_it_wrapping_4(_S83);
        entryPointParams_a_4[_S84] = entryPointParams_b_4[_S85];
        thread_id_4 = thread_id_4 + u32(16776960);
    }
    return;
}

fn Shape_len_5() -> u32
{
    return entryPointParams_shape_a_5.nrows_0 * entryPointParams_shape_a_5.ncols_0 * entryPointParams_shape_a_5.nmats_0 * entryPointParams_shape_a_5.ncubes_0;
}

fn Shape_decompose_5( _S86 : u32) -> vec4<u32>
{
    var _S87 : u32 = entryPointParams_shape_a_5.nrows_0;
    var _S88 : u32 = entryPointParams_shape_a_5.ncols_0;
    var _S89 : u32 = entryPointParams_shape_a_5.nmats_0;
    var _S90 : u32 = entryPointParams_shape_a_5.nrows_0 * entryPointParams_shape_a_5.ncols_0;
    var _S91 : u32 = _S86 / (_S90 * entryPointParams_shape_a_5.nmats_0);
    var _S92 : u32 = _S86 - _S91 * (_S89 * _S88 * _S87);
    var _S93 : u32 = _S92 / _S90;
    var _S94 : u32 = _S92 - _S93 * _S90;
    var _S95 : u32 = _S94 / _S87;
    return vec4<u32>(_S94 - _S95 * _S87, _S95, _S93, _S91);
}

fn Shape_it_10( _S96 : vec4<u32>) -> u32
{
    return _S96.x * entryPointParams_shape_a_5.row_stride_0 + _S96.y * entryPointParams_shape_a_5.col_stride_0 + _S96.z * entryPointParams_shape_a_5.mat_stride_0 + _S96.w * entryPointParams_shape_a_5.cube_stride_0;
}

fn Shape_it_11( _S97 : vec4<u32>) -> u32
{
    return _S97.x * entryPointParams_shape_b_5.row_stride_0 + _S97.y * entryPointParams_shape_b_5.col_stride_0 + _S97.z * entryPointParams_shape_b_5.mat_stride_0 + _S97.w * entryPointParams_shape_b_5.cube_stride_0;
}

fn Shape_it_wrapping_5( _S98 : vec4<u32>) -> u32
{
    var _S99 : vec4<u32> = _S98 % vec4<u32>(entryPointParams_shape_b_5.nrows_0, entryPointParams_shape_b_5.ncols_0, entryPointParams_shape_b_5.nmats_0, entryPointParams_shape_b_5.ncubes_0);
    return Shape_it_11(_S99);
}

@compute
@workgroup_size(256, 1, 1)
fn copy_with_offsets(@builtin(global_invocation_id) invocation_id_5 : vec3<u32>)
{
    var thread_id_5 : u32 = invocation_id_5.x;
    for(;;)
    {
        if(thread_id_5 < (Shape_len_5()))
        {
        }
        else
        {
            break;
        }
        var _S100 : vec4<u32> = Shape_decompose_5(thread_id_5);
        var _S101 : u32 = Shape_it_10(_S100);
        var _S102 : u32 = Shape_it_wrapping_5(_S100);
        entryPointParams_a_5[entryPointParams_offsets_0.a_0 + _S101] = entryPointParams_b_5[entryPointParams_offsets_0.b_0 + _S102];
        thread_id_5 = thread_id_5 + u32(16776960);
    }
    return;
}

