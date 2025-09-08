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
@binding(1) @group(0) var<uniform> entryPointParams_shape_m_0 : Shape_std140_0;
@binding(2) @group(0) var<uniform> entryPointParams_shape_v_0 : Shape_std140_0;
@binding(3) @group(0) var<storage, read_write> entryPointParams_out_0 : array<vec4<f32>>;

@binding(4) @group(0) var<storage, read> entryPointParams_m_0 : array<vec4<f32>>;

@binding(5) @group(0) var<storage, read> entryPointParams_v_0 : array<vec4<f32>>;

@binding(6) @group(0) var<uniform> entryPointParams_shape_out_1 : Shape_std140_0;
@binding(7) @group(0) var<uniform> entryPointParams_shape_m_1 : Shape_std140_0;
@binding(8) @group(0) var<uniform> entryPointParams_shape_v_1 : Shape_std140_0;
@binding(9) @group(0) var<storage, read_write> entryPointParams_out_1 : array<vec4<f32>>;

@binding(10) @group(0) var<storage, read> entryPointParams_m_1 : array<vec4<f32>>;

@binding(11) @group(0) var<storage, read> entryPointParams_v_1 : array<vec4<f32>>;

@binding(12) @group(0) var<uniform> entryPointParams_shape_out_2 : Shape_std140_0;
@binding(13) @group(0) var<uniform> entryPointParams_shape_m_2 : Shape_std140_0;
@binding(14) @group(0) var<uniform> entryPointParams_shape_v_2 : Shape_std140_0;
@binding(15) @group(0) var<storage, read_write> entryPointParams_out_2 : array<vec4<f32>>;

@binding(16) @group(0) var<storage, read> entryPointParams_m_2 : array<vec4<f32>>;

@binding(17) @group(0) var<storage, read> entryPointParams_v_2 : array<vec4<f32>>;

@binding(18) @group(0) var<uniform> entryPointParams_shape_out_3 : Shape_std140_0;
@binding(19) @group(0) var<uniform> entryPointParams_shape_m_3 : Shape_std140_0;
@binding(20) @group(0) var<uniform> entryPointParams_shape_v_3 : Shape_std140_0;
@binding(21) @group(0) var<storage, read_write> entryPointParams_out_3 : array<vec4<f32>>;

@binding(22) @group(0) var<storage, read> entryPointParams_m_3 : array<vec4<f32>>;

@binding(23) @group(0) var<storage, read> entryPointParams_v_3 : array<vec4<f32>>;

@binding(24) @group(0) var<uniform> entryPointParams_shape_out_4 : Shape_std140_0;
@binding(25) @group(0) var<uniform> entryPointParams_shape_m_4 : Shape_std140_0;
@binding(26) @group(0) var<uniform> entryPointParams_shape_v_4 : Shape_std140_0;
@binding(27) @group(0) var<storage, read_write> entryPointParams_out_4 : array<f32>;

@binding(28) @group(0) var<storage, read> entryPointParams_m_4 : array<f32>;

@binding(29) @group(0) var<storage, read> entryPointParams_v_4 : array<f32>;

@binding(30) @group(0) var<uniform> entryPointParams_shape_out_5 : Shape_std140_0;
@binding(31) @group(0) var<uniform> entryPointParams_shape_m_5 : Shape_std140_0;
@binding(32) @group(0) var<uniform> entryPointParams_shape_v_5 : Shape_std140_0;
@binding(33) @group(0) var<storage, read_write> entryPointParams_out_5 : array<f32>;

@binding(34) @group(0) var<storage, read> entryPointParams_m_5 : array<f32>;

@binding(35) @group(0) var<storage, read> entryPointParams_v_5 : array<f32>;

var<workgroup> sketch_0 : array<vec4<f32>, i32(32)>;

fn reduce_sum_0( index_0 : u32,  stride_0 : u32)
{
    if(index_0 < stride_0)
    {
        sketch_0[index_0] = sketch_0[index_0] + sketch_0[index_0 + stride_0];
    }
    workgroupBarrier();
    return;
}

fn Shape_it_0( _S1 : u32,  _S2 : u32,  _S3 : u32,  _S4 : u32) -> u32
{
    return _S1 * entryPointParams_shape_m_0.row_stride_0 + _S2 * entryPointParams_shape_m_0.col_stride_0 + _S3 * entryPointParams_shape_m_0.mat_stride_0 + _S4 * entryPointParams_shape_m_0.cube_stride_0;
}

fn Shape_it_wrapping_0( _S5 : u32,  _S6 : u32,  _S7 : u32,  _S8 : u32) -> u32
{
    var _S9 : u32 = entryPointParams_shape_m_0.ncols_0;
    var _S10 : u32 = entryPointParams_shape_m_0.nmats_0;
    var _S11 : u32 = entryPointParams_shape_m_0.ncubes_0;
    var _S12 : u32 = _S5 % entryPointParams_shape_m_0.nrows_0;
    var _S13 : u32 = _S6 % _S9;
    var _S14 : u32 = _S7 % _S10;
    var _S15 : u32 = _S8 % _S11;
    return Shape_it_0(_S12, _S13, _S14, _S15);
}

fn Shape_it_1( _S16 : u32,  _S17 : u32,  _S18 : u32,  _S19 : u32) -> u32
{
    return _S16 * entryPointParams_shape_v_0.row_stride_0 + _S17 * entryPointParams_shape_v_0.col_stride_0 + _S18 * entryPointParams_shape_v_0.mat_stride_0 + _S19 * entryPointParams_shape_v_0.cube_stride_0;
}

fn Shape_it_wrapping_1( _S20 : u32,  _S21 : u32,  _S22 : u32,  _S23 : u32) -> u32
{
    var _S24 : u32 = entryPointParams_shape_v_0.ncols_0;
    var _S25 : u32 = entryPointParams_shape_v_0.nmats_0;
    var _S26 : u32 = entryPointParams_shape_v_0.ncubes_0;
    var _S27 : u32 = _S20 % entryPointParams_shape_v_0.nrows_0;
    var _S28 : u32 = _S21 % _S24;
    var _S29 : u32 = _S22 % _S25;
    var _S30 : u32 = _S23 % _S26;
    return Shape_it_1(_S27, _S28, _S29, _S30);
}

fn Shape_it_2( _S31 : u32,  _S32 : u32,  _S33 : u32,  _S34 : u32) -> u32
{
    return _S31 * entryPointParams_shape_out_0.row_stride_0 + _S32 * entryPointParams_shape_out_0.col_stride_0 + _S33 * entryPointParams_shape_out_0.mat_stride_0 + _S34 * entryPointParams_shape_out_0.cube_stride_0;
}

@compute
@workgroup_size(1, 32, 1)
fn gemv_fast(@builtin(workgroup_id) workgroup_id_0 : vec3<u32>, @builtin(local_invocation_id) local_id_0 : vec3<u32>)
{
    var _S35 : u32 = local_id_0.y;
    var l_0 : u32 = u32(0);
    for(;;)
    {
        if(l_0 < (entryPointParams_shape_out_0.ncubes_0))
        {
        }
        else
        {
            break;
        }
        var _S36 : vec4<f32> = vec4<f32>(0.0f);
        var j_0 : u32 = u32(0);
        var sum_0 : vec4<f32> = _S36;
        for(;;)
        {
            if(j_0 < (entryPointParams_shape_m_0.ncols_0))
            {
            }
            else
            {
                break;
            }
            var _S37 : u32 = workgroup_id_0.z;
            var _S38 : u32 = Shape_it_wrapping_0(workgroup_id_0.x, j_0 + _S35 * u32(4), _S37, l_0);
            var _S39 : u32 = _S38 + entryPointParams_shape_m_0.col_stride_0;
            var _S40 : u32 = _S39 + entryPointParams_shape_m_0.col_stride_0;
            var _S41 : mat4x4<f32> = mat4x4<f32>(entryPointParams_m_0[_S38], entryPointParams_m_0[_S39], entryPointParams_m_0[_S40], entryPointParams_m_0[_S40 + entryPointParams_shape_m_0.col_stride_0]);
            var _S42 : u32 = Shape_it_wrapping_1(j_0 / u32(4) + _S35, workgroup_id_0.y, _S37, l_0);
            var sum_1 : vec4<f32> = sum_0 + (((_S41) * (entryPointParams_v_0[_S42])));
            j_0 = j_0 + u32(128);
            sum_0 = sum_1;
        }
        sketch_0[_S35] = sum_0;
        workgroupBarrier();
        reduce_sum_0(_S35, u32(16));
        reduce_sum_0(_S35, u32(8));
        reduce_sum_0(_S35, u32(4));
        reduce_sum_0(_S35, u32(2));
        reduce_sum_0(_S35, u32(1));
        if(_S35 == u32(0))
        {
            entryPointParams_out_0[Shape_it_2(workgroup_id_0.x, workgroup_id_0.y, workgroup_id_0.z, l_0)] = sketch_0[i32(0)];
        }
        l_0 = l_0 + u32(1);
    }
    return;
}

fn Shape_it_3( _S43 : u32,  _S44 : u32,  _S45 : u32,  _S46 : u32) -> u32
{
    return _S43 * entryPointParams_shape_m_1.row_stride_0 + _S44 * entryPointParams_shape_m_1.col_stride_0 + _S45 * entryPointParams_shape_m_1.mat_stride_0 + _S46 * entryPointParams_shape_m_1.cube_stride_0;
}

fn Shape_it_wrapping_2( _S47 : u32,  _S48 : u32,  _S49 : u32,  _S50 : u32) -> u32
{
    var _S51 : u32 = entryPointParams_shape_m_1.ncols_0;
    var _S52 : u32 = entryPointParams_shape_m_1.nmats_0;
    var _S53 : u32 = entryPointParams_shape_m_1.ncubes_0;
    var _S54 : u32 = _S47 % entryPointParams_shape_m_1.nrows_0;
    var _S55 : u32 = _S48 % _S51;
    var _S56 : u32 = _S49 % _S52;
    var _S57 : u32 = _S50 % _S53;
    return Shape_it_3(_S54, _S55, _S56, _S57);
}

fn Shape_it_4( _S58 : u32,  _S59 : u32,  _S60 : u32,  _S61 : u32) -> u32
{
    return _S58 * entryPointParams_shape_v_1.row_stride_0 + _S59 * entryPointParams_shape_v_1.col_stride_0 + _S60 * entryPointParams_shape_v_1.mat_stride_0 + _S61 * entryPointParams_shape_v_1.cube_stride_0;
}

fn Shape_it_wrapping_3( _S62 : u32,  _S63 : u32,  _S64 : u32,  _S65 : u32) -> u32
{
    var _S66 : u32 = entryPointParams_shape_v_1.ncols_0;
    var _S67 : u32 = entryPointParams_shape_v_1.nmats_0;
    var _S68 : u32 = entryPointParams_shape_v_1.ncubes_0;
    var _S69 : u32 = _S62 % entryPointParams_shape_v_1.nrows_0;
    var _S70 : u32 = _S63 % _S66;
    var _S71 : u32 = _S64 % _S67;
    var _S72 : u32 = _S65 % _S68;
    return Shape_it_4(_S69, _S70, _S71, _S72);
}

fn Shape_it_5( _S73 : u32,  _S74 : u32,  _S75 : u32,  _S76 : u32) -> u32
{
    return _S73 * entryPointParams_shape_out_1.row_stride_0 + _S74 * entryPointParams_shape_out_1.col_stride_0 + _S75 * entryPointParams_shape_out_1.mat_stride_0 + _S76 * entryPointParams_shape_out_1.cube_stride_0;
}

@compute
@workgroup_size(32, 1, 1)
fn gemv(@builtin(global_invocation_id) invocation_id_0 : vec3<u32>)
{
    var _S77 : u32 = invocation_id_0.x;
    if(_S77 < (entryPointParams_shape_m_1.nrows_0))
    {
        var l_1 : u32 = u32(0);
        for(;;)
        {
            if(l_1 < (entryPointParams_shape_out_1.ncubes_0))
            {
            }
            else
            {
                break;
            }
            var _S78 : vec4<f32> = vec4<f32>(0.0f);
            var j_1 : u32 = u32(0);
            var sum_2 : vec4<f32> = _S78;
            for(;;)
            {
                if(j_1 < (entryPointParams_shape_m_1.ncols_0))
                {
                }
                else
                {
                    break;
                }
                var _S79 : u32 = invocation_id_0.z;
                var _S80 : u32 = Shape_it_wrapping_2(_S77, j_1, _S79, l_1);
                var _S81 : u32 = _S80 + entryPointParams_shape_m_1.col_stride_0;
                var _S82 : u32 = _S81 + entryPointParams_shape_m_1.col_stride_0;
                var _S83 : mat4x4<f32> = mat4x4<f32>(entryPointParams_m_1[_S80], entryPointParams_m_1[_S81], entryPointParams_m_1[_S82], entryPointParams_m_1[_S82 + entryPointParams_shape_m_1.col_stride_0]);
                var _S84 : u32 = Shape_it_wrapping_3(j_1 / u32(4), invocation_id_0.y, _S79, l_1);
                var sum_3 : vec4<f32> = sum_2 + (((_S83) * (entryPointParams_v_1[_S84])));
                j_1 = j_1 + u32(4);
                sum_2 = sum_3;
            }
            entryPointParams_out_1[Shape_it_5(_S77, invocation_id_0.y, invocation_id_0.z, l_1)] = sum_2;
            l_1 = l_1 + u32(1);
        }
    }
    return;
}

fn Shape_it_6( _S85 : u32,  _S86 : u32,  _S87 : u32,  _S88 : u32) -> u32
{
    return _S85 * entryPointParams_shape_m_2.row_stride_0 + _S86 * entryPointParams_shape_m_2.col_stride_0 + _S87 * entryPointParams_shape_m_2.mat_stride_0 + _S88 * entryPointParams_shape_m_2.cube_stride_0;
}

fn Shape_it_wrapping_4( _S89 : u32,  _S90 : u32,  _S91 : u32,  _S92 : u32) -> u32
{
    var _S93 : u32 = entryPointParams_shape_m_2.ncols_0;
    var _S94 : u32 = entryPointParams_shape_m_2.nmats_0;
    var _S95 : u32 = entryPointParams_shape_m_2.ncubes_0;
    var _S96 : u32 = _S89 % entryPointParams_shape_m_2.nrows_0;
    var _S97 : u32 = _S90 % _S93;
    var _S98 : u32 = _S91 % _S94;
    var _S99 : u32 = _S92 % _S95;
    return Shape_it_6(_S96, _S97, _S98, _S99);
}

fn Shape_it_7( _S100 : u32,  _S101 : u32,  _S102 : u32,  _S103 : u32) -> u32
{
    return _S100 * entryPointParams_shape_v_2.row_stride_0 + _S101 * entryPointParams_shape_v_2.col_stride_0 + _S102 * entryPointParams_shape_v_2.mat_stride_0 + _S103 * entryPointParams_shape_v_2.cube_stride_0;
}

fn Shape_it_wrapping_5( _S104 : u32,  _S105 : u32,  _S106 : u32,  _S107 : u32) -> u32
{
    var _S108 : u32 = entryPointParams_shape_v_2.ncols_0;
    var _S109 : u32 = entryPointParams_shape_v_2.nmats_0;
    var _S110 : u32 = entryPointParams_shape_v_2.ncubes_0;
    var _S111 : u32 = _S104 % entryPointParams_shape_v_2.nrows_0;
    var _S112 : u32 = _S105 % _S108;
    var _S113 : u32 = _S106 % _S109;
    var _S114 : u32 = _S107 % _S110;
    return Shape_it_7(_S111, _S112, _S113, _S114);
}

fn Shape_it_8( _S115 : u32,  _S116 : u32,  _S117 : u32,  _S118 : u32) -> u32
{
    return _S115 * entryPointParams_shape_out_2.row_stride_0 + _S116 * entryPointParams_shape_out_2.col_stride_0 + _S117 * entryPointParams_shape_out_2.mat_stride_0 + _S118 * entryPointParams_shape_out_2.cube_stride_0;
}

@compute
@workgroup_size(32, 1, 1)
fn gemv_tr(@builtin(global_invocation_id) invocation_id_1 : vec3<u32>)
{
    var _S119 : u32 = invocation_id_1.x;
    if(_S119 < ((entryPointParams_shape_m_2.ncols_0 + u32(3)) / u32(4)))
    {
        var l_2 : u32 = u32(0);
        for(;;)
        {
            if(l_2 < (entryPointParams_shape_out_2.ncubes_0))
            {
            }
            else
            {
                break;
            }
            var _S120 : vec4<f32> = vec4<f32>(0.0f);
            var j_2 : u32 = u32(0);
            var sum_4 : vec4<f32> = _S120;
            for(;;)
            {
                if(j_2 < (entryPointParams_shape_m_2.nrows_0))
                {
                }
                else
                {
                    break;
                }
                var _S121 : u32 = invocation_id_1.z;
                var _S122 : u32 = Shape_it_wrapping_4(j_2, _S119 * u32(4), _S121, l_2);
                var _S123 : u32 = _S122 + entryPointParams_shape_m_2.col_stride_0;
                var _S124 : u32 = _S123 + entryPointParams_shape_m_2.col_stride_0;
                var _S125 : mat4x4<f32> = mat4x4<f32>(entryPointParams_m_2[_S122], entryPointParams_m_2[_S123], entryPointParams_m_2[_S124], entryPointParams_m_2[_S124 + entryPointParams_shape_m_2.col_stride_0]);
                var _S126 : u32 = Shape_it_wrapping_5(j_2, invocation_id_1.y, _S121, l_2);
                var sum_5 : vec4<f32> = sum_4 + (((entryPointParams_v_2[_S126]) * (_S125)));
                j_2 = j_2 + u32(1);
                sum_4 = sum_5;
            }
            entryPointParams_out_2[Shape_it_8(_S119, invocation_id_1.y, invocation_id_1.z, l_2)] = sum_4;
            l_2 = l_2 + u32(1);
        }
    }
    return;
}

fn Shape_it_9( _S127 : u32,  _S128 : u32,  _S129 : u32,  _S130 : u32) -> u32
{
    return _S127 * entryPointParams_shape_m_3.row_stride_0 + _S128 * entryPointParams_shape_m_3.col_stride_0 + _S129 * entryPointParams_shape_m_3.mat_stride_0 + _S130 * entryPointParams_shape_m_3.cube_stride_0;
}

fn Shape_it_wrapping_6( _S131 : u32,  _S132 : u32,  _S133 : u32,  _S134 : u32) -> u32
{
    var _S135 : u32 = entryPointParams_shape_m_3.ncols_0;
    var _S136 : u32 = entryPointParams_shape_m_3.nmats_0;
    var _S137 : u32 = entryPointParams_shape_m_3.ncubes_0;
    var _S138 : u32 = _S131 % entryPointParams_shape_m_3.nrows_0;
    var _S139 : u32 = _S132 % _S135;
    var _S140 : u32 = _S133 % _S136;
    var _S141 : u32 = _S134 % _S137;
    return Shape_it_9(_S138, _S139, _S140, _S141);
}

fn Shape_it_10( _S142 : u32,  _S143 : u32,  _S144 : u32,  _S145 : u32) -> u32
{
    return _S142 * entryPointParams_shape_v_3.row_stride_0 + _S143 * entryPointParams_shape_v_3.col_stride_0 + _S144 * entryPointParams_shape_v_3.mat_stride_0 + _S145 * entryPointParams_shape_v_3.cube_stride_0;
}

fn Shape_it_wrapping_7( _S146 : u32,  _S147 : u32,  _S148 : u32,  _S149 : u32) -> u32
{
    var _S150 : u32 = entryPointParams_shape_v_3.ncols_0;
    var _S151 : u32 = entryPointParams_shape_v_3.nmats_0;
    var _S152 : u32 = entryPointParams_shape_v_3.ncubes_0;
    var _S153 : u32 = _S146 % entryPointParams_shape_v_3.nrows_0;
    var _S154 : u32 = _S147 % _S150;
    var _S155 : u32 = _S148 % _S151;
    var _S156 : u32 = _S149 % _S152;
    return Shape_it_10(_S153, _S154, _S155, _S156);
}

fn Shape_it_11( _S157 : u32,  _S158 : u32,  _S159 : u32,  _S160 : u32) -> u32
{
    return _S157 * entryPointParams_shape_out_3.row_stride_0 + _S158 * entryPointParams_shape_out_3.col_stride_0 + _S159 * entryPointParams_shape_out_3.mat_stride_0 + _S160 * entryPointParams_shape_out_3.cube_stride_0;
}

@compute
@workgroup_size(1, 32, 1)
fn gemv_tr_fast(@builtin(workgroup_id) workgroup_id_1 : vec3<u32>, @builtin(local_invocation_id) local_id_1 : vec3<u32>)
{
    var _S161 : u32 = local_id_1.y;
    var l_3 : u32 = u32(0);
    for(;;)
    {
        if(l_3 < (entryPointParams_shape_out_3.ncubes_0))
        {
        }
        else
        {
            break;
        }
        var _S162 : vec4<f32> = vec4<f32>(0.0f);
        var j_3 : u32 = u32(0);
        var sum_6 : vec4<f32> = _S162;
        for(;;)
        {
            if(j_3 < (entryPointParams_shape_m_3.nrows_0))
            {
            }
            else
            {
                break;
            }
            var _S163 : u32 = j_3 + _S161;
            var _S164 : u32 = workgroup_id_1.z;
            var _S165 : u32 = Shape_it_wrapping_6(_S163, workgroup_id_1.x * u32(4), _S164, l_3);
            var _S166 : u32 = _S165 + entryPointParams_shape_m_3.col_stride_0;
            var _S167 : u32 = _S166 + entryPointParams_shape_m_3.col_stride_0;
            var _S168 : mat4x4<f32> = mat4x4<f32>(entryPointParams_m_3[_S165], entryPointParams_m_3[_S166], entryPointParams_m_3[_S167], entryPointParams_m_3[_S167 + entryPointParams_shape_m_3.col_stride_0]);
            var _S169 : u32 = Shape_it_wrapping_7(_S163, workgroup_id_1.y, _S164, l_3);
            var sum_7 : vec4<f32> = sum_6 + (((entryPointParams_v_3[_S169]) * (_S168)));
            j_3 = j_3 + u32(32);
            sum_6 = sum_7;
        }
        sketch_0[_S161] = sum_6;
        workgroupBarrier();
        reduce_sum_0(_S161, u32(16));
        reduce_sum_0(_S161, u32(8));
        reduce_sum_0(_S161, u32(4));
        reduce_sum_0(_S161, u32(2));
        reduce_sum_0(_S161, u32(1));
        if(_S161 == u32(0))
        {
            entryPointParams_out_3[Shape_it_11(workgroup_id_1.x, workgroup_id_1.y, workgroup_id_1.z, l_3)] = sketch_0[i32(0)];
        }
        l_3 = l_3 + u32(1);
    }
    return;
}

fn Shape_it_12( _S170 : u32,  _S171 : u32,  _S172 : u32,  _S173 : u32) -> u32
{
    return _S170 * entryPointParams_shape_m_4.row_stride_0 + _S171 * entryPointParams_shape_m_4.col_stride_0 + _S172 * entryPointParams_shape_m_4.mat_stride_0 + _S173 * entryPointParams_shape_m_4.cube_stride_0;
}

fn Shape_it_wrapping_8( _S174 : u32,  _S175 : u32,  _S176 : u32,  _S177 : u32) -> u32
{
    var _S178 : u32 = entryPointParams_shape_m_4.ncols_0;
    var _S179 : u32 = entryPointParams_shape_m_4.nmats_0;
    var _S180 : u32 = entryPointParams_shape_m_4.ncubes_0;
    var _S181 : u32 = _S174 % entryPointParams_shape_m_4.nrows_0;
    var _S182 : u32 = _S175 % _S178;
    var _S183 : u32 = _S176 % _S179;
    var _S184 : u32 = _S177 % _S180;
    return Shape_it_12(_S181, _S182, _S183, _S184);
}

fn Shape_it_13( _S185 : u32,  _S186 : u32,  _S187 : u32,  _S188 : u32) -> u32
{
    return _S185 * entryPointParams_shape_v_4.row_stride_0 + _S186 * entryPointParams_shape_v_4.col_stride_0 + _S187 * entryPointParams_shape_v_4.mat_stride_0 + _S188 * entryPointParams_shape_v_4.cube_stride_0;
}

fn Shape_it_wrapping_9( _S189 : u32,  _S190 : u32,  _S191 : u32,  _S192 : u32) -> u32
{
    var _S193 : u32 = entryPointParams_shape_v_4.ncols_0;
    var _S194 : u32 = entryPointParams_shape_v_4.nmats_0;
    var _S195 : u32 = entryPointParams_shape_v_4.ncubes_0;
    var _S196 : u32 = _S189 % entryPointParams_shape_v_4.nrows_0;
    var _S197 : u32 = _S190 % _S193;
    var _S198 : u32 = _S191 % _S194;
    var _S199 : u32 = _S192 % _S195;
    return Shape_it_13(_S196, _S197, _S198, _S199);
}

fn Shape_it_14( _S200 : u32,  _S201 : u32,  _S202 : u32,  _S203 : u32) -> u32
{
    return _S200 * entryPointParams_shape_out_4.row_stride_0 + _S201 * entryPointParams_shape_out_4.col_stride_0 + _S202 * entryPointParams_shape_out_4.mat_stride_0 + _S203 * entryPointParams_shape_out_4.cube_stride_0;
}

@compute
@workgroup_size(32, 1, 1)
fn gemv_naive(@builtin(global_invocation_id) invocation_id_2 : vec3<u32>)
{
    var _S204 : u32 = invocation_id_2.x;
    if(_S204 < (entryPointParams_shape_m_4.nrows_0))
    {
        var l_4 : u32 = u32(0);
        for(;;)
        {
            if(l_4 < (entryPointParams_shape_out_4.ncubes_0))
            {
            }
            else
            {
                break;
            }
            var j_4 : u32 = u32(0);
            var sum_8 : f32 = 0.0f;
            for(;;)
            {
                if(j_4 < (entryPointParams_shape_m_4.ncols_0))
                {
                }
                else
                {
                    break;
                }
                var _S205 : u32 = invocation_id_2.z;
                var _S206 : u32 = Shape_it_wrapping_8(_S204, j_4, _S205, l_4);
                var _S207 : u32 = Shape_it_wrapping_9(j_4, invocation_id_2.y, _S205, l_4);
                var sum_9 : f32 = sum_8 + entryPointParams_m_4[_S206] * entryPointParams_v_4[_S207];
                j_4 = j_4 + u32(1);
                sum_8 = sum_9;
            }
            entryPointParams_out_4[Shape_it_14(_S204, invocation_id_2.y, invocation_id_2.z, l_4)] = sum_8;
            l_4 = l_4 + u32(1);
        }
    }
    return;
}

fn Shape_it_15( _S208 : u32,  _S209 : u32,  _S210 : u32,  _S211 : u32) -> u32
{
    return _S208 * entryPointParams_shape_m_5.row_stride_0 + _S209 * entryPointParams_shape_m_5.col_stride_0 + _S210 * entryPointParams_shape_m_5.mat_stride_0 + _S211 * entryPointParams_shape_m_5.cube_stride_0;
}

fn Shape_it_wrapping_10( _S212 : u32,  _S213 : u32,  _S214 : u32,  _S215 : u32) -> u32
{
    var _S216 : u32 = entryPointParams_shape_m_5.ncols_0;
    var _S217 : u32 = entryPointParams_shape_m_5.nmats_0;
    var _S218 : u32 = entryPointParams_shape_m_5.ncubes_0;
    var _S219 : u32 = _S212 % entryPointParams_shape_m_5.nrows_0;
    var _S220 : u32 = _S213 % _S216;
    var _S221 : u32 = _S214 % _S217;
    var _S222 : u32 = _S215 % _S218;
    return Shape_it_15(_S219, _S220, _S221, _S222);
}

fn Shape_it_16( _S223 : u32,  _S224 : u32,  _S225 : u32,  _S226 : u32) -> u32
{
    return _S223 * entryPointParams_shape_v_5.row_stride_0 + _S224 * entryPointParams_shape_v_5.col_stride_0 + _S225 * entryPointParams_shape_v_5.mat_stride_0 + _S226 * entryPointParams_shape_v_5.cube_stride_0;
}

fn Shape_it_wrapping_11( _S227 : u32,  _S228 : u32,  _S229 : u32,  _S230 : u32) -> u32
{
    var _S231 : u32 = entryPointParams_shape_v_5.ncols_0;
    var _S232 : u32 = entryPointParams_shape_v_5.nmats_0;
    var _S233 : u32 = entryPointParams_shape_v_5.ncubes_0;
    var _S234 : u32 = _S227 % entryPointParams_shape_v_5.nrows_0;
    var _S235 : u32 = _S228 % _S231;
    var _S236 : u32 = _S229 % _S232;
    var _S237 : u32 = _S230 % _S233;
    return Shape_it_16(_S234, _S235, _S236, _S237);
}

fn Shape_it_17( _S238 : u32,  _S239 : u32,  _S240 : u32,  _S241 : u32) -> u32
{
    return _S238 * entryPointParams_shape_out_5.row_stride_0 + _S239 * entryPointParams_shape_out_5.col_stride_0 + _S240 * entryPointParams_shape_out_5.mat_stride_0 + _S241 * entryPointParams_shape_out_5.cube_stride_0;
}

@compute
@workgroup_size(32, 1, 1)
fn gemv_tr_naive(@builtin(global_invocation_id) invocation_id_3 : vec3<u32>)
{
    var _S242 : u32 = invocation_id_3.x;
    if(_S242 < (entryPointParams_shape_out_5.nrows_0))
    {
        var l_5 : u32 = u32(0);
        for(;;)
        {
            if(l_5 < (entryPointParams_shape_out_5.ncubes_0))
            {
            }
            else
            {
                break;
            }
            var j_5 : u32 = u32(0);
            var sum_10 : f32 = 0.0f;
            for(;;)
            {
                if(j_5 < (entryPointParams_shape_m_5.nrows_0))
                {
                }
                else
                {
                    break;
                }
                var _S243 : u32 = invocation_id_3.z;
                var _S244 : u32 = Shape_it_wrapping_10(j_5, _S242, _S243, l_5);
                var _S245 : u32 = Shape_it_wrapping_11(j_5, invocation_id_3.y, _S243, l_5);
                var sum_11 : f32 = sum_10 + entryPointParams_m_5[_S244] * entryPointParams_v_5[_S245];
                j_5 = j_5 + u32(1);
                sum_10 = sum_11;
            }
            entryPointParams_out_5[Shape_it_17(_S242, invocation_id_3.y, invocation_id_3.z, l_5)] = sum_10;
            l_5 = l_5 + u32(1);
        }
    }
    return;
}

