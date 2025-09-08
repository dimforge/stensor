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

@binding(0) @group(0) var<uniform> entryPointParams_shape_0 : Shape_std140_0;
@binding(1) @group(0) var<storage, read> entryPointParams_input_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> entryPointParams_output_0 : array<f32>;

@binding(3) @group(0) var<uniform> entryPointParams_shape_1 : Shape_std140_0;
@binding(4) @group(0) var<storage, read> entryPointParams_input_1 : array<f32>;

@binding(5) @group(0) var<storage, read_write> entryPointParams_output_1 : array<f32>;

@binding(6) @group(0) var<uniform> entryPointParams_shape_2 : Shape_std140_0;
@binding(7) @group(0) var<storage, read> entryPointParams_input_2 : array<f32>;

@binding(8) @group(0) var<storage, read_write> entryPointParams_output_2 : array<f32>;

@binding(9) @group(0) var<uniform> entryPointParams_shape_3 : Shape_std140_0;
@binding(10) @group(0) var<storage, read> entryPointParams_input_3 : array<f32>;

@binding(11) @group(0) var<storage, read_write> entryPointParams_output_3 : array<f32>;

@binding(12) @group(0) var<uniform> entryPointParams_shape_4 : Shape_std140_0;
@binding(13) @group(0) var<storage, read> entryPointParams_input_4 : array<f32>;

@binding(14) @group(0) var<storage, read_write> entryPointParams_output_4 : array<f32>;

var<workgroup> workspace_0 : array<f32, i32(128)>;

fn Sum_reduce_workspace_0( a_0 : f32,  b_0 : f32) -> f32
{
    return a_0 + b_0;
}

fn reduce_0( thread_id_0 : u32,  stride_0 : u32)
{
    if(thread_id_0 < stride_0)
    {
        workspace_0[thread_id_0] = Sum_reduce_workspace_0(workspace_0[thread_id_0], workspace_0[thread_id_0 + stride_0]);
    }
    workgroupBarrier();
    return;
}

fn Prod_reduce_workspace_0( a_1 : f32,  b_1 : f32) -> f32
{
    return a_1 * b_1;
}

fn reduce_1( thread_id_1 : u32,  stride_1 : u32)
{
    if(thread_id_1 < stride_1)
    {
        workspace_0[thread_id_1] = Prod_reduce_workspace_0(workspace_0[thread_id_1], workspace_0[thread_id_1 + stride_1]);
    }
    workgroupBarrier();
    return;
}

fn Min_reduce_workspace_0( a_2 : f32,  b_2 : f32) -> f32
{
    return min(a_2, b_2);
}

fn reduce_2( thread_id_2 : u32,  stride_2 : u32)
{
    if(thread_id_2 < stride_2)
    {
        workspace_0[thread_id_2] = Min_reduce_workspace_0(workspace_0[thread_id_2], workspace_0[thread_id_2 + stride_2]);
    }
    workgroupBarrier();
    return;
}

fn Max_reduce_workspace_0( a_3 : f32,  b_3 : f32) -> f32
{
    return max(a_3, b_3);
}

fn reduce_3( thread_id_3 : u32,  stride_3 : u32)
{
    if(thread_id_3 < stride_3)
    {
        workspace_0[thread_id_3] = Max_reduce_workspace_0(workspace_0[thread_id_3], workspace_0[thread_id_3 + stride_3]);
    }
    workgroupBarrier();
    return;
}

fn SqNorm_reduce_workspace_0( a_4 : f32,  b_4 : f32) -> f32
{
    return a_4 + b_4;
}

fn reduce_4( thread_id_4 : u32,  stride_4 : u32)
{
    if(thread_id_4 < stride_4)
    {
        workspace_0[thread_id_4] = SqNorm_reduce_workspace_0(workspace_0[thread_id_4], workspace_0[thread_id_4 + stride_4]);
    }
    workgroupBarrier();
    return;
}

fn Sum_init_0() -> f32
{
    return 0.0f;
}

fn Shape_iv_0( _S1 : u32) -> u32
{
    return _S1 * entryPointParams_shape_0.row_stride_0;
}

fn Sum_reduce_buffer_0( a_5 : f32,  b_5 : f32) -> f32
{
    return a_5 + b_5;
}

fn run_reduction_0( _S2 : u32) -> f32
{
    workspace_0[_S2] = Sum_init_0();
    var i_0 : u32 = _S2;
    for(;;)
    {
        if(i_0 < (entryPointParams_shape_0.nrows_0))
        {
        }
        else
        {
            break;
        }
        workspace_0[_S2] = Sum_reduce_buffer_0(workspace_0[_S2], entryPointParams_input_0[Shape_iv_0(i_0)]);
        i_0 = i_0 + u32(128);
    }
    workgroupBarrier();
    reduce_0(_S2, u32(64));
    reduce_0(_S2, u32(32));
    reduce_0(_S2, u32(16));
    reduce_0(_S2, u32(8));
    reduce_0(_S2, u32(4));
    reduce_0(_S2, u32(2));
    reduce_0(_S2, u32(1));
    return workspace_0[i32(0)];
}

fn main_0( _S3 : vec3<u32>)
{
    var _S4 : u32 = _S3.x;
    var _S5 : f32 = run_reduction_0(_S4);
    if(_S4 == u32(0))
    {
        entryPointParams_output_0[i32(0)] = _S5;
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn reduce_sum(@builtin(global_invocation_id) invocation_id_0 : vec3<u32>)
{
    main_0(invocation_id_0);
    return;
}

fn Prod_init_0() -> f32
{
    return 1.0f;
}

fn Shape_iv_1( _S6 : u32) -> u32
{
    return _S6 * entryPointParams_shape_1.row_stride_0;
}

fn Prod_reduce_buffer_0( a_6 : f32,  b_6 : f32) -> f32
{
    return a_6 * b_6;
}

fn run_reduction_1( _S7 : u32) -> f32
{
    workspace_0[_S7] = Prod_init_0();
    var i_1 : u32 = _S7;
    for(;;)
    {
        if(i_1 < (entryPointParams_shape_1.nrows_0))
        {
        }
        else
        {
            break;
        }
        workspace_0[_S7] = Prod_reduce_buffer_0(workspace_0[_S7], entryPointParams_input_1[Shape_iv_1(i_1)]);
        i_1 = i_1 + u32(128);
    }
    workgroupBarrier();
    reduce_1(_S7, u32(64));
    reduce_1(_S7, u32(32));
    reduce_1(_S7, u32(16));
    reduce_1(_S7, u32(8));
    reduce_1(_S7, u32(4));
    reduce_1(_S7, u32(2));
    reduce_1(_S7, u32(1));
    return workspace_0[i32(0)];
}

fn main_1( _S8 : vec3<u32>)
{
    var _S9 : u32 = _S8.x;
    var _S10 : f32 = run_reduction_1(_S9);
    if(_S9 == u32(0))
    {
        entryPointParams_output_1[i32(0)] = _S10;
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn reduce_product(@builtin(global_invocation_id) invocation_id_1 : vec3<u32>)
{
    main_1(invocation_id_1);
    return;
}

fn Min_init_0() -> f32
{
    return 3.4028234663852886e+38f;
}

fn Shape_iv_2( _S11 : u32) -> u32
{
    return _S11 * entryPointParams_shape_2.row_stride_0;
}

fn Min_reduce_buffer_0( a_7 : f32,  b_7 : f32) -> f32
{
    return min(a_7, b_7);
}

fn run_reduction_2( _S12 : u32) -> f32
{
    workspace_0[_S12] = Min_init_0();
    var i_2 : u32 = _S12;
    for(;;)
    {
        if(i_2 < (entryPointParams_shape_2.nrows_0))
        {
        }
        else
        {
            break;
        }
        workspace_0[_S12] = Min_reduce_buffer_0(workspace_0[_S12], entryPointParams_input_2[Shape_iv_2(i_2)]);
        i_2 = i_2 + u32(128);
    }
    workgroupBarrier();
    reduce_2(_S12, u32(64));
    reduce_2(_S12, u32(32));
    reduce_2(_S12, u32(16));
    reduce_2(_S12, u32(8));
    reduce_2(_S12, u32(4));
    reduce_2(_S12, u32(2));
    reduce_2(_S12, u32(1));
    return workspace_0[i32(0)];
}

fn main_2( _S13 : vec3<u32>)
{
    var _S14 : u32 = _S13.x;
    var _S15 : f32 = run_reduction_2(_S14);
    if(_S14 == u32(0))
    {
        entryPointParams_output_2[i32(0)] = _S15;
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn reduce_min(@builtin(global_invocation_id) invocation_id_2 : vec3<u32>)
{
    main_2(invocation_id_2);
    return;
}

fn Max_init_0() -> f32
{
    return -3.4028234663852886e+38f;
}

fn Shape_iv_3( _S16 : u32) -> u32
{
    return _S16 * entryPointParams_shape_3.row_stride_0;
}

fn Max_reduce_buffer_0( a_8 : f32,  b_8 : f32) -> f32
{
    return max(a_8, b_8);
}

fn run_reduction_3( _S17 : u32) -> f32
{
    workspace_0[_S17] = Max_init_0();
    var i_3 : u32 = _S17;
    for(;;)
    {
        if(i_3 < (entryPointParams_shape_3.nrows_0))
        {
        }
        else
        {
            break;
        }
        workspace_0[_S17] = Max_reduce_buffer_0(workspace_0[_S17], entryPointParams_input_3[Shape_iv_3(i_3)]);
        i_3 = i_3 + u32(128);
    }
    workgroupBarrier();
    reduce_3(_S17, u32(64));
    reduce_3(_S17, u32(32));
    reduce_3(_S17, u32(16));
    reduce_3(_S17, u32(8));
    reduce_3(_S17, u32(4));
    reduce_3(_S17, u32(2));
    reduce_3(_S17, u32(1));
    return workspace_0[i32(0)];
}

fn main_3( _S18 : vec3<u32>)
{
    var _S19 : u32 = _S18.x;
    var _S20 : f32 = run_reduction_3(_S19);
    if(_S19 == u32(0))
    {
        entryPointParams_output_3[i32(0)] = _S20;
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn reduce_max(@builtin(global_invocation_id) invocation_id_3 : vec3<u32>)
{
    main_3(invocation_id_3);
    return;
}

fn SqNorm_init_0() -> f32
{
    return 0.0f;
}

fn Shape_iv_4( _S21 : u32) -> u32
{
    return _S21 * entryPointParams_shape_4.row_stride_0;
}

fn SqNorm_reduce_buffer_0( a_9 : f32,  b_9 : f32) -> f32
{
    return a_9 + b_9 * b_9;
}

fn run_reduction_4( _S22 : u32) -> f32
{
    workspace_0[_S22] = SqNorm_init_0();
    var i_4 : u32 = _S22;
    for(;;)
    {
        if(i_4 < (entryPointParams_shape_4.nrows_0))
        {
        }
        else
        {
            break;
        }
        workspace_0[_S22] = SqNorm_reduce_buffer_0(workspace_0[_S22], entryPointParams_input_4[Shape_iv_4(i_4)]);
        i_4 = i_4 + u32(128);
    }
    workgroupBarrier();
    reduce_4(_S22, u32(64));
    reduce_4(_S22, u32(32));
    reduce_4(_S22, u32(16));
    reduce_4(_S22, u32(8));
    reduce_4(_S22, u32(4));
    reduce_4(_S22, u32(2));
    reduce_4(_S22, u32(1));
    return workspace_0[i32(0)];
}

fn main_4( _S23 : vec3<u32>)
{
    var _S24 : u32 = _S23.x;
    var _S25 : f32 = run_reduction_4(_S24);
    if(_S24 == u32(0))
    {
        entryPointParams_output_4[i32(0)] = _S25;
    }
    return;
}

@compute
@workgroup_size(128, 1, 1)
fn reduce_sqnorm(@builtin(global_invocation_id) invocation_id_4 : vec3<u32>)
{
    main_4(invocation_id_4);
    return;
}

