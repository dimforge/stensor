module mat;

/*
 * Trace of a matrix.
 */

/// The trace of a 2x2 matrix.
public func trace(m: float2x2) -> float {
    return m[0][0] + m[1][1];
}

/// The trace of a 3x3 matrix.
public func trace(m: float3x3) -> float {
    return m[0][0] + m[1][1] + m[2][2];
}

/// The trace of a 4x4 matrix.
public func trace(m: float4x4) -> float {
    return m[0][0] + m[1][1] + m[2][2] + m[3][3];
}

/*
 * Diagonal extraction and diagonal matrix init.
 */

/// Initializes a diagonal 2x2 matrix.
public func diag(d: float2) -> float2x2 {
    return float2x2(float2(d.x, 0.0), float2(0.0, d.y));
}

/// Initializes a diagonal 3x3 matrix.
public func diag(d: float3) -> float3x3 {
    return float3x3(float3(d.x, 0.0, 0.0), float3(0.0, d.y, 0.0), float3(0.0, 0.0, d.z));
}

/// Initializes a diagonal 4x4 matrix.
public func diag(d: float4) -> float4x4 {
    return float4x4(float4(d.x, 0.0, 0.0, 0.0), float4(0.0, d.y, 0.0, 0.0), float4(0.0, 0.0, d.z, 0.0), float4(0.0, 0.0, 0.0, d.w));
}


/// Return the diagonal of a 2x2 matrix.
public func diag(m: float2x2) -> float2 {
    return float2(m[0][0], m[1][1]);
}

/// Return the diagonal of a 3x3 matrix.
public func diag(m: float3x3) -> float3 {
    return float3(m[0][0], m[1][1], m[2][2]);
}

/// Return the diagonal of a 4x4 matrix.
public func diag(m: float4x4) -> float4 {
    return float4(m[0][0], m[1][1], m[2][2], m[3][3]);
}