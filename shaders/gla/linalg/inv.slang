module inv;

// These inverse functions were copied from https://github.com/gfx-rs/wgpu/tree/trunk/naga/src/back/wgsl/polyfill/inverse (MIT/Apache 2 license)

/// The inverse of a 2x2 matrix.
///
/// Returns an invalid result if the matrix is not invertible.
public func inv(m: float2x2) -> float2x2 {
    var adj: float2x2;
    adj[0][0] = m[1][1];
    adj[0][1] = -m[0][1];
    adj[1][0] = -m[1][0];
    adj[1][1] = m[0][0];

    let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    return adj * (1 / det);
}


/// The inverse of a 3x3 matrix.
///
/// Returns an invalid result if the matrix is not invertible.
public func inv(m: float3x3) -> float3x3 {
    var adj: float3x3;

    adj[0][0] =   (m[1][1] * m[2][2] - m[2][1] * m[1][2]);
    adj[1][0] = - (m[1][0] * m[2][2] - m[2][0] * m[1][2]);
    adj[2][0] =   (m[1][0] * m[2][1] - m[2][0] * m[1][1]);
    adj[0][1] = - (m[0][1] * m[2][2] - m[2][1] * m[0][2]);
    adj[1][1] =   (m[0][0] * m[2][2] - m[2][0] * m[0][2]);
    adj[2][1] = - (m[0][0] * m[2][1] - m[2][0] * m[0][1]);
    adj[0][2] =   (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
    adj[1][2] = - (m[0][0] * m[1][2] - m[1][0] * m[0][2]);
    adj[2][2] =   (m[0][0] * m[1][1] - m[1][0] * m[0][1]);

    let det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    		- m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    		+ m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));

    return adj * (1 / det);
}


/// The inverse of a 4x4 matrix.
///
/// Returns an invalid result if the matrix is not invertible.
public func inv(m: float4x4) -> float4x4 {
   let sub_factor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
   let sub_factor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
   let sub_factor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
   let sub_factor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
   let sub_factor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
   let sub_factor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
   let sub_factor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
   let sub_factor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
   let sub_factor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
   let sub_factor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
   let sub_factor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
   let sub_factor11 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
   let sub_factor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
   let sub_factor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
   let sub_factor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
   let sub_factor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
   let sub_factor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
   let sub_factor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
   let sub_factor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

   var adj: float4x4;
   adj[0][0] =   (m[1][1] * sub_factor00 - m[1][2] * sub_factor01 + m[1][3] * sub_factor02);
   adj[1][0] = - (m[1][0] * sub_factor00 - m[1][2] * sub_factor03 + m[1][3] * sub_factor04);
   adj[2][0] =   (m[1][0] * sub_factor01 - m[1][1] * sub_factor03 + m[1][3] * sub_factor05);
   adj[3][0] = - (m[1][0] * sub_factor02 - m[1][1] * sub_factor04 + m[1][2] * sub_factor05);
   adj[0][1] = - (m[0][1] * sub_factor00 - m[0][2] * sub_factor01 + m[0][3] * sub_factor02);
   adj[1][1] =   (m[0][0] * sub_factor00 - m[0][2] * sub_factor03 + m[0][3] * sub_factor04);
   adj[2][1] = - (m[0][0] * sub_factor01 - m[0][1] * sub_factor03 + m[0][3] * sub_factor05);
   adj[3][1] =   (m[0][0] * sub_factor02 - m[0][1] * sub_factor04 + m[0][2] * sub_factor05);
   adj[0][2] =   (m[0][1] * sub_factor06 - m[0][2] * sub_factor07 + m[0][3] * sub_factor08);
   adj[1][2] = - (m[0][0] * sub_factor06 - m[0][2] * sub_factor09 + m[0][3] * sub_factor10);
   adj[2][2] =   (m[0][0] * sub_factor11 - m[0][1] * sub_factor09 + m[0][3] * sub_factor12);
   adj[3][2] = - (m[0][0] * sub_factor08 - m[0][1] * sub_factor10 + m[0][2] * sub_factor12);
   adj[0][3] = - (m[0][1] * sub_factor13 - m[0][2] * sub_factor14 + m[0][3] * sub_factor15);
   adj[1][3] =   (m[0][0] * sub_factor13 - m[0][2] * sub_factor16 + m[0][3] * sub_factor17);
   adj[2][3] = - (m[0][0] * sub_factor14 - m[0][1] * sub_factor16 + m[0][3] * sub_factor18);
   adj[3][3] =   (m[0][0] * sub_factor15 - m[0][1] * sub_factor17 + m[0][2] * sub_factor18);

   let det = (m[0][0] * adj[0][0] + m[0][1] * adj[1][0] + m[0][2] * adj[2][0] + m[0][3] * adj[3][0]);

   return adj * (1 / det);
}