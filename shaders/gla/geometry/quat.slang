module quat;

/// A (generally unit-sized) quaternion representing a rotation.
public struct Quat {
    /// The quaternion’s coordinates (i, j, k, w).
    public float4 coords;

    /// The quaternion representing an identity rotation.
    public __init() {
        this.coords = float4(0.0, 0.0, 0.0, 1.0);
    }

    __init(float4 coords) {
        this.coords = coords;
    }

    /// Convert an axis-angle (represented as the axis multiplied by the angle) to
    /// a quaternion.
    public static func from_scaled_axis(axisangle: float3) -> Quat {
        let angle = length(axisangle);
        let is_zero = float(angle == 0.0);

        if (angle == 0.0) {
            return Quat();
        } else {
            let hs = sin(angle / 2.0);
            let hc = cos(angle / 2.0);
            let axis = axisangle / angle;
            return Quat(float4(axis * hs, hc));
        }
    }

    // Converts this quaternion to a rotation matrix.
    public func to_matrix() -> float3x3 {
        let i = this.coords.x;
        let j = this.coords.y;
        let k = this.coords.z;
        let w = this.coords.w;

        let ww = w * w;
        let ii = i * i;
        let jj = j * j;
        let kk = k * k;
        let ij = i * j * 2.0;
        let wk = w * k * 2.0;
        let wj = w * j * 2.0;
        let ik = i * k * 2.0;
        let jk = j * k * 2.0;
        let wi = w * i * 2.0;

        return float3x3(
            float3(ww + ii - jj - kk, wk + ij, ik - wj),
            float3(ij - wk, ww - ii + jj - kk, wi + jk),
            float3(wj + ik, jk - wi, ww - ii - jj + kk),
        );
    }

    /// Normalizes this quaternion again using a first-order Taylor approximation.
    /// This is useful when repeated computations might cause a drift in the norm
    /// because of float inaccuracies.
    public func renormalize_fast() -> Quat {
        let sq_norm = dot(this.coords, this.coords);
        return Quat(this.coords * (0.5 * (3.0 - sq_norm)));
    }

    /// The inverse (conjugate) of a unit quaternion.
    public func inv() -> Quat {
        return Quat(float4(-this.coords.xyz, this.coords.w));
    }

    /// Multiplies a quaternion’s inverse by a vector (inverse-rotates the vector).
    public func inv_mul_vec(v: float3) -> float3 {
        let t = cross(this.coords.xyz, v) * 2.0;
        let c = cross(this.coords.xyz, t);
        return t * -this.coords.w + c + v;
    }
}

/// Multiplies two quaternions (combines their rotations).
public Quat operator*(Quat lhs, Quat rhs) {
    let scalar = lhs.coords.w * rhs.coords.w - dot(lhs.coords.xyz, rhs.coords.xyz);
    let v = cross(lhs.coords.xyz, rhs.coords.xyz) + lhs.coords.w * rhs.coords.xyz + rhs.coords.w * lhs.coords.xyz;
    return Quat(float4(v, scalar));
}

/// Multiplies a quaternion by a vector (rotates the vector).
public float3 operator*(Quat q, float3 v) {
    let t = cross(q.coords.xyz, v) * 2.0;
    let c = cross(q.coords.xyz, t);
    return t * q.coords.w + c + v;
}