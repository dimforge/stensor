module rot2;


/// Compact representation of a 2D rotation.
public struct Rot2 {
    public float2 cos_sin;

    public __init() {
        this.cos_sin = float2(1.0, 0.0);
    }

    public __init(float2 cos_sin) {
        this.cos_sin = cos_sin;
    }

    /// Initializes a 2D rotation from an angle (radians).
    public static func from_angle(angle: float) -> Rot2 {
        return Rot2(float2(cos(angle), sin(angle)));
    }

    /// Returns `true` if `rot` isn’t zero.
    ///
    /// Fallible functions that return a Rot2 will generally return zero
    /// as the value to indicate failure.
    public func is_valid(rot: Rot2) -> bool {
        return rot.cos_sin.x != 0.0 || rot.cos_sin.y != 0.0;
    }



    /// Computes the rotation `R` required such that the `y` component of `R * v` is zero.
    ///
    /// Returns `Rot2()` (i.e. Rot2 filled with zeros) if no rotation is needed (i.e. if `v.y == 0`). Otherwise, this returns
    /// the rotation `R` such that `R * v = [ |v|, 0.0 ]^t` where `|v|` is the norm of `v`.
    public func cancel_y(v: float2) -> Rot2 {
        if (v.y != 0.0) {
            let r = float2(sign(v.x)) / length(v);
            let cos_sin = float2(v.x, -v.y) * r;
            return Rot2(cos_sin);
        } else {
            return Rot2();
        }
    }

    /// The quaternion representing an identity rotation.
    public func identity() -> Rot2 {
        return Rot2(float2(1.0, 0.0));
    }

    public func to_matrix() -> float2x2 {
        return float2x2(
            float2(this.cos_sin.x, this.cos_sin.y),
            float2(-this.cos_sin.y, this.cos_sin.x)
        );
    }

    /// The inverse of a 2d rotation.
    public func inv() -> Rot2 {
        return Rot2(float2(this.cos_sin.x, -this.cos_sin.y));
    }

    /// Multiplies the inverse of a 2D rotation by a vector (applies inverse rotation to the vector).
    public func inv_mul_vec(v: float2) -> float2 {
        return float2(this.cos_sin.x * v.x + this.cos_sin.y * v.y, -this.cos_sin.y * v.x + this.cos_sin.x * v.y);
    }

    // Apply the rotation to rows i and i + 1 to the given 3x3 matrix.
    public func rotate_rows3(inout m: float3x3, i: uint) {
        for (var r = 0; r < 3; r++) {
            let v = float2(m[i][r], m[i + 1][r]);
            let rv = this.inv_mul_vec(v);
            m[i][r] = rv.x;
            m[i + 1][r] = rv.y;
        }
    }

    // Apply the rotation to rows i and i + 1 to the given 4x4 matrix.
    public func rotate_rows4(inout m: float4x4, i: uint) {
        for (var r = 0; r < 4; r++) {
            let v = float2(m[i][r], m[i + 1][r]);
            let rv = this.inv_mul_vec(v);
            m[i][r] = rv.x;
            m[i + 1][r] = rv.y;
        }
    }
}


/// Multiplication of two 2D rotations.
public Rot2 operator*(Rot2 lhs, Rot2 rhs) {
    let new_cos = lhs.cos_sin.x * rhs.cos_sin.x - lhs.cos_sin.y * rhs.cos_sin.y;
    let new_sin = lhs.cos_sin.y * rhs.cos_sin.x + lhs.cos_sin.x * rhs.cos_sin.y;
    return Rot2(float2(new_cos, new_sin));
}

/// Multiplies a 2D rotation by a vector (rotates the vector).
public float2 operator*(Rot2 r, float2 v) {
    return float2(r.cos_sin.x * v.x - r.cos_sin.y * v.y, r.cos_sin.y * v.x + r.cos_sin.x * v.y);
}