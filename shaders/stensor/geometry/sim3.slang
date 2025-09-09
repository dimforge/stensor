module sim3;

import stensor.geometry.quat;

/// An 3D similarity representing a uniform scale, followed by a rotation, followed by a translation.
public struct Sim3 {
    /// The similarity’s rotational part.
    public Quat rotation;
    /// The similarity’s translational (xyz) and scaling (w) part.
    public float4 translation_scale;

    /// The identity similarity.
    public __init() {
        this.rotation = Quat();
        this.translation_scale = float4(0.0, 0.0, 0.0, 1.0);
    }

    public __init(Quat rotation, float4 translation_scale) {
        this.rotation = rotation;
        this.translation_scale = translation_scale;
    }

    /// Inverts a similarity.
    public func inv() -> Sim3 {
        let scale = 1.0f / this.translation_scale.w;
        let rotation = this.rotation.inv();
        let translation = rotation * (-this.translation_scale.xyz) * scale;
        return Sim3(rotation, float4(translation, scale));
    }

    /// Multiplies a similarity and a point (scales, rotates then translates the point).
    public func mul_pt(pt: float3) -> float3 {
        return this.rotation * (pt * this.translation_scale.w) + this.translation_scale.xyz;
    }

    /// Multiplies the inverse of a similarity and a point (inv-translates, inv-rotates, then inv-scales the point).
    public func inv_mul_pt(pt: float3) -> float3 {
        return this.rotation.inv_mul_vec(pt - this.translation_scale.xyz) / this.translation_scale.w;
    }

    /// Multiplies a similarity and a vector (scales and rotates the vector; the translation is ignored).
    public func mul_vec(vec: float3) -> float3 {
        return this.rotation * vec * this.translation_scale.w;
    }

    /// Multiplies the inverse of a similarity and a vector (inv-rotates then inv-scales the point; the translation is ignored).
    public func inv_mul_vec(vec: float3) -> float3 {
        return this.rotation.inv_mul_vec(vec) / this.translation_scale.w;
    }

    /// Multiplies the inverse of a similarity and a unit vector.
    ///
    /// This is similar to `inv_mul_vec` but the scaling part of the similarity is ignored to preserve the vector’s unit size.
    public func inv_mul_unit_vec(vec: float3) -> float3 {
        return this.rotation.inv_mul_vec(vec);
    }
}

/// Multiplies two similarities.
public Sim3 operator*(Sim3 lhs, Sim3 rhs) {
    let rotation = lhs.rotation * rhs.rotation;
    let translation = lhs.translation_scale.xyz + lhs.rotation * rhs.translation_scale.xyz * lhs.translation_scale.w;
    return Sim3(rotation, float4(translation, lhs.translation_scale.w * rhs.translation_scale.w));
}