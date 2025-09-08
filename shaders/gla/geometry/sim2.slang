module sim2;

import gla.geometry.rot2;

/// An 2D similarity representing a uniform scale, followed by a rotation, followed by a translation.
public struct Sim2 {
    /// The similarity’s rotational part.
    public Rot2 rotation;
    /// The similarity’s translational part.
    public float2 translation;
    /// The similarity’s scaling part.
    public float scale;

    public __init() {
        this.rotation = Rot2();
        this.translation = float2(0.0);
        this.scale = 1.0;
    }

    public __init(Rot2 rotation, float2 translation, float scale) {
        this.rotation = rotation;
        this.translation = translation;
        this.scale = scale;
    }

    /// Inverts a similarity.
    public func inv() -> Sim2 {
        let scale = 1.0f / this.scale;
        let rotation = this.rotation.inv();
        let translation = rotation * (-this.translation) * scale;
        return Sim2(rotation, translation, scale);
    }

    /// Multiplies a similarity and a point (scales, rotates then translates the point).
    public func mul_pt(pt: float2) -> float2 {
        return this.rotation * (pt * this.scale) + this.translation;
    }

    /// Multiplies the inverse of a similarity and a point (inv-translates, inv-rotates, then inv-scales the point).
    public func inv_mul_pt(pt: float2) -> float2 {
        return this.rotation.inv_mul_vec(pt - this.translation) / this.scale;
    }

    /// Multiplies a similarity and a vector (scales and rotates the vector; the translation is ignored).
    public func mul_vec(vec: float2) -> float2 {
        return (this.rotation * vec) * this.scale;
    }

    /// Multiplies the inverse of a similarity and a vector (inv-rotates then inv-scales the point; the translation is ignored).
    public func inv_mul_vec(vec: float2) -> float2 {
        return this.rotation.inv_mul_vec(vec) / this.scale;
    }

    /// Multiplies the inverse of a similarity and a unit vector.
    ///
    /// This is similar to `inv_mul_vec` but the scaling part of the similarity is ignored to preserve the vector’s unit size.
    public func invMulUnitVec(vec: float2) -> float2 {
        return this.rotation.inv_mul_vec(vec);
    }
}

/// Multiplies two similarities.
public Sim2 operator*(Sim2 lhs, Sim2 rhs) {
    let rotation = lhs.rotation * rhs.rotation;
    let translation = lhs.translation + (lhs.rotation * rhs.translation) * lhs.scale;
    return Sim2(rotation, translation, lhs.scale * rhs.scale);
}
