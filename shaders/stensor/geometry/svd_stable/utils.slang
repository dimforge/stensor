module utils;

// Rust-compatible signum: returns 1.0 for x >= 0.0, -1.0 for x < 0.0.
// Unlike HLSL's sign() which returns 0 for 0.0.
public func signum(float x) -> float {
    return select(x < 0.0, -1.0, 1.0);
}
