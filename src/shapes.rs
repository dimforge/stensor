//! Tensor shape definition.

use slang_hal::backend::Backend;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Mutex;
use wgpu::BufferUsages;

pub const GGML_IDS: [usize; 4] = [1, 0, 2, 3];
pub const GGML_IDS_U32: [u32; 4] = [1, 0, 2, 3];

#[derive(Copy, Clone, PartialEq, Eq, Default, Debug, Hash)]
pub enum MatrixOrdering {
    #[default]
    ColumnMajor,
    RowMajor,
    // TODO: should we generalize this to a `MajorAxis(i)` where any
    //       dimension of the tensor can be interpreted as the main one?
}

impl MatrixOrdering {
    pub fn transpose(self) -> Self {
        match self {
            Self::ColumnMajor => Self::RowMajor,
            Self::RowMajor => Self::ColumnMajor,
        }
    }
}

#[derive(
    Debug, Copy, Clone, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable, encase::ShaderType,
)]
#[repr(C)]
/// The shape of a matrix view over a GPU tensor.
pub struct ViewShape {
    /// The tensor view’s number of rows, columns, matrices, and 3-tensors.
    pub size: [u32; 4],
    /// The stride along each dimension.
    pub stride: [u32; 4],
}

impl ViewShape {
    pub fn contiguous(size: [u32; 4], ordering: MatrixOrdering) -> Self {
        let stride = match ordering {
            MatrixOrdering::ColumnMajor => {
                [1, size[0], size[0] * size[1], size[0] * size[1] * size[2]]
            }
            MatrixOrdering::RowMajor => {
                [size[1], 1, size[0] * size[1], size[0] * size[1] * size[2]]
            }
        };
        Self { size, stride }
    }

    pub fn transpose(&self) -> Self {
        self.permute([1, 0, 2, 3])
    }

    pub fn maybe_transpose(&self, transpose: bool) -> Self {
        if transpose { self.transpose() } else { *self }
    }

    pub fn permute_ggml(&self, mut permutations: [usize; 4]) -> Self {
        permutations.swap(0, 1);
        self.permute(permutations.map(|i| GGML_IDS[i]))
    }

    pub fn permute(&self, permutations: [usize; 4]) -> Self {
        // Check all the permutation indices are valid and without
        // duplicate.
        assert_ne!(
            permutations[0], permutations[1],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[0], permutations[2],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[0], permutations[3],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[1], permutations[2],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[1], permutations[3],
            "Permutation indices must not overlap."
        );
        assert_ne!(
            permutations[2], permutations[3],
            "Permutation indices must not overlap."
        );
        assert!(permutations[0] <= 3);
        assert!(permutations[1] <= 3);
        assert!(permutations[2] <= 3);
        assert!(permutations[3] <= 3);
        let mut size = [0; 4];
        let mut stride = [0; 4];

        for k in 0..4 {
            size[permutations[k]] = self.size[k];
            stride[permutations[k]] = self.stride[k];
        }

        Self { size, stride }
    }

    /// Attempts to detect the matrix ordering from this shape.
    ///
    /// This check whether two successive elements of the same row (column-major), or the same
    /// column (row-major), are contiguous in memory.
    pub fn ordering(&self) -> Option<MatrixOrdering> {
        if self.stride[0] == 1 {
            Some(MatrixOrdering::ColumnMajor)
        } else if self.stride[1] == 1 {
            Some(MatrixOrdering::RowMajor)
        } else {
            None
        }
    }

    /// Checks if a tensor with this shape is contiguous in memory.
    ///
    /// Returns the [`MatrixOrdering`] under which this tensor can be interpreted as contiguous.
    pub fn is_contiguous(&self) -> Option<MatrixOrdering> {
        let [nrows, ncols, nmats, _] = self.size;

        if self.stride[0] == 1 {
            let expected_stride = [1, nrows, nrows * ncols, nrows * ncols * nmats];
            (expected_stride == self.stride).then_some(MatrixOrdering::ColumnMajor)
        } else if self.stride[1] == 1 {
            let expected_stride = [ncols, 1, nrows * ncols, nrows * ncols * nmats];
            (expected_stride == self.stride).then_some(MatrixOrdering::RowMajor)
        } else {
            None
        }
    }

    pub fn is_multiple_of(&self, of: Self) -> bool {
        for k in 0..4 {
            if self.size[k] % of.size[k] != 0 {
                return false;
            }
        }

        true
    }

    pub fn view<const DIM2: usize>(&self, shape: [u32; DIM2], stride: [Option<u32>; DIM2]) -> Self {
        assert!(DIM2 <= 4);

        let Some(mut ordering) = self.is_contiguous() else {
            panic!("Cannot take a view of a non-contiguous tensor.");
        };

        // Special case where the ordering is ambiguous.
        if self.size[0] == 1 && self.size[1] == 1 {
            // See if the provided stride allows breaking the ambiguity.
            if stride[0] == Some(1) {
                ordering = MatrixOrdering::ColumnMajor;
            } else if stride[1] == Some(1) {
                ordering = MatrixOrdering::RowMajor;
            } else if stride[0].is_none() || stride[1].is_none() {
                panic!("Ambiguous view matrix ordering. Row and column strides must be specified.")
            }
        }

        let mut size = [1; 4];
        size[..DIM2].copy_from_slice(&shape[..DIM2]);
        let mut strd = [None; 4];
        strd[..DIM2].copy_from_slice(&stride[..DIM2]);

        let stride = match ordering {
            MatrixOrdering::ColumnMajor => {
                let stride0 = strd[0].unwrap_or(1);
                let stride1 = strd[1].unwrap_or(stride0 * size[0]);
                let stride2 = strd[2].unwrap_or(stride1 * size[1]);
                let stride3 = strd[3].unwrap_or(stride2 * size[2]);
                [stride0, stride1, stride2, stride3]
            }
            MatrixOrdering::RowMajor => {
                let stride1 = strd[1].unwrap_or(1);
                let stride0 = strd[0].unwrap_or(stride1 * size[1]);
                let stride2 = strd[2].unwrap_or(stride0 * size[0]);
                let stride3 = strd[3].unwrap_or(stride2 * size[2]);
                [stride0, stride1, stride2, stride3]
            }
        };

        Self { size, stride }
    }

    /// Converts the shape `self` for a buffer `&[f32]` to a buffer `&[vec4f]`.
    pub fn f32_to_vec4(self) -> Self {
        let dim = if self.stride[0] == 1 {
            0
        } else {
            assert_eq!(self.stride[1], 1);
            1
        };

        assert_eq!(
            self.stride[dim], 1,
            "Cannot convert from f32 to vec4 with a stride[{dim}] of {} != 1",
            self.stride[dim]
        );
        assert_eq!(
            self.size[dim] % 4,
            0,
            "Matrix row count no properly aligned."
        );

        let new_stride = self.stride.map(|s| {
            assert!(s == 1 || s % 4 == 0);
            s.div_ceil(4)
        });
        let mut new_size = self.size;
        new_size[dim] /= 4;

        Self {
            size: new_size,
            stride: new_stride,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> u64 {
        (self.size[0] * self.size[1] * self.size[2] * self.size[3]) as u64
    }
}

/// A map between a `ViewShape` and a uniform storage `Buffer` containing its value on the gpu.
///
/// Ideally, we should use push-constants for view shapes. Unfortunately, push-constants is an
/// optional extension, so we have to emulate them with uniforms for maximum portability.
#[derive(Default)]
pub struct ViewShapeBuffers<B: Backend> {
    buffers: HashMap<ViewShape, B::Buffer<ViewShape>>,
    tmp_buffers: HashMap<ViewShape, B::Buffer<ViewShape>>,
    // TODO: is this still needed?
    recycled: Mutex<Vec<B::Buffer<ViewShape>>>,
}

impl<B: Backend> ViewShapeBuffers<B> {
    /// Creates an empty map.
    pub fn new(_backend: &B) -> Self {
        Self {
            buffers: HashMap::new(),
            tmp_buffers: HashMap::new(),
            recycled: Mutex::new(vec![]),
        }
    }

    pub fn clear_tmp(&mut self) {
        let mut recycled = self.recycled.lock().unwrap();
        recycled.extend(self.tmp_buffers.drain().map(|(_, buf)| buf));
    }

    pub fn put_tmp(&mut self, backend: &B, shape: ViewShape) -> Result<(), B::Error> {
        if self.contains(shape) {
            return Ok(());
        }

        let mut recycled = self.recycled.lock().unwrap();
        let buffer = if let Some(mut buffer) = recycled.pop() {
            backend.write_buffer(&mut buffer, 0, &[shape])?;
            buffer
        } else {
            // println!("Couldn’t find recycling for {:?}", shape);
            drop(recycled);
            Self::make_buffer(
                backend,
                shape,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::STORAGE,
            )?
        };

        self.tmp_buffers.insert(shape, buffer);
        Ok(())
    }

    fn make_buffer(
        backend: &B,
        shape: ViewShape,
        usage: BufferUsages,
    ) -> Result<B::Buffer<ViewShape>, B::Error> {
        // println!("Making buffer for shape: {:?}", shape);
        backend.init_buffer(&[shape], usage | BufferUsages::STORAGE)
    }

    pub fn contains(&self, shape: ViewShape) -> bool {
        self.buffers.contains_key(&shape) || self.tmp_buffers.contains_key(&shape)
    }

    pub fn insert(
        &mut self,
        backend: &B,
        shape: ViewShape,
    ) -> Result<&mut B::Buffer<ViewShape>, B::Error> {
        if let Some(buffer) = self.tmp_buffers.get_mut(&shape) {
            return Ok(buffer);
        }

        let buf = match self.buffers.entry(shape) {
            Entry::Vacant(e) => e.insert(Self::make_buffer(backend, shape, BufferUsages::UNIFORM)?),
            Entry::Occupied(e) => e.into_mut(),
        };
        Ok(buf)
    }

    /// Gets the gpu uniform storage `Buffer` containing the value of `shape`.
    ///
    /// Returns `None` if it doesn't exist.
    pub fn get(&self, shape: ViewShape) -> Option<&B::Buffer<ViewShape>> {
        self.tmp_buffers
            .get(&shape)
            .or_else(|| self.buffers.get(&shape))
    }
}
