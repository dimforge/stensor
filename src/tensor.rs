//! Utilities for initializing and slicing tensors, matrices, vectors, and scalars gpu storage
//! buffers.

// TODO: feels like this should be in stensor instead of slang-hal

use crate::shapes::{GGML_IDS, MatrixOrdering, ViewShape};
use bytemuck::NoUninit;
use encase::ShaderType;
use nalgebra::{Dim, IsContiguous, Matrix, Storage};
use slang_hal::backend::{Backend, Buffer, DeviceValue, EncaseType, Encoder, ShaderBinding};
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use slang_hal::backend::WebGpu;

#[cfg(feature = "cuda")]
use crate::cuda::Cuda;
use slang_hal::{BufferUsages, ShaderArgs};
use slang_hal::shader::ShaderArgsError;

/// Helper struct for creating gpu storage buffers (scalars, vectors, matrices, tensors).
pub struct TensorBuilder {
    shape: [u32; 4],
    usage: BufferUsages,
    ordering: MatrixOrdering,
    label: Option<String>,
}

impl TensorBuilder {
    /// Starts building a storage buffer containing a single scalar value.
    pub fn scalar(usage: BufferUsages) -> Self {
        Self::tensor([1, 1, 1, 1], usage)
    }

    /// Starts building a storage buffer containing a vector.
    pub fn vector(dim: u32, usage: BufferUsages) -> Self {
        Self::tensor([dim, 1, 1, 1], usage)
    }

    /// Starts building a storage buffer containing a single matrix with `nrows` rows and
    /// `ncols` columns.
    pub fn matrix(nrows: u32, ncols: u32, usage: BufferUsages) -> Self {
        Self::tensor([nrows, ncols, 1, 1], usage)
    }

    /// Starts building a storage buffer containing a tensor with the specified `shape`.
    pub fn tensor(shape: [u32; 4], usage: BufferUsages) -> Self {
        Self {
            shape,
            usage,
            ordering: MatrixOrdering::ColumnMajor,
            label: None,
        }
    }

    /// The number of elements in this tensor.
    fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    /// Sets the matrix ordering for this tensor.
    pub fn ordering(mut self, ordering: MatrixOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Sets the debug label of this tensor.
    pub fn label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Builds the uninitialized gpu tensor.
    pub fn build_uninit<T: DeviceValue + NoUninit, B: Backend>(
        self,
        backend: &B,
    ) -> Result<GpuTensor<T, B>, B::Error> {
        let buffer = backend.uninit_buffer(self.len() as usize, self.usage)?;
        Ok(GpuTensor {
            shape: self.shape,
            buffer,
            ordering: self.ordering,
        })
    }

    /// Builds the gpu tensor.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn build_uninit_encased<T: DeviceValue + EncaseType, B: Backend>(
        self,
        backend: &B,
    ) -> Result<GpuTensor<T, B>, B::Error> {
        let buffer = backend.uninit_buffer_encased(self.len() as usize, self.usage)?;
        Ok(GpuTensor {
            shape: self.shape,
            buffer,
            ordering: self.ordering,
        })
    }

    // /// Builds this tensor with raw bytes given for its initial value.
    // pub fn build_bytes<T: DeviceValue>(self, device: &Device, data: &[u8]) -> WgpuTensor<T, DIM> {
    //     let buffer = device.create_buffer_init(&BufferInitDescriptor {
    //         label: self.label.as_deref(),
    //         contents: bytemuck::cast_slice(data),
    //         usage: self.usage,
    //     });
    //
    //     GpuTensor {
    //         shape: self.shape,
    //         buffer,
    //     }
    // }

    // /// Builds this tensor with raw bytes given for its initial value.
    // pub fn build_encase<T: DeviceValue>(self, device: &Device, data: impl AsRef<[T]>) -> WgpuTensor<T, DIM>
    // where
    //     T: EncaseTypes,
    // {
    //     let vector = data.as_ref();
    //     let mut bytes = vec![]; // TODO: can we avoid the allocation?
    //     let mut buffer = StorageBuffer::new(&mut bytes);
    //     buffer.write(vector).unwrap();
    //     self.build_bytes(device, &bytes)
    // }

    /// Builds this tensor with an array of values given for its initial value.
    pub fn build_init<T: DeviceValue + NoUninit, B: Backend>(
        self,
        backend: &B,
        data: &[T],
    ) -> Result<GpuTensor<T, B>, B::Error> {
        assert!(
            data.len() as u64 >= self.len(),
            "Incorrect number of elements provided for initializing Tensor.\
            Expected at least {}, found {}",
            self.len(),
            data.len()
        );

        let buffer = backend.init_buffer(data, self.usage)?;
        Ok(GpuTensor {
            shape: self.shape,
            buffer,
            ordering: self.ordering,
        })
    }

    /// Builds this tensor with an array of encase-encoded values given for its initial value.
    pub fn build_encased<T: DeviceValue + EncaseType, B: Backend>(
        self,
        backend: &B,
        data: &[T],
    ) -> Result<GpuTensor<T, B>, B::Error> {
        assert!(
            data.len() as u64 >= self.len(),
            "Incorrect number of elements provided for initializing Tensor.\
            Expected at least {}, found {}",
            self.len(),
            data.len()
        );

        let buffer = backend.init_buffer_encased(data, self.usage)?;
        Ok(GpuTensor {
            shape: self.shape,
            buffer,
            ordering: self.ordering,
        })
    }
}

/// Type alias for a vector stored on the GPU.
pub type GpuVector<T, B> = GpuTensor<T, B>;
/// Type alias for a matrix stored on the GPU.
pub type GpuMatrix<T, B> = GpuTensor<T, B>;
/// Type alias for a scalar stored on the GPU.
pub type GpuScalar<T, B> = GpuTensor<T, B>;

/// A tensor stored in the GPU.
///
/// When the tensor is a matrix, they are generally seen as being column-major.
pub struct GpuTensor<T: DeviceValue, B: Backend> {
    shape: [u32; 4],
    buffer: B::Buffer<T>,
    ordering: MatrixOrdering,
}

/// Type alias for a tensor stored on the WebGPU backend.
pub type WgpuTensor<T> = GpuTensor<T, WebGpu>;
#[cfg(feature = "cuda")]
pub type CudaTensor<T> = GpuTensor<T, Cuda>;

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    /// Returns the matrix ordering of this tensor.
    pub fn ordering(&self) -> MatrixOrdering {
        self.ordering
    }

    /// Returns a transposed version of this tensor.
    pub fn transposed(mut self) -> Self {
        self.transpose();
        self
    }

    /// Transposes this tensor in place.
    pub fn transpose(&mut self) {
        self.shape.swap(0, 1);
        self.ordering = self.ordering.transpose();
    }

    /// Does this tensor contain zero elements?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this tensor.
    pub fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    // /// The tensor’s rank.
    // pub fn rank(&self) -> u64 {
    //     self.shape.iter().filter(|i| **i != 1).count() as u64
    // }

    /// The maximum number of elements this tensor can hold without needing a resize of the
    /// underlying GPU buffer.
    pub fn capacity(&self) -> u64
    where
        T: NoUninit,
    {
        self.buffer.len() as u64
    }

    /// The maximum number of elements this tensor can hold without needing a resize of the
    /// underlying GPU buffer.
    pub fn capacity_encased(&self) -> u64
    where
        T: EncaseType,
    {
        self.buffer.len_encased() as u64
    }

    /// The tensor’s order (i.e. the number of dimensions with a size > 1).
    pub fn order(&self) -> u8 {
        self.shape.iter().map(|s| (*s > 1) as u8).sum()
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size(&self, i: usize) -> u32 {
        self.shape[i]
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size_ggml(&self, i: usize) -> u32 {
        self.size(GGML_IDS[i])
    }

    /// Size of this tensor along the dimension `i`.
    pub fn stride(&self, i: usize) -> u32 {
        self.as_view().view_shape.stride[i]
    }

    /// Size of this tensor along the dimension `i`.
    pub fn stride_ggml(&self, i: usize) -> u32 {
        self.stride(GGML_IDS[i])
    }

    /// The size, in bytes, of this tensor’s content.
    pub fn bytes_len(&self) -> u64
    where
        T: DeviceValue,
    {
        std::mem::size_of::<T>() as u64 * self.len()
    }

    // /// The size, in bytes, of this tensor’s content.
    // pub fn bytes_len_encased(&self) -> u64
    // where
    //     T: ShaderType,
    // {
    //     T::min_size().get() * self.len()
    // }

    // /// Queues a buffer-to-buffer copy from `source` to `self`.
    // ///
    // /// Panics if the lengths do not match.
    // pub fn copy_from(&self, encoder: &mut CommandEncoder, source: &GpuTensor<T, B>)
    // where
    //     T: DeviceValue,
    // {
    //     assert_eq!(self.len(), source.len());
    //     encoder.copy_buffer_to_buffer(&source.buffer, 0, &self.buffer, 0, self.bytes_len())
    // }
    //
    // pub fn copy_from_encased(&self, encoder: &mut CommandEncoder, source: &GpuTensor<T, B>)
    // where
    //     T: ShaderType,
    // {
    //     assert_eq!(self.len(), source.len());
    //     encoder.copy_buffer_to_buffer(&source.buffer, 0, &self.buffer, 0, self.bytes_len_encased())
    // }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    pub fn copy_from_view<'a>(
        &mut self,
        encoder: &mut B::Encoder,
        source: impl Into<GpuTensorView<'a, T, B>>,
    ) -> Result<(), B::Error>
    where
        T: DeviceValue + NoUninit,
    {
        let source = source.into();
        let copy_len = self.len();

        // FIXME: assert that the source view is contiguous in a way that is
        //        compatible with `self`.
        encoder.copy_buffer_to_buffer(
            source.buffer,
            source.offset as usize,
            &mut self.buffer,
            0,
            copy_len as usize,
        )
    }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    pub fn copy_from_view_encased<'a>(
        &mut self,
        encoder: &mut B::Encoder,
        source: impl Into<GpuTensorView<'a, T, B>>,
    ) -> Result<(), B::Error>
    where
        T: DeviceValue + ShaderType,
    {
        let source = source.into();
        let copy_len = self.len();

        // FIXME: assert that the source view is contiguous in a way that is
        //        compatible with `self`.
        encoder.copy_buffer_to_buffer_encased(
            source.buffer,
            source.offset as usize,
            &mut self.buffer,
            0,
            copy_len as usize,
        )
    }

    /// The tensor’s shape (typically `[num_rows, num_cols, ...]`).
    pub fn shape(&self) -> [u32; 4] {
        self.shape
    }

    /// The tensor’s underlying GPU buffer.
    pub fn buffer(&self) -> &B::Buffer<T> {
        &self.buffer
    }

    /// The tensor’s underlying GPU buffer.
    pub fn buffer_mut(&mut self) -> &mut B::Buffer<T> {
        &mut self.buffer
    }

    /// Extracts the underlying GPU buffer.
    pub fn into_inner(self) -> B::Buffer<T> {
        self.buffer
    }

    /// Builds a tensor view sharing the same shape, stride, and buffer, as `self`.
    pub fn as_view(&self) -> GpuTensorView<'_, T, B> {
        GpuTensorView {
            view_shape: ViewShape::contiguous(self.shape, self.ordering),
            offset: 0,
            buffer: &self.buffer,
        }
    }

    /// Builds a mutable tensor view sharing the same shape, stride, and buffer, as `self`.
    pub fn as_view_mut(&mut self) -> GpuTensorViewMut<'_, T, B> {
        GpuTensorViewMut {
            view_shape: ViewShape::contiguous(self.shape, self.ordering),
            offset: 0,
            buffer: &mut self.buffer,
        }
    }

    fn vector_dim(&self) -> usize {
        let dim = match self.ordering {
            MatrixOrdering::RowMajor => 1,
            MatrixOrdering::ColumnMajor => 0,
        };
        let mut required_shape = [1; 4];
        required_shape[dim] = self.shape[dim];
        assert_eq!(
            required_shape, self.shape,
            "Operation only supported on vector tensors."
        );
        dim
    }

    // /// Reads the buffer’s content into a vector.
    // pub async fn read_bytes<'a>(&'a self, device: &'a Device) -> anyhow::Result<BufferView<'a>> {
    //     // TODO: could probably be optimized?
    //     let buffer_slice = self.buffer.slice(..);
    //
    //     #[cfg(not(target_arch = "wasm32"))]
    //     {
    //         let (sender, receiver) = async_channel::bounded(1);
    //         buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
    //             sender.send_blocking(v).unwrap()
    //         });
    //         device.poll(wgpu::PollType::wait());
    //         receiver.recv().await?.unwrap();
    //     }
    //     #[cfg(target_arch = "wasm32")]
    //     {
    //         let (sender, receiver) = async_channel::bounded(1);
    //         buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
    //             let _ = sender.force_send(v).unwrap();
    //         });
    //         device.poll(wgpu::PollType::wait());
    //         receiver.recv().await?.unwrap();
    //     }
    //
    //     let data = buffer_slice.get_mapped_range();
    //     Ok(data)
    // }
    //
    // /// Reads the buffer’s content into a slice.
    // pub async fn read_to(&self, device: &Device, out: &mut [T]) -> anyhow::Result<()>
    // where
    //     T: DeviceValue,
    // {
    //     let data = self.read_bytes(device).await?;
    //     let result = bytemuck::try_cast_slice(&data)?;
    //     out.copy_from_slice(result);
    //     drop(data);
    //     self.buffer.unmap();
    //     Ok(())
    // }
    //
    // /// Reads the buffer’s content into a vector.
    // pub async fn read(&self, device: &Device) -> anyhow::Result<Vec<T>>
    // where
    //     T: DeviceValue,
    // {
    //     let data = self.read_bytes(device).await?;
    //     let result = bytemuck::try_cast_slice(&data)?.to_vec();
    //     drop(data);
    //     self.buffer.unmap();
    //     Ok(result)
    // }
    //
    // /// Reads the buffer’s content into a vector.
    // pub async fn read_encased(&self, device: &Device) -> anyhow::Result<Vec<T>>
    // where
    //     T: ShaderType + ReadFrom + ShaderSize + CreateFrom,
    // {
    //     let data = self.read_bytes(device).await?;
    //     let mut result = vec![];
    //     let bytes = data.as_ref();
    //     let buffer = StorageBuffer::new(&bytes);
    //     buffer.read(&mut result)?;
    //     drop(data);
    //     self.buffer.unmap();
    //     Ok(result)
    // }
}

impl<'a, T: DeviceValue, B: Backend> From<&'a Arc<GpuTensor<T, B>>> for GpuTensorView<'a, T, B> {
    fn from(val: &'a Arc<GpuTensor<T, B>>) -> Self {
        val.as_view()
    }
}

impl<'a, T: DeviceValue, B: Backend> From<&'a GpuTensor<T, B>> for GpuTensorView<'a, T, B> {
    fn from(val: &'a GpuTensor<T, B>) -> Self {
        val.as_view()
    }
}

impl<'a, T: DeviceValue, B: Backend> From<&'a mut GpuTensor<T, B>> for GpuTensorViewMut<'a, T, B> {
    fn from(val: &'a mut GpuTensor<T, B>) -> Self {
        val.as_view_mut()
    }
}

/// A view over a mutable stensor.
///
/// This is a view over an entier tensor, or only part of it, with a shape that doesn’t necessarily
/// match the original tensor’s shape.
pub struct GpuTensorViewMut<'a, T: DeviceValue, B: Backend> {
    view_shape: ViewShape,
    buffer: &'a mut B::Buffer<T>,
    offset: u32,
}

/// A view over a tensor.
///
/// This is a view over an entier tensor, or only part of it, with a shape that doesn’t necessarily
/// match the original tensor’s shape.
pub struct GpuTensorView<'a, T: DeviceValue, B: Backend> {
    view_shape: ViewShape,
    buffer: &'a B::Buffer<T>,
    offset: u32,
}

impl<'a, T: DeviceValue, B: Backend> Clone for GpuTensorView<'a, T, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: DeviceValue, B: Backend> Copy for GpuTensorView<'a, T, B> {}

impl<'a, T: DeviceValue, B: Backend> GpuTensorView<'a, T, B> {
    /// Returns the [`MatrixOrdering`] of thes matrix if one of its row or column strides is 1.
    pub fn ordering(&self) -> Option<MatrixOrdering> {
        self.view_shape.ordering()
    }

    /// Checks if this tensor is contiguous in memory.
    ///
    /// Returns the [`MatrixOrdering`] under which this tensor can be interpreted as contiguous.
    pub fn is_contiguous(&self) -> Option<MatrixOrdering> {
        self.view_shape.is_contiguous()
    }

    /// Checks if `self` contains the same number oof elements and matches exactly the layout of
    /// its underlying `GpuTensor`.
    ///
    /// If it matches, returns the tensor's matrix ordering.
    pub fn is_entire_tensor(&self) -> Option<MatrixOrdering>
    where
        T: NoUninit,
    {
        if self.buffer.len() == self.len() as usize && self.offset == 0 {
            self.is_contiguous()
        } else {
            None
        }
    }

    /// The view’s shape.
    pub fn shape(&self) -> ViewShape {
        self.view_shape
    }

    /// The view’s buffer.
    pub fn buffer(&self) -> B::BufferSlice<'_, T> {
        self.buffer.slice(self.offset as usize..)
    }

    /// The view’s underlying buffer without any offset.
    pub fn raw_buffer(&self) -> &B::Buffer<T> {
        self.buffer
    }

    /// Is this view empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this view.
    pub fn len(&self) -> u64 {
        self.view_shape.len()
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size(&self, i: usize) -> u32 {
        self.view_shape.size[i]
    }

    /// Size of this tensor along the dimension `i`.
    pub fn size_ggml(&self, i: usize) -> u32 {
        self.view_shape.size[GGML_IDS[i]]
    }

    /// Stride of this tensor along the dimension `i`.
    pub fn stride(&self, i: usize) -> u32 {
        self.view_shape.stride[i]
    }

    /// Stride of this tensor along the dimension `i`.
    pub fn stride_ggml(&self, i: usize) -> u32 {
        self.view_shape.stride[GGML_IDS[i]]
    }

    /// Returns a transposed view of this tensor.
    pub fn transposed(&self) -> Self {
        self.permute([1, 0, 2, 3])
    }

    /// Permutes the dimensions of this view according to the given permutation array.
    pub fn permute(&self, permutations: [usize; 4]) -> Self {
        Self {
            view_shape: self.view_shape.permute(permutations),
            offset: self.offset,
            buffer: self.buffer,
        }
    }

    /// Permutes the dimensions according to GGML's dimension ordering convention.
    pub fn permute_ggml(&self, permutations: [usize; 4]) -> Self {
        Self {
            view_shape: self.view_shape.permute_ggml(permutations),
            offset: self.offset,
            buffer: self.buffer,
        }
    }

    /// Reshapes this view with an explicit ordering to avoid ambiguities.
    ///
    /// This is useful when the original shape has 1 row and 1 column.
    pub fn reshape_with_ordering<const DIM2: usize>(
        &self,
        shape: [u32; DIM2],
        ordering: MatrixOrdering,
    ) -> Self {
        assert!(DIM2 <= 4);
        let mut shape4 = [1; 4];
        shape4[..DIM2].copy_from_slice(&shape);
        let view_shape = ViewShape::contiguous(shape4, ordering);
        self.view(0, shape4, view_shape.stride.map(Some))
    }

    /// Reshapes this view to the specified shape, preserving the matrix ordering.
    pub fn reshape<const DIM2: usize>(&self, shape: [u32; DIM2]) -> Self {
        self.view(0, shape, [None; DIM2])
    }

    /// Reshapes this view using GGML's dimension ordering convention.
    pub fn reshape_ggml<const DIM2: usize>(&self, mut shape: [u32; DIM2]) -> Self {
        shape.swap(0, 1);

        if self.view_shape.size[0] == 1 && self.view_shape.size[1] == 1 {
            // Resolve ambiguity. GGML is always row-major.
            self.reshape_with_ordering(shape, MatrixOrdering::RowMajor)
        } else {
            self.reshape(shape)
        }
    }

    /// Reshapes this view using GGML's ordering with an explicit matrix ordering.
    ///
    /// This is useful to avoid ambiguities when the original shape has 1 row and 1 column.
    pub fn reshape_ggml_with_ordering<const DIM2: usize>(
        &self,
        mut shape: [u32; DIM2],
        ordering: MatrixOrdering,
    ) -> Self {
        shape.swap(0, 1);
        self.reshape_with_ordering(shape, ordering)
    }

    /// Creates a view of a sub-tensor with the specified offset, shape, and optional strides.
    pub fn view<const DIM2: usize>(
        &self,
        mut offset: u32,
        shape: [u32; DIM2],
        stride: [Option<u32>; DIM2],
    ) -> Self {
        let available_elts = self.view_shape.size.iter().product::<u32>();
        let needed_elts = shape.iter().product::<u32>() + offset;
        assert!(
            needed_elts <= available_elts,
            "Source tensor is too small for reshaping. Expected at least {needed_elts} elements (shape: {shape:?}, offset: {offset}), found {available_elts} (shape: {:?})",
            self.view_shape.size
        );

        offset += self.offset;

        GpuTensorView {
            view_shape: self.view_shape.view(shape, stride),
            offset,
            buffer: self.buffer,
        }
    }

    /// Creates a view using GGML's dimension ordering convention.
    pub fn view_ggml<const DIM2: usize>(
        &self,
        offset: u32,
        mut shape: [u32; DIM2],
        mut stride: [Option<u32>; DIM2],
    ) -> Self {
        shape.swap(0, 1);
        stride.swap(0, 1);
        self.view(offset, shape, stride)
    }

    /// Returns a view of the `matrix_id`-th matrix in this tensor.
    pub fn matrix(&self, matrix_id: u32) -> Self {
        let [nrows, ncols, nmats, ncubes] = self.view_shape.size;
        assert!(matrix_id < nmats);

        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, ncols, 1, ncubes],
                stride: self.view_shape.stride,
            },
            offset: self.offset + self.view_shape.stride[2] * matrix_id,
            buffer: self.buffer,
        }
    }

    /// Returns a view containing `new_ncols` columns starting from `first_col`.
    pub fn columns(&self, first_col: u32, new_ncols: u32) -> Self {
        let [nrows, ncols, nmats, ncubes] = self.view_shape.size;
        assert!(first_col + new_ncols < ncols);
        GpuTensorView {
            view_shape: ViewShape {
                size: [nrows, new_ncols, nmats, ncubes],
                stride: self.view_shape.stride,
            },
            offset: self.offset + self.view_shape.stride[1] * first_col,
            buffer: self.buffer,
        }
    }

    /// Returns a view of the specified column.
    pub fn column(&self, col: u32) -> Self {
        self.columns(col, 1)
    }

    /// Returns a view containing `new_nrows` rows starting from `first_row`.
    pub fn rows(&self, first_row: u32, new_nrows: u32) -> Self {
        let [nrows, ncols, nmats, ncubes] = self.view_shape.size;
        assert!(first_row + new_nrows < nrows);
        GpuTensorView {
            view_shape: ViewShape {
                size: [new_nrows, ncols, nmats, ncubes],
                stride: self.view_shape.stride,
            },
            offset: self.offset + self.view_shape.stride[0] * first_row,
            buffer: self.buffer,
        }
    }

    /// Returns a view of the specified row.
    pub fn row(&self, row: u32) -> Self {
        self.rows(row, 1)
    }
}

impl<'a, T: DeviceValue, B: Backend> GpuTensorViewMut<'a, T, B> {
    /// Converts this mutable view into an immutable view.
    pub fn as_ref(&self) -> GpuTensorView<'_, T, B> {
        GpuTensorView {
            view_shape: self.view_shape,
            buffer: &*self.buffer,
            offset: self.offset,
        }
    }
    /// Checks if this tensor is contiguous in memory.
    ///
    /// Returns the [`MatrixOrdering`] under which this tensor can be interpreted as contiguous.
    pub fn is_contiguous(&self) -> Option<MatrixOrdering> {
        self.as_ref().is_contiguous()
    }

    /// Checks if `self` contains the same number oof elements and matches exactly the layout of
    /// its underlying `GpuTensor`.
    ///
    /// If it matches, returns the tensor's matrix ordering.
    pub fn is_entire_tensor(&self) -> Option<MatrixOrdering>
    where
        T: NoUninit,
    {
        self.as_ref().is_entire_tensor()
    }

    /// The view’s shape.
    pub fn shape(&self) -> ViewShape {
        self.view_shape
    }

    // /// The view’s buffer.
    // pub fn buffer(&mut self) -> B::BufferSliceMut<'_, T> {
    //     self.buffer.slice_mut(self.offset as usize..)
    // }

    /// The view’s underlying buffer without any offset.
    pub fn raw_buffer(&mut self) -> &mut B::Buffer<T> {
        self.buffer
    }

    /// Is this view empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this view.
    pub fn len(&self) -> u64 {
        self.view_shape.len()
    }

    /// Returns a transposed mutable view of this tensor.
    pub fn transposed(&mut self) -> GpuTensorViewMut<'_, T, B> {
        self.permute([1, 0, 2, 3])
    }

    /// Permutes the dimensions of this mutable view according to the given permutation array.
    pub fn permute(&mut self, permutations: [usize; 4]) -> GpuTensorViewMut<'_, T, B> {
        GpuTensorViewMut {
            view_shape: self.view_shape.permute(permutations),
            offset: self.offset,
            buffer: self.buffer,
        }
    }

    /// Reshapes this mutable view to the specified shape.
    pub fn reshape<const DIM2: usize>(&mut self, shape: [u32; DIM2]) -> GpuTensorViewMut<'_, T, B> {
        self.view(0, shape, [None; DIM2])
    }

    /// Creates a mutable view of a sub-tensor with the specified offset, shape, and optional strides.
    pub fn view<const DIM2: usize>(
        &mut self,
        mut offset: u32,
        shape: [u32; DIM2],
        stride: [Option<u32>; DIM2],
    ) -> GpuTensorViewMut<'_, T, B> {
        offset += self.offset;

        let available_elts = self.view_shape.size.iter().product::<u32>();
        let needed_elts = shape.iter().product::<u32>() + offset;
        assert!(
            needed_elts <= available_elts,
            "Source tensor is too small for reshaping. Expected at least {needed_elts} elements (shape: {shape:?}), found {available_elts} (shape: {:?})",
            self.view_shape.size
        );

        GpuTensorViewMut {
            view_shape: self.view_shape.view(shape, stride),
            offset,
            buffer: self.buffer,
        }
    }

    /// Returns a mutable view of the `matrix_id`-th matrix in this tensor.
    pub fn matrix(&mut self, matrix_id: u32) -> GpuTensorViewMut<'_, T, B> {
        let [nrows, ncols, nmats, ncubes] = self.view_shape.size;
        assert!(matrix_id < nmats);

        GpuTensorViewMut {
            view_shape: ViewShape {
                size: [nrows, ncols, 1, ncubes],
                stride: self.view_shape.stride,
            },
            offset: self.offset + self.view_shape.stride[2] * matrix_id,
            buffer: self.buffer,
        }
    }

    /// Returns a mutable view containing `new_ncols` columns starting from `first_col`.
    pub fn columns(&mut self, first_col: u32, new_ncols: u32) -> GpuTensorViewMut<'_, T, B> {
        let [nrows, ncols, nmats, ncubes] = self.view_shape.size;
        assert!(first_col + new_ncols < ncols);
        GpuTensorViewMut {
            view_shape: ViewShape {
                size: [nrows, new_ncols, nmats, ncubes],
                stride: self.view_shape.stride,
            },
            offset: self.offset + self.view_shape.stride[1] * first_col,
            buffer: self.buffer,
        }
    }

    /// Returns a mutable view of the specified column.
    pub fn column(&mut self, col: u32) -> GpuTensorViewMut<'_, T, B> {
        self.columns(col, 1)
    }

    /// Returns a mutable view containing `new_nrows` rows starting from `first_row`.
    pub fn rows(&mut self, first_row: u32, new_nrows: u32) -> GpuTensorViewMut<'_, T, B> {
        let [nrows, ncols, nmats, ncubes] = self.view_shape.size;
        assert!(first_row + new_nrows < nrows);
        GpuTensorViewMut {
            view_shape: ViewShape {
                size: [new_nrows, ncols, nmats, ncubes],
                stride: self.view_shape.stride,
            },
            offset: self.offset + self.view_shape.stride[0] * first_row,
            buffer: self.buffer,
        }
    }

    /// Returns a mutable view of the specified row.
    pub fn row(&mut self, row: u32) -> GpuTensorViewMut<'_, T, B> {
        self.rows(row, 1)
    }
}

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    /// Reshapes this tensor to the specified shape.
    pub fn reshape<const DIM2: usize>(&self, shape: [u32; DIM2]) -> GpuTensorView<'_, T, B> {
        self.as_view().reshape_with_ordering(shape, self.ordering)
    }

    /// Reshapes this tensor using GGML's dimension ordering convention.
    pub fn reshape_ggml<const DIM2: usize>(&self, shape: [u32; DIM2]) -> GpuTensorView<'_, T, B> {
        self.as_view()
            .reshape_ggml_with_ordering(shape, self.ordering)
    }

    /// Permutes the dimensions of this tensor according to the given permutation array.
    pub fn permute(&self, permutations: [usize; 4]) -> GpuTensorView<'_, T, B> {
        self.as_view().permute(permutations)
    }

    /// Permutes the dimensions according to GGML's dimension ordering convention.
    pub fn permute_ggml(&self, permutations: [usize; 4]) -> GpuTensorView<'_, T, B> {
        self.as_view().permute_ggml(permutations)
    }

    /// Creates a view of a sub-tensor with the specified offset, shape, and optional strides.
    pub fn view<const DIM2: usize>(
        &self,
        offset: u32,
        shape: [u32; DIM2],
        stride: [Option<u32>; DIM2],
    ) -> GpuTensorView<'_, T, B> {
        self.as_view().view(offset, shape, stride)
    }

    /// Creates a view using GGML's dimension ordering convention.
    pub fn view_ggml<const DIM2: usize>(
        &self,
        offset: u32,
        shape: [u32; DIM2],
        stride: [Option<u32>; DIM2],
    ) -> GpuTensorView<'_, T, B> {
        self.as_view().view_ggml(offset, shape, stride)
    }

    /// Takes a view over the `i`-th column of `self`.
    pub fn column(&self, i: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().column(i)
    }

    /// Returns a view containing `ncols` columns starting from `first_col`.
    pub fn columns(&self, first_col: u32, ncols: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().columns(first_col, ncols)
    }

    /// Returns a view of the specified row.
    pub fn row(&self, i: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().row(i)
    }

    /// Returns a view containing `nrows` rows starting from `first_row`.
    pub fn rows(&self, first_row: u32, nrows: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().rows(first_row, nrows)
    }
}

impl<T: DeviceValue + NoUninit, B: Backend> GpuTensor<T, B> {
    /// Allocates a new matrix on the gpu with uninitialized elements.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn matrix_uninit(
        backend: &B,
        nrows: u32,
        ncols: u32,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue,
    {
        TensorBuilder::matrix(nrows, ncols, usage).build_uninit(backend)
    }

    // pub fn uninit_encased(device: &Device, nrows: u32, ncols: u32, usage: BufferUsages) -> Self
    // where
    //     T: ShaderType,
    // {
    //     TensorBuilder::matrix(nrows, ncols, usage).build_uninit_encased(device)
    // }

    /// Allocates a new matrix on the gpu initialized from `matrix`.
    pub fn matrix<R: Dim, C: Dim, S: Storage<T, R, C> + IsContiguous>(
        backend: &B,
        matrix: &Matrix<T, R, C, S>,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + nalgebra::Scalar,
    {
        Self::matrix_with_ordering(backend, matrix, MatrixOrdering::default(), usage)
    }

    /// Allocates a new matrix on the gpu initialized from `matrix`.
    pub fn matrix_with_ordering<R: Dim, C: Dim, S: Storage<T, R, C> + IsContiguous>(
        backend: &B,
        matrix: &Matrix<T, R, C, S>,
        ordering: MatrixOrdering,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + nalgebra::Scalar,
    {
        TensorBuilder::matrix(matrix.nrows() as u32, matrix.ncols() as u32, usage)
            .ordering(ordering)
            .build_init(backend, matrix.as_slice())
    }
}

// impl<T: DeviceValue, B: Backend> GpuMatrix<T, B> {
//     pub fn slice(&self, (i, j): (u32, u32), (nrows, ncols): (u32, u32)) -> GpuTensorView<'_, T, B> {
//         GpuTensorView {
//             view_shape: ViewShape {
//                 size: [nrows, ncols, 1, 1],
//                 stride: [1, self.shape[0], self.shape[0] * self.shape[1], 1],
//             },
//             offset: i + j * nrows,
//             buffer: &self.buffer,
//         }
//     }
// }

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn vector_uninit(backend: &B, len: u32, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + NoUninit,
    {
        TensorBuilder::vector(len, usage).build_uninit(backend)
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` does not implement `NoUninit`, use [`GpuTensor::vector_encased`] instead.
    pub fn vector(
        backend: &B,
        vector: impl AsRef<[T]>,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + NoUninit,
    {
        let v = vector.as_ref();
        TensorBuilder::vector(v.len() as u32, usage).build_init(backend, v.as_ref())
    }
    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn vector_uninit_encased(
        backend: &B,
        len: u32,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + EncaseType,
    {
        TensorBuilder::vector(len, usage).build_uninit_encased(backend)
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    pub fn vector_encased(
        backend: &B,
        vector: impl AsRef<[T]>,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + EncaseType,
    {
        let v = vector.as_ref();
        TensorBuilder::vector(v.len() as u32, usage).build_encased(backend, v.as_ref())
    }
}

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    /// Allocates a new gpu storage buffer with a single uninitialized element.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn scalar_uninit(backend: &B, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + NoUninit,
    {
        TensorBuilder::scalar(usage).build_uninit(backend)
    }

    /// Allocates a new gpu storage buffer with a single uninitialized element.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub fn scalar_uninit_encased(backend: &B, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + EncaseType,
    {
        TensorBuilder::scalar(usage).build_uninit_encased(backend)
    }

    /// Allocates a new gpu storage buffer with a single element initialized to `value`.
    pub fn scalar(backend: &B, value: T, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + NoUninit,
    {
        TensorBuilder::scalar(usage).build_init(backend, &[value])
    }

    /// Allocates a new gpu storage buffer with a single element initialized to `value`.
    pub fn scalar_encased(backend: &B, value: T, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + EncaseType,
    {
        TensorBuilder::scalar(usage).build_encased(backend, &[value])
    }
}

impl<'b, B: Backend, T: DeviceValue> ShaderArgs<'b, B> for GpuTensor<T, B> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        name: &str,
        dispatch: &mut B::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        self.buffer.write_arg(binding, name, dispatch)
    }
}

macro_rules! append_and_remove(
    ($append: ident, $shift_remove: ident, $TraitBound: ident, $capacity: ident, $copy_buffer_to_buffer: ident, $uninit_buffer: ident, $write_buffer: ident) => {
        /// Append the `data` elements at the end of this tensor if it is a vector.
        ///
        /// Panics if the tensor isn’t a vector. The tensor is a vector if:
        /// - It is a row-major tensor and is made of a single row. Its size is `[1, *, 1, 1]` (where
        ///   `*` is any non-zero positive integer).
        /// - It is a column-major tensor and its size is made of a single column. Its size is
        ///   `[*, 1, 1, 1]` (where `*` is any non-zero positive integer).
        ///
        /// If the underlying GPU buffer is too small to contain the extra elements, it is automatically
        /// resized. If a resize happens, the tensor’s capacity is the next power of two sufficient
        /// to contain the appended data.
        // TODO: broadcast automatically to generalize to any tensor order.
        pub fn $append(&mut self, backend: &B, data: &[T]) -> Result<(), B::Error>
        where
            T: $TraitBound,
        {
            let dim_to_grow = self.vector_dim();
            let num_added = data.len();
            let curr_len = self.shape[dim_to_grow];
            let new_len = curr_len + num_added as u32;

            let mut encoder = backend.begin_encoding();


            if new_len as u64 >= self.$capacity() {
                // We need to grow the buffer.
                let new_capacity = new_len.next_power_of_two();
                // SAFETY: will be initialized by the buffer init.
                let mut new_buffer = backend.$uninit_buffer(
                    new_capacity as usize,
                    self.buffer().usage() | BufferUsages::COPY_DST
                )?;

                encoder.$copy_buffer_to_buffer(
                    &self.buffer,
                    0,
                    &mut new_buffer,
                    0,
                    curr_len as usize,
                )?;
                self.buffer = new_buffer;
            }

            backend.$write_buffer(&mut self.buffer, curr_len as u64, data)?;
            backend.submit(encoder)?;
            self.shape[dim_to_grow] = new_len;
            Ok(())
        }

        /// Removes a `range` of elements from this tensor if it is a vector, shifting back elements to
        /// fill the gap.
        ///
        /// Panics if the tensor isn’t a vector. The tensor is a vector if:
        /// - It is a row-major tensor and is made of a single row. Its size is `[1, *, 1, 1]` (where
        ///   `*` is any non-zero positive integer).
        /// - It is a column-major tensor and its size is made of a single column. Its size is
        ///   `[*, 1, 1, 1]` (where `*` is any non-zero positive integer).
        ///
        /// This method doesn’t change the tensor’s capacity so the internal GPU buffer isn’t resized.
        ///
        /// # Performance note
        ///
        /// This method is currently fairly expensive as it always involves the creation of a staging
        /// buffer for copying the data being moved. The staging buffer size is equal to the number of
        /// moved elements.
        ///
        /// # Panic
        ///
        /// Panics if `self` wasn’t created with the `BufferUsages::COPY_SRC | BufferUsages::COPY_DST` flags.
        /// Panics if the range is out of the bounds of `self`.
        ///
        /// # Return
        ///
        /// If the operation suceeded, returns the number of removed elements.
        // TODO: add a special case for targets capable of copying slices within the same buffer.
        // TODO: it would be worth benchmarking with doing the shift with a compute shader instead.
        pub fn $shift_remove(
            &mut self,
            backend: &B,
            range: impl RangeBounds<usize>,
        ) -> Result<usize, B::Error>
        where T: $TraitBound {
            let dim_to_shrink = self.vector_dim();
            let curr_len = self.shape[dim_to_shrink] as usize;
            let range_start = match range.start_bound() {
                Bound::Included(i) => *i,
                Bound::Excluded(i) => *i + 1,
                Bound::Unbounded => 0,
            };
            let range_end = match range.end_bound() {
                Bound::Included(i) => *i + 1,
                Bound::Excluded(i) => *i,
                Bound::Unbounded => curr_len,
            };

            if range_end <= range_start {
                // The range to remove is empty.
                return Ok(0);
            }

            assert!(range_end <= curr_len, "Range index out of bounds.");
            let num_elements_to_move = curr_len - range_end;

            // NOTE: if `curr_end == range_end` we don’t actually need to move any data, shrinking
            //       the shape is sufficient.
            if num_elements_to_move > 0 {
                // SAFETY: will be initialized with a buffer-to-buffer copy.
                let mut staging = backend.$uninit_buffer(
                    num_elements_to_move,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                )?;

                let mut encoder = backend.begin_encoding();
                encoder.$copy_buffer_to_buffer(
                    &self.buffer,
                    range_end,
                    &mut staging,
                    0,
                    num_elements_to_move,
                )?;
                encoder.$copy_buffer_to_buffer(
                    &staging,
                    0,
                    &mut self.buffer,
                    range_start,
                    num_elements_to_move,
                )?;
                backend.submit(encoder)?;
            }

            let num_removed = range_end - range_start;
            self.shape[dim_to_shrink] -= num_removed as u32;
            Ok(num_removed)
        }
    }
);

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    append_and_remove!(
        append,
        shift_remove,
        NoUninit,
        capacity,
        copy_buffer_to_buffer,
        uninit_buffer,
        write_buffer
    );
    append_and_remove!(
        append_encased,
        shift_remove_encased,
        EncaseType,
        capacity_encased,
        copy_buffer_to_buffer_encased,
        uninit_buffer_encased,
        write_buffer_encased
    );
}
