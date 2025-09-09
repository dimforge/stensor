//! Utilities for initializing and slicing tensors, matrices, vectors, and scalars gpu storage
//! buffers.

// TODO: feels like this should be in stensor instead of slang-hal

use slang_hal::backend::{Backend, Buffer, DeviceValue, EncaseType, Encoder, ShaderBinding};
use crate::shapes::{GGML_IDS, MatrixOrdering, ViewShape};
use bytemuck::Pod;
use encase::ShaderType;
use nalgebra::{Dim, IsContiguous, Matrix, Storage};
use std::sync::Arc;

use slang_hal::backend::WebGpu;
use wgpu::BufferUsages;

use slang_hal::ShaderArgs;
#[cfg(feature = "cuda")]
use crate::cuda::Cuda;
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

    pub fn ordering(mut self, ordering: MatrixOrdering) -> Self {
        self.ordering = ordering;
        self
    }

    /// Sets the debug label of this tensor.
    pub fn label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Builds the gpu tensor.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub unsafe fn build_uninit<T: DeviceValue + Pod, B: Backend>(
        self,
        backend: &B,
    ) -> Result<GpuTensor<T, B>, B::Error> {
        let buffer = unsafe { backend.uninit_buffer(self.len() as usize, self.usage)? };
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
    pub unsafe fn build_uninit_encased<T: DeviceValue + EncaseType, B: Backend>(
        self,
        backend: &B,
    ) -> Result<GpuTensor<T, B>, B::Error> {
        let buffer = unsafe { backend.uninit_buffer_encased(self.len() as usize, self.usage)? };
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
    pub fn build_init<T: DeviceValue + Pod, B: Backend>(
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

pub type GpuVector<T, B> = GpuTensor<T, B>;
pub type GpuMatrix<T, B> = GpuTensor<T, B>;
pub type GpuScalar<T, B> = GpuTensor<T, B>;

/// A tensor stored in the GPU.
///
/// When the tensor is a matrix, they are generally seen as being column-major.
pub struct GpuTensor<T: DeviceValue, B: Backend> {
    shape: [u32; 4],
    buffer: B::Buffer<T>,
    ordering: MatrixOrdering,
}

pub type WgpuTensor<T> = GpuTensor<T, WebGpu>;
#[cfg(feature = "cuda")]
pub type CudaTensor<T> = GpuTensor<T, Cuda>;

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    pub fn ordering(&self) -> MatrixOrdering {
        self.ordering
    }

    pub fn transposed(mut self) -> Self {
        self.transpose();
        self
    }

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
        T: DeviceValue + Pod,
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
    pub fn is_entire_tensor(&self) -> Option<MatrixOrdering> {
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

    pub fn transposed(&self) -> Self {
        self.permute([1, 0, 2, 3])
    }

    pub fn permute(&self, permutations: [usize; 4]) -> Self {
        Self {
            view_shape: self.view_shape.permute(permutations),
            offset: self.offset,
            buffer: self.buffer,
        }
    }

    pub fn permute_ggml(&self, permutations: [usize; 4]) -> Self {
        Self {
            view_shape: self.view_shape.permute_ggml(permutations),
            offset: self.offset,
            buffer: self.buffer,
        }
    }

    // Specify the ordering explicitly to avoid ambiguities if the original shape has 1 row and 1 col.
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

    pub fn reshape<const DIM2: usize>(&self, shape: [u32; DIM2]) -> Self {
        self.view(0, shape, [None; DIM2])
    }

    pub fn reshape_ggml<const DIM2: usize>(&self, mut shape: [u32; DIM2]) -> Self {
        shape.swap(0, 1);

        if self.view_shape.size[0] == 1 && self.view_shape.size[1] == 1 {
            // Resolve ambiguity. GGML is always row-major.
            self.reshape_with_ordering(shape, MatrixOrdering::RowMajor)
        } else {
            self.reshape(shape)
        }
    }

    // Specify the ordering explicitly to avoid ambiguities if the original shape has 1 row and 1 col.
    pub fn reshape_ggml_with_ordering<const DIM2: usize>(
        &self,
        mut shape: [u32; DIM2],
        ordering: MatrixOrdering,
    ) -> Self {
        shape.swap(0, 1);
        self.reshape_with_ordering(shape, ordering)
    }

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

    pub fn column(&self, col: u32) -> Self {
        self.columns(col, 1)
    }

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

    pub fn row(&self, row: u32) -> Self {
        self.rows(row, 1)
    }
}

impl<'a, T: DeviceValue, B: Backend> GpuTensorViewMut<'a, T, B> {
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
    pub fn is_entire_tensor(&self) -> Option<MatrixOrdering> {
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

    pub fn transposed(&mut self) -> GpuTensorViewMut<'_, T, B> {
        self.permute([1, 0, 2, 3])
    }

    pub fn permute(&mut self, permutations: [usize; 4]) -> GpuTensorViewMut<'_, T, B> {
        GpuTensorViewMut {
            view_shape: self.view_shape.permute(permutations),
            offset: self.offset,
            buffer: self.buffer,
        }
    }

    pub fn reshape<const DIM2: usize>(&mut self, shape: [u32; DIM2]) -> GpuTensorViewMut<'_, T, B> {
        self.view(0, shape, [None; DIM2])
    }

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

    pub fn column(&mut self, col: u32) -> GpuTensorViewMut<'_, T, B> {
        self.columns(col, 1)
    }

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

    pub fn row(&mut self, row: u32) -> GpuTensorViewMut<'_, T, B> {
        self.rows(row, 1)
    }
}

impl<T: DeviceValue, B: Backend> GpuTensor<T, B> {
    pub fn reshape<const DIM2: usize>(&self, shape: [u32; DIM2]) -> GpuTensorView<'_, T, B> {
        self.as_view().reshape_with_ordering(shape, self.ordering)
    }

    pub fn reshape_ggml<const DIM2: usize>(&self, shape: [u32; DIM2]) -> GpuTensorView<'_, T, B> {
        self.as_view()
            .reshape_ggml_with_ordering(shape, self.ordering)
    }

    pub fn permute(&self, permutations: [usize; 4]) -> GpuTensorView<'_, T, B> {
        self.as_view().permute(permutations)
    }

    pub fn permute_ggml(&self, permutations: [usize; 4]) -> GpuTensorView<'_, T, B> {
        self.as_view().permute_ggml(permutations)
    }

    pub fn view<const DIM2: usize>(
        &self,
        offset: u32,
        shape: [u32; DIM2],
        stride: [Option<u32>; DIM2],
    ) -> GpuTensorView<'_, T, B> {
        self.as_view().view(offset, shape, stride)
    }

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

    pub fn columns(&self, first_col: u32, ncols: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().columns(first_col, ncols)
    }

    pub fn row(&self, i: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().row(i)
    }

    pub fn rows(&self, first_row: u32, nrows: u32) -> GpuTensorView<'_, T, B> {
        self.as_view().rows(first_row, nrows)
    }
}

impl<T: DeviceValue + Pod, B: Backend> GpuTensor<T, B> {
    /// Allocates a new matrix on the gpu with uninitialized elements.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub unsafe fn matrix_uninit(
        backend: &B,
        nrows: u32,
        ncols: u32,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue,
    {
        unsafe { TensorBuilder::matrix(nrows, ncols, usage).build_uninit(backend) }
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
    pub unsafe fn vector_uninit(
        backend: &B,
        len: u32,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + Pod,
    {
        unsafe { TensorBuilder::vector(len, usage).build_uninit(backend) }
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` does not implement `Pod`, use [`GpuMatrix::encase`] instead.
    pub fn vector(
        backend: &B,
        vector: impl AsRef<[T]>,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + Pod,
    {
        let v = vector.as_ref();
        TensorBuilder::vector(v.len() as u32, usage).build_init(backend, v.as_ref())
    }
    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub unsafe fn vector_uninit_encased(
        backend: &B,
        len: u32,
        usage: BufferUsages,
    ) -> Result<Self, B::Error>
    where
        T: DeviceValue + EncaseType,
    {
        unsafe { TensorBuilder::vector(len, usage).build_uninit_encased(backend) }
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` does not implement `Pod`, use [`GpuMatrix::encase`] instead.
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
    pub unsafe fn scalar_uninit(backend: &B, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + Pod,
    {
        unsafe { TensorBuilder::scalar(usage).build_uninit(backend) }
    }

    /// Allocates a new gpu storage buffer with a single uninitialized element.
    ///
    /// # Safety
    ///
    /// The returned buffer must be initialized before being read from.
    pub unsafe fn scalar_uninit_encased(backend: &B, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + EncaseType,
    {
        unsafe { TensorBuilder::scalar(usage).build_uninit_encased(backend) }
    }

    /// Allocates a new gpu storage buffer with a single element initialized to `value`.
    pub fn scalar(backend: &B, value: T, usage: BufferUsages) -> Result<Self, B::Error>
    where
        T: DeviceValue + Pod,
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
