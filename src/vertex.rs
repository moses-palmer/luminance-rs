//! Vertex formats, associated types and functions.
//!
//! A vertex is a type representing a point. It’s common to find vertex positions, normals, colors
//! or even texture coordinates. However, you’re free to use whichever type you want. Nevertheless,
//! you’re limited to a range of types and dimensions. See `Type` and `Dim` for further details.
//!
//! # `Vertex`
//!
//! ## Rules
//!
//! To be able to use a type as a vertex, you have to implement the `Vertex` trait. That trait
//! represents a mapping between your type and `VertexFormat`. A `VertexFormat` gives runtime hints
//! about your type and restricts the supported type. If you cannot map your type to `VertexFormat`,
//! that means you cannot use it as a `Vertex`.
//!
//! The rule is that your type should have a static size greater than 0 and less than or equal to 4.
//! It should also be either integral, unsigned, floating or boolean. If your type is a complex one
//! – for instance a `struct` – you have to recursively apply that rule to all its fields.
//! For instance, the tuple `(i32, bool)` implements `Vertex` by providing an implementation using
//! the ones of `i32` and `bool`.
//!
//! ## Components list
//!
//! As mentionned above, you can use tuples and structs as `Vertex`. If you look at the definition
//! of `VertexFormat`, you’ll notice that it’s a `Vec<VertexComponentFormat>`. That means simple
//! and primary types map to unit vectors – i.e. their size is 1 – but tuples and structs need
//! several `VertexComponentFormat`s to be represented, hence vectors with sizes greater than 1. No
//! check is made on how many vertex components you’re using – there’s a practical limit, set by the
//! GPU, but it’s not checked (yet).
//!
//! # Generic implementation
//!
//! You have `Vertex` implementations for all the primary types that can be mapped to
//! `VertexFormat`. However, as it’s not possible to automatically implement `Vertex` for your
//! structure (yet?), a type is provided to help you design your vertex type so that you’re
//! automatically provided with a `Vertex` implementation if you use `GTup`.
//!
//! `GTup` is a special type used to represent static heterogeneous list of types. With that in
//! hand, you can easily create `Vertex` types and start using them without even implementing
//! `Vertex`, as long as you use `Vertex` types. Feel free to dig in the `GTup` documentation for
//! further details.
//!
//! If you absolutely want to use your own types – which is legit, you can implement `Vertex` by
//! mapping your inner fields to a tuple or `GTup`, and call the right `Vertex` method on that
//! tuple.

#[cfg(feature = "std")] use std::vec::Vec;
#[cfg(not(feature = "std"))] use alloc::vec::Vec;

use gtup::GTup;

/// A `VertexFormat` is a list of `VertexComponentFormat`s.
pub type VertexFormat = Vec<VertexComponentFormat>;

/// Vertex component format. It gives information on how vertices should be passed to the GPU and
/// optimized in buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VertexComponentFormat {
  /// Type of the component. See `Type` for further details.
  pub comp_type: Type,
  /// Dimension of the component. It should be in 1–4. See `Dim` for further details.
  pub dim: Dim,
  /// Size in bytes that a single element of the component takes. That is, if your component has
  /// a dimension set to 2, then the unit size should be the size of a single element (not two).
  pub unit_size: usize,
  /// Alignment of the component. The best advice is to respect what Rust does, so it’s highly
  /// recommended to use `::std::mem::align_of` to let it does the job for you.
  pub align: usize
}

/// Possible type of vertex components.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
  Integral,
  Unsigned,
  Floating,
  Boolean
}

/// Possible dimension of vertex components.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dim {
  Dim1,
  Dim2,
  Dim3,
  Dim4
}

/// A type that can be used as a `Vertex` has to implement that trait – it must provide a mapping
/// to `VertexFormat`.
///
/// If you’re not sure on how to implement that or if you want to use automatic types, feel free
/// to use the primary supported types and `GTup` or tuples.
pub unsafe trait Vertex {
  fn vertex_format() -> VertexFormat;
}

/// A hint trait to implement to state whether a vertex type is compatible with another.
///
/// If you have two types `V0: Vertex` and `V1: Vertex`, we say that `V1` is compatible with `V0`
/// only if `&V0::vertex_format()[0..V1::vertex_format().len()] == &V1::vertex_format()[..]`. That
/// is, if `V1` is a sub-slice of `V0` starting at 0.
///
/// We node that as `V1: CompatibleVertex<V0>`.
pub unsafe trait CompatibleVertex<V>: Vertex where V: Vertex  {}

unsafe impl<V> CompatibleVertex<V> for V where V: Vertex {}

/// Macro used to implement the `Vertex` trait.
///
/// The first form implements `Vertex` for `$t`. `$q` refers to the inner representation of `$t` –
/// for instance, `[f32; N]` has `f32` as inner representation. `$comp_type` is the type of the
/// component (variant of `Type`) and `$dim` is the dimension (variant of `Dim`).
///
/// The second form implement `Vertex` for `$t` for types that have the same inner representation
/// than theirselves – usually, scalars.
macro_rules! impl_base {
  ($t:ty, $q:ty, $comp_type:ident, $dim:ident) => {
    unsafe impl Vertex for $t {
      fn vertex_format() -> VertexFormat {
        vec![VertexComponentFormat {
          comp_type: Type::$comp_type,
          dim: Dim::$dim,
          unit_size: $crate::vertex::size_of::<$q>(),
          align: $crate::vertex::align_of::<$q>()
        }]
      }
    }
  };

  ($t:ty, $comp_type:ident, $dim:ident) => {
    impl_base!($t, $t, $comp_type, $dim);
  }
}

macro_rules! impl_arr {
  ($t:ty, $q:ident) => {
    impl_base!([$t; 1], $t, $q, Dim1);
    impl_base!([$t; 2], $t, $q, Dim2);
    impl_base!([$t; 3], $t, $q, Dim3);
    impl_base!([$t; 4], $t, $q, Dim4);
  }
}

unsafe impl Vertex for () {
  fn vertex_format() -> VertexFormat {
    Vec::new()
  }
}

// scalars
impl_base!(i8, Integral, Dim1);
impl_base!(i16, Integral, Dim1);
impl_base!(i32, Integral, Dim1);

impl_base!(u8, Unsigned, Dim1);
impl_base!(u16, Unsigned, Dim1);
impl_base!(u32, Unsigned, Dim1);

impl_base!(f32, Floating, Dim1);
impl_base!(f64, Floating, Dim1);

impl_base!(bool, Floating, Dim1);

// arrays
impl_arr!(i8, Integral);
impl_arr!(i16, Integral);
impl_arr!(i32, Integral);

impl_arr!(u8, Unsigned);
impl_arr!(u16, Unsigned);
impl_arr!(u32, Unsigned);

impl_arr!(f32, Floating);
impl_arr!(f64, Floating);

impl_arr!(bool, Boolean);

unsafe impl<A, B> Vertex for GTup<A, B> where A: Vertex, B: Vertex {
  fn vertex_format() -> VertexFormat {
    let mut t = A::vertex_format();
    t.extend(B::vertex_format());
    t
  }
}

macro_rules! impl_vertex_for_tuple {
  ($($t:tt),+) => {
    unsafe impl<$($t),+> Vertex for ($($t),+) where $($t: Vertex),+ {
      fn vertex_format() -> VertexFormat {
        <gtup!(:$($t),+) as Vertex>::vertex_format()
      }
    }
  }
}

impl_vertex_for_tuple!(A, B);
impl_vertex_for_tuple!(A, B, C);
impl_vertex_for_tuple!(A, B, C, D);
impl_vertex_for_tuple!(A, B, C, D, E);
impl_vertex_for_tuple!(A, B, C, D, E, F);
impl_vertex_for_tuple!(A, B, C, D, E, F, G);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_vertex_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

/// A local version of size_of that depends on the state of the std feature.
#[inline(always)]
pub fn size_of<T>() -> usize {
  #[cfg(feature = "std")]
  {
    ::std::mem::size_of::<T>()
  }

  #[cfg(not(feature = "std"))]
  {
    ::core::mem::size_of::<T>()
  }
}

/// A local version of align_of that depends on the state of the std feature.
#[inline(always)]
pub fn align_of<T>() -> usize {
  #[cfg(feature = "std")]
  {
    ::std::mem::align_of::<T>()
  }

  #[cfg(not(feature = "std"))]
  {
    ::core::mem::align_of::<T>()
  }
}
