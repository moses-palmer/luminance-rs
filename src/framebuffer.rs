//! Framebuffers and utility types and functions.
//!
//! Framebuffers are at the core of rendering. They’re the support of rendering operations and can
//! be used to highly enhance the visual aspect of a render. You’re always provided with at least
//! one framebuffer, `Framebuffer::back_buffer`. That function returns a framebuffer that represents –
//! for short – the current back framebuffer. You can render to that framebuffer and when you
//! *swap* the buffers, your render appears in the front framebuffer (likely your screen).
//!
//! # Framebuffers
//!
//! A framebuffer is an object maintaining the required GPU state to hold images you render to. It
//! gathers two important concepts:
//!
//!   - *Color buffers*.
//!   - *Depth buffers*.
//!
//! The *color buffers* hold the color images you render to. A framebuffer can hold several of them
//! with different color formats. The *depth buffers* hold the depth images you render to.
//! Framebuffers can hold only one depth buffer.
//!
//! # Framebuffer slots
//!
//! A framebuffer slot contains either its color buffers or its depth buffer. Sometimes, you might
//! find it handy to have no slot at all for a given type of buffer. In that case, we use `()`.
//!
//! The slots are a way to convert the different formats you use for your framebuffers’ buffers into
//! their respective texture representation so that you can handle the corresponding texels.
//!
//! Color buffers are abstracted by `ColorSlot` and the depth buffer by `DepthSlot`.

#[cfg(feature = "std")] use std::fmt;
#[cfg(feature = "std")] use std::marker::PhantomData;

#[cfg(not(feature = "std"))] use alloc::vec::Vec;
#[cfg(not(feature = "std"))] use core::fmt;
#[cfg(not(feature = "std"))] use core::marker::PhantomData;

use context::GraphicsContext;
use gtup::GTup;
use metagl::*;
use pixel::{ColorPixel, DepthPixel, PixelFormat, RenderablePixel};
use texture::{Dim2, Dimensionable, Flat, Layerable, RawTexture, Texture, TextureError,
              create_texture, opengl_target};

/// Framebuffer error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FramebufferError {
  TextureError(TextureError),
  Incomplete(IncompleteReason)
}

impl fmt::Display for FramebufferError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      FramebufferError::TextureError(ref e) => {
        write!(f, "framebuffer texture error: {}", e)
      }

      FramebufferError::Incomplete(ref e) => {
        write!(f, "incomplete framebuffer: {}", e)
      }
    }
  }
}

/// Reason a framebuffer is incomplete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IncompleteReason {
  Undefined,
  IncompleteAttachment,
  MissingAttachment,
  IncompleteDrawBuffer,
  IncompleteReadBuffer,
  Unsupported,
  IncompleteMultisample,
  IncompleteLayerTargets,
}

impl fmt::Display for IncompleteReason {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      IncompleteReason::Undefined => write!(f, "incomplete reason"),
      IncompleteReason::IncompleteAttachment => write!(f, "incomplete attachment"),
      IncompleteReason::MissingAttachment => write!(f, "missing attachment"),
      IncompleteReason::IncompleteDrawBuffer => write!(f, "incomplete draw buffer"),
      IncompleteReason::IncompleteReadBuffer => write!(f, "incomplete read buffer"),
      IncompleteReason::Unsupported => write!(f, "unsupported"),
      IncompleteReason::IncompleteMultisample => write!(f, "incomplete multisample"),
      IncompleteReason::IncompleteLayerTargets => write!(f, "incomplete layer targets")
    }
  }
}

/// Framebuffer with static layering, dimension, access and slots formats.
///
/// A `Framebuffer` is a *GPU* special object used to render to. Because framebuffers have a
/// *layering* property, it’s possible to have regular render and *layered rendering*. The dimension
/// of a framebuffer makes it possible to render to 1D, 2D, 3D and cubemaps.
///
/// A framebuffer has two kind of slots:
///
/// - **color slot** ;
/// - **depth slot**.
///
/// A framebuffer can have zero or several color slots and it can have zero or one depth slot. If
/// you use several color slots, you’ll be performing what’s called *MRT* (*M* ultiple *R* ender
/// *T* argets), enabling to render to several textures at once.
#[derive(Debug)]
pub struct Framebuffer<L, D, CS, DS>
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          CS: ColorSlot<L, D>,
          DS: DepthSlot<L, D> {
  handle: GLuint,
  renderbuffer: Option<GLuint>,
  w: u32,
  h: u32,
  color_slot: CS,
  depth_slot: DS,
  _l: PhantomData<L>,
  _d: PhantomData<D>,
}

impl Framebuffer<Flat, Dim2, (), ()> {
  /// Get the back buffer with the given dimension.
  pub fn back_buffer(size: <Dim2 as Dimensionable>::Size) -> Self {
    Framebuffer {
      handle: 0,
      renderbuffer: None,
      w: size[0],
      h: size[1],
      color_slot: (),
      depth_slot: (),
      _l: PhantomData,
      _d: PhantomData,
    }
  }
}

impl<L, D, CS, DS> Drop for Framebuffer<L, D, CS, DS>
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          CS: ColorSlot<L, D>,
          DS: DepthSlot<L, D> {
  fn drop(&mut self) {
    self.destroy();
  }
}

impl<L, D, CS, DS> Framebuffer<L, D, CS, DS>
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          CS: ColorSlot<L, D>,
          DS: DepthSlot<L, D> {
  /// Create a new farmebuffer.
  ///
  /// You’re always handed at least the base level of the texture. If you require any *additional*
  /// levels, you can pass the number via the `mipmaps` parameter.
  pub fn new<C>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize
  ) -> Result<Framebuffer<L, D, CS, DS>, FramebufferError>
  where C: GraphicsContext {
    let mipmaps = mipmaps + 1;
    let mut handle: GLuint = 0;
    let color_formats = CS::color_formats();
    let depth_format = DS::depth_format();
    let target = opengl_target(L::layering(), D::dim());
    let mut textures = vec![0; color_formats.len() + if depth_format.is_some() { 1 } else { 0 }];
    let mut depth_texture: Option<GLuint> = None;
    let mut depth_renderbuffer: Option<GLuint> = None;

    unsafe {
      gl::GenFramebuffers(1, &mut handle);

      ctx.state().borrow_mut().bind_draw_framebuffer(handle);

      // generate all the required textures once; the textures vec will be reduced and dispatched
      // into other containers afterwards (in ColorSlot::reify_textures)
      gl::GenTextures((textures.len()) as GLint, textures.as_mut_ptr());

      // color textures
      if color_formats.is_empty() {
        gl::DrawBuffer(gl::NONE);
      } else {
        for (i, (format, texture)) in color_formats.iter().zip(&textures).enumerate() {
          ctx.state().borrow_mut().bind_texture(target, *texture);
          create_texture::<L, D>(target, size, mipmaps, *format, &Default::default()).map_err(FramebufferError::TextureError)?;
          gl::FramebufferTexture(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0 + i as GLenum, *texture, 0);
        }

        // specify the list of color buffers to draw to
        let color_buf_nb = color_formats.len() as GLsizei;
        let color_buffers: Vec<_> = (gl::COLOR_ATTACHMENT0..gl::COLOR_ATTACHMENT0 + color_buf_nb as GLenum).collect();

        gl::DrawBuffers(color_buf_nb, color_buffers.as_ptr());
      }

      // depth texture, if exists
      if let Some(format) = depth_format {
        let texture = textures.pop().unwrap();

        ctx.state().borrow_mut().bind_texture(target, texture);
        create_texture::<L, D>(target, size, mipmaps, format, &Default::default()).map_err(FramebufferError::TextureError)?;
        gl::FramebufferTexture(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, texture, 0);

        depth_texture = Some(texture);
      } else {
        let mut renderbuffer: GLuint = 0;

        gl::GenRenderbuffers(1, &mut renderbuffer);
        gl::BindRenderbuffer(gl::RENDERBUFFER, renderbuffer);
        gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT32F, D::width(size) as GLsizei, D::height(size) as GLsizei);
        gl::BindRenderbuffer(gl::RENDERBUFFER, 0); // FIXME: see whether really needed

        gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, renderbuffer);

        depth_renderbuffer = Some(renderbuffer);
      }

      ctx.state().borrow_mut().bind_texture(target, 0); // FIXME: see whether really needed

      let framebuffer = Framebuffer {
        handle: handle,
        renderbuffer: depth_renderbuffer,
        w: D::width(size),
        h: D::height(size),
        color_slot: CS::reify_textures(ctx, size, mipmaps, &mut textures.into_iter()),
        depth_slot: DS::reify_texture(ctx, size, mipmaps, depth_texture),
        _l: PhantomData,
        _d: PhantomData,
      };

      match get_status() {
        Ok(_) => {
          ctx.state().borrow_mut().bind_draw_framebuffer(0); // FIXME: see whether really needed

          Ok(framebuffer)
        },
        Err(reason) => {
          ctx.state().borrow_mut().bind_draw_framebuffer(0); // FIXME: see whether really needed

          Self::destroy(&framebuffer);

          Err(FramebufferError::Incomplete(reason))
        }
      }
    }
  }

  // Destroy OpenGL-side stuff.
  fn destroy(&self) {
    unsafe {
      if let Some(renderbuffer) = self.renderbuffer {
        gl::DeleteRenderbuffers(1, &renderbuffer);
      }

      if self.handle != 0 {
        gl::DeleteFramebuffers(1, &self.handle);
      }
    }
  }

  #[inline]
  pub(crate) fn handle(&self) -> GLuint {
    self.handle
  }

  #[inline]
  pub fn width(&self) -> u32 {
    self.w
  }

  #[inline]
  pub fn height(&self) -> u32 {
    self.h
  }

  #[inline]
  pub fn color_slot(&self) -> &CS {
    &self.color_slot
  }

  #[inline]
  pub fn depth_slot(&self) -> &DS {
    &self.depth_slot
  }
}

fn get_status() -> Result<(), IncompleteReason> {
  let status = unsafe { gl::CheckFramebufferStatus(gl::FRAMEBUFFER) };

  match status {
    gl::FRAMEBUFFER_COMPLETE => Ok(()),
    gl::FRAMEBUFFER_UNDEFINED => Err(IncompleteReason::Undefined),
    gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => Err(IncompleteReason::IncompleteAttachment),
    gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => Err(IncompleteReason::MissingAttachment),
    gl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER => Err(IncompleteReason::IncompleteDrawBuffer),
    gl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER => Err(IncompleteReason::IncompleteReadBuffer),
    gl::FRAMEBUFFER_UNSUPPORTED => Err(IncompleteReason::Unsupported),
    gl::FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => Err(IncompleteReason::IncompleteMultisample),
    gl::FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS => Err(IncompleteReason::IncompleteLayerTargets),
    _ => panic!("unknown OpenGL framebuffer incomplete status! status={}", status)
  }
}

/// A framebuffer has a color slot. A color slot can either be empty (the *unit* type is used,`()`)
/// or several color formats.
pub unsafe trait ColorSlot<L, D> where L: Layerable, D: Dimensionable, D::Size: Copy {
  /// Turn a color slot into a list of pixel formats.
  fn color_formats() -> Vec<PixelFormat>;
  /// Reify a list of raw textures into a color slot.
  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self
  where C: GraphicsContext, I: Iterator<Item = GLuint>;
}

unsafe impl<L, D> ColorSlot<L, D> for () where L: Layerable, D: Dimensionable, D::Size: Copy {
  fn color_formats() -> Vec<PixelFormat> { Vec::new() }

  fn reify_textures<C, I>(
    _: &mut C,
    _: D::Size,
    _: usize,
    _: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> { () }
}

unsafe impl<L, D, P> ColorSlot<L, D> for Texture<L, D, P>
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> { vec![P::pixel_format()] }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let color_texture = textures.next().unwrap();

    unsafe {
      let raw = RawTexture::new(ctx.state().clone(), color_texture, opengl_target(L::layering(), D::dim()));
      Texture::from_raw(raw, size, mipmaps)
    }
  }
}

unsafe impl<L, D, P, B> ColorSlot<L, D> for GTup<Texture<L, D, P>, B>
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P: ColorPixel + RenderablePixel,
          B: ColorSlot<L, D> {
  fn color_formats() -> Vec<PixelFormat> {
    let mut a = Texture::<L, D, P>::color_formats();
    a.extend(B::color_formats());
    a
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let a = Texture::<L, D, P>::reify_textures(ctx, size, mipmaps, textures);
    let b = B::reify_textures(ctx, size, mipmaps, textures);

    GTup(a, b)
  }
}

unsafe impl<L, D, P0, P1> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, Texture<L, D, P1>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, b) = GTup::<Texture<L, D, P0>, Texture<L, D, P1>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b)
  }
}

unsafe impl<L, D, P0, P1, P2> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, Texture<L, D, P2>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, c)) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, Texture<L, D, P2>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c)
  }
}

unsafe impl<L, D, P0, P1, P2, P3> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, Texture<L, D, P3>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, d))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, Texture<L, D, P3>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d)
  }
}

unsafe impl<L, D, P0, P1, P2, P3, P4> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>, Texture<L, D, P4>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel,
          P4: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, Texture<L, D, P4>>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, GTup(d, e)))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, Texture<L, D, P4>>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d, e)
  }
}

unsafe impl<L, D, P0, P1, P2, P3, P4, P5> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>, Texture<L, D, P4>, Texture<L, D, P5>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel,
          P4: ColorPixel + RenderablePixel,
          P5: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, Texture<L, D, P5>>>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, GTup(d, GTup(e, f))))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, Texture<L, D, P5>>>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d, e, f)
  }
}

unsafe impl<L, D, P0, P1, P2, P3, P4, P5, P6> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>, Texture<L, D, P4>, Texture<L, D, P5>, Texture<L, D, P6>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel,
          P4: ColorPixel + RenderablePixel,
          P5: ColorPixel + RenderablePixel,
          P6: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, Texture<L, D, P6>>>>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, GTup(d, GTup(e, GTup(f, g)))))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, Texture<L, D, P6>>>>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d, e, f, g)
  }
}

unsafe impl<L, D, P0, P1, P2, P3, P4, P5, P6, P7> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>, Texture<L, D, P4>, Texture<L, D, P5>, Texture<L, D, P6>, Texture<L, D, P7>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel,
          P4: ColorPixel + RenderablePixel,
          P5: ColorPixel + RenderablePixel,
          P6: ColorPixel + RenderablePixel,
          P7: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, GTup<Texture<L, D, P6>, Texture<L, D, P7>>>>>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, GTup(d, GTup(e, GTup(f, GTup(g, h))))))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, GTup<Texture<L, D, P6>, Texture<L, D, P7>>>>>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d, e, f, g, h)
  }
}

unsafe impl<L, D, P0, P1, P2, P3, P4, P5, P6, P7, P8> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>, Texture<L, D, P4>, Texture<L, D, P5>, Texture<L, D, P6>, Texture<L, D, P7>, Texture<L, D, P8>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel,
          P4: ColorPixel + RenderablePixel,
          P5: ColorPixel + RenderablePixel,
          P6: ColorPixel + RenderablePixel,
          P7: ColorPixel + RenderablePixel,
          P8: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, GTup<Texture<L, D, P6>, GTup<Texture<L, D, P7>, Texture<L, D, P8>>>>>>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, GTup(d, GTup(e, GTup(f, GTup(g, GTup(h, i)))))))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, GTup<Texture<L, D, P6>, GTup<Texture<L, D, P7>, Texture< L, D, P8>>>>>>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d, e, f, g, h, i)
  }
}

unsafe impl<L, D, P0, P1, P2, P3, P4, P5, P6, P7, P8, P9> ColorSlot<L, D> for (Texture<L, D, P0>, Texture<L, D, P1>, Texture<L, D, P2>, Texture<L, D, P3>, Texture<L, D, P4>, Texture<L, D, P5>, Texture<L, D, P6>, Texture<L, D, P7>, Texture<L, D, P8>, Texture<L, D, P9>)
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P0: ColorPixel + RenderablePixel,
          P1: ColorPixel + RenderablePixel,
          P2: ColorPixel + RenderablePixel,
          P3: ColorPixel + RenderablePixel,
          P4: ColorPixel + RenderablePixel,
          P5: ColorPixel + RenderablePixel,
          P6: ColorPixel + RenderablePixel,
          P7: ColorPixel + RenderablePixel,
          P8: ColorPixel + RenderablePixel,
          P9: ColorPixel + RenderablePixel {
  fn color_formats() -> Vec<PixelFormat> {
    GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, GTup<Texture<L, D, P6>, GTup<Texture<L, D, P7>, GTup<Texture<L, D, P8>, Texture<L, D, P9>>>>>>>>>>::color_formats()
  }

  fn reify_textures<C, I>(
    ctx: &mut C,
    size: D::Size,
    mipmaps: usize,
    textures: &mut I
  ) -> Self where C: GraphicsContext, I: Iterator<Item = GLuint> {
    let GTup(a, GTup(b, GTup(c, GTup(d, GTup(e, GTup(f, GTup(g, GTup(h, GTup(i, j))))))))) = GTup::<Texture<L, D, P0>, GTup<Texture<L, D, P1>, GTup<Texture<L, D, P2>, GTup<Texture<L, D, P3>, GTup<Texture<L, D, P4>, GTup<Texture<L, D, P5>, GTup<Texture<L, D, P6>, GTup<Texture<L, D, P7>, GTup<Texture<L, D, P8>, Texture<L, D, P9>>>>>>>>>>::reify_textures(ctx, size, mipmaps, textures);
    (a, b, c, d, e, f, g, h, i, j)
  }
}

/// A framebuffer has a depth slot. A depth slot can either be empty (the *unit* type is used, `()`)
/// or a single depth format.
pub unsafe trait DepthSlot<L, D> where L: Layerable, D: Dimensionable, D::Size: Copy {
  /// Turn a depth slot into a pixel format.
  fn depth_format() -> Option<PixelFormat>;
  /// Reify a raw textures into a depth slot.
  fn reify_texture<C, T>(
    ctx: &mut C,
    size: D::Size, 
    mipmaps: usize, 
    texture: T
  ) -> Self where C: GraphicsContext, T: Into<Option<GLuint>>;
}

unsafe impl<L, D> DepthSlot<L, D> for () where L: Layerable, D: Dimensionable, D::Size: Copy {
  fn depth_format() -> Option<PixelFormat> { None }

  fn reify_texture<C, T>(
    _: &mut C,
    _: D::Size, 
    _: usize, 
    _: T
  ) -> Self where C: GraphicsContext, T: Into<Option<GLuint>> {
    ()
  }
}

unsafe impl<L, D, P> DepthSlot<L, D> for Texture<L, D, P>
    where L: Layerable,
          D: Dimensionable,
          D::Size: Copy,
          P: DepthPixel {
  fn depth_format() -> Option<PixelFormat> { Some(P::pixel_format()) }

  fn reify_texture<C, T>(
    ctx: &mut C,
    size: D::Size, 
    mipmaps: usize, 
    texture: T
  ) -> Self where C: GraphicsContext, T: Into<Option<GLuint>> {
    unsafe {
      let raw = RawTexture::new(ctx.state().clone(), texture.into().unwrap(), opengl_target(L::layering(), D::dim()));
      Texture::from_raw(raw, size, mipmaps)
    }
  }
}
