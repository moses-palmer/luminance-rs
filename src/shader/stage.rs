use gl;
use gl::types::*;
use std::error::Error;
use std::ffi::CString;
use std::fmt;
use std::ptr::{null, null_mut};

/// A shader stage type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Type {
  TessellationControlShader,
  TessellationEvaluationShader,
  VertexShader,
  GeometryShader,
  FragmentShader
}

impl fmt::Display for Type {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      Type::TessellationControlShader => f.write_str("tessellation control shader"),
      Type::TessellationEvaluationShader => f.write_str("tessellation evaluation shader"),
      Type::VertexShader => f.write_str("vertex shader"),
      Type::GeometryShader => f.write_str("geometry shader"),
      Type::FragmentShader => f.write_str("fragment shader")
    }
  }
}

/// A shader stage.
#[derive(Debug)]
pub struct Stage {
  handle: GLuint,
  ty: Type
}

impl Stage {
  /// Create a new shader stage.
  pub fn new(ty: Type, src: &str) -> Result<Self, StageError> {
    unsafe {
      let src = CString::new(glsl_pragma_src(src).as_bytes()).unwrap();
      let handle = gl::CreateShader(opengl_shader_type(ty));

      if handle == 0 {
        return Err(StageError::CompilationFailed(ty, String::from("unable to create shader stage")));
      }

      gl::ShaderSource(handle, 1, [src.as_ptr()].as_ptr(), null());
      gl::CompileShader(handle);

      let mut compiled: GLint = gl::FALSE as GLint;
      gl::GetShaderiv(handle, gl::COMPILE_STATUS, &mut compiled);

      if compiled == (gl::TRUE as GLint) {
        Ok(Stage {
          handle: handle,
          ty: ty
        })
      } else {
        let mut log_len: GLint = 0;
        gl::GetShaderiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

        let mut log: Vec<u8> = Vec::with_capacity(log_len as usize);
        gl::GetShaderInfoLog(handle, log_len, null_mut(), log.as_mut_ptr() as *mut GLchar);

        gl::DeleteShader(handle);

        log.set_len(log_len as usize);

        Err(StageError::CompilationFailed(ty, String::from_utf8(log).unwrap()))
      }
    }
  }

  #[inline]
  pub(crate) fn handle(&self) -> GLuint {
    self.handle
  }
}

impl Drop for Stage {
  fn drop(&mut self) {
    unsafe { gl::DeleteShader(self.handle) }
  }
}

/// Errors that shader stages can emit.
#[derive(Clone, Debug)]
pub enum StageError {
  /// Occurs when a shader fails to compile.
  CompilationFailed(Type, String),
  /// Occurs when you try to create a shader which type is not supported on the current hardware.
  UnsupportedType(Type)
}

impl fmt::Display for StageError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StageError::CompilationFailed(ref ty, ref r) => {
        write!(f, "{} compilation error: {}", ty, r)
      }

      StageError::UnsupportedType(ty) => {
        write!(f, "unsupported {}", ty)
      }
    }
  }
}

impl Error for StageError {}

fn glsl_pragma_src(src: &str) -> String {
  let mut pragma = String::from(GLSL_PRAGMA);
  pragma.push_str(src);
  pragma
}

const GLSL_PRAGMA: &'static str = "\
#version 330 core\n\
#extension GL_ARB_separate_shader_objects : require\n";

fn opengl_shader_type(t: Type) -> GLenum {
  match t {
    Type::TessellationControlShader => gl::TESS_CONTROL_SHADER,
    Type::TessellationEvaluationShader => gl::TESS_EVALUATION_SHADER,
    Type::VertexShader => gl::VERTEX_SHADER,
    Type::GeometryShader => gl::GEOMETRY_SHADER,
    Type::FragmentShader => gl::FRAGMENT_SHADER
  }
}
