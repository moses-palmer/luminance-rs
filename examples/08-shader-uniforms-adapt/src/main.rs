//! > This program is a sequel to 04-shader-uniforms. Be sure to have read it first.
//!
//! This example shows you how to change the type of a shader program’s interface on the fly without
//! changing the GPU object. This might be wanted whenever you need to use a different type which
//! fields overlap the former type you used, or to implement a form of dynamic introspection. By
//! readapting the uniform interface (to the same type), you can use a *value-driven* approach to
//! add new uniforms on the fly, which comes in very handy when writing UI systems for instance.
//!
//! The program should start black so press space and enjoy.
//!
//! Press the <a>, <s>, <d>, <z> or the arrow keys to move the triangle on the screen.
//! Press the <space> key to switch between uniform interfaces.
//! Press <escape> to quit or close the window.
//!
//! https://docs.rs/luminance

#[macro_use]
extern crate luminance;
extern crate luminance_glfw;

use luminance::framebuffer::Framebuffer;
use luminance::shader::program::Program;
use luminance::tess::{Mode, Tess};
use luminance::render_state::RenderState;
use luminance_glfw::event::{Action, Key, WindowEvent};
use luminance_glfw::surface::{GlfwSurface, Surface, WindowDim, WindowOpt};
use luminance::context::GraphicsContext;
use std::time::Instant;

const VS: &'static str = include_str!("vs.glsl");
const FS: &'static str = include_str!("fs.glsl");

type Vertex = ([f32; 2], [f32; 3]);

// Only one triangle this time.
const TRI_VERTICES: [Vertex; 3] = [
  ([ 0.5, -0.5], [1., 0., 0.]),
  ([ 0.0,  0.5], [0., 1., 0.]),
  ([-0.5, -0.5], [0., 0., 1.]),
];

/// First uniform interface.
uniform_interface! {
  struct ShaderInterface1 {
    #[as("t")]
    time: f32,
    triangle_pos: [f32; 2]
  }
}

/// Second uniform interface.
uniform_interface! {
  struct ShaderInterface2 {
    #[as("t")]
    time: f32,
    triangle_size: f32
  }
}

// Which interface to use?
enum ProgramMode {
  First(Program<Vertex, (), ShaderInterface1>),
  Second(Program<Vertex, (), ShaderInterface2>)
}

impl ProgramMode {
  fn toggle(self) -> Self {
    match self {
      ProgramMode::First(p) => {
        match p.adapt() {
          Ok((x, _)) => ProgramMode::Second(x),
          Err((e, y)) => {
            eprintln!("unable to switch to second uniform interface: {:?}", e);
            ProgramMode::First(y)
          }
        }
      }

      ProgramMode::Second(p) => {
        match p.adapt() {
          Ok((x, _)) => ProgramMode::First(x),
          Err((e, y)) => {
            eprintln!("unable to switch to first uniform interface: {:?}", e);
            ProgramMode::Second(y)
          }
        }
      }
    }
  }
}

fn main() {
  let mut surface = GlfwSurface::new(WindowDim::Windowed(960, 540), "Hello, world!", WindowOpt::default()).expect("GLFW surface creation");

  let mut program = ProgramMode::First(Program::<Vertex, (), ShaderInterface1>::from_strings(None, VS, None, FS).expect("program creation").0);

  let triangle = Tess::new(&mut surface, Mode::Triangle, &TRI_VERTICES[..], None);

  let mut back_buffer = Framebuffer::back_buffer(surface.size());

  let mut triangle_pos = [0., 0.];

  let start_t = Instant::now();

  'app: loop {
    for event in surface.poll_events() {
      match event {
        WindowEvent::Close | WindowEvent::Key(Key::Escape, _, Action::Release, _) => {
          break 'app
        }

        WindowEvent::Key(Key::Space, _, Action::Release, _) => {
          program = program.toggle();
        }

        WindowEvent::Key(Key::A, _, action, _) | WindowEvent::Key(Key::Left, _, action, _) if action == Action::Press || action == Action::Repeat => {
          triangle_pos[0] -= 0.1;
        }

        WindowEvent::Key(Key::D, _, action, _) | WindowEvent::Key(Key::Right, _, action, _) if action == Action::Press || action == Action::Repeat => {
          triangle_pos[0] += 0.1;
        }

        WindowEvent::Key(Key::Z, _, action, _) | WindowEvent::Key(Key::Up, _, action, _) if action == Action::Press || action == Action::Repeat => {
          triangle_pos[1] += 0.1;
        }

        WindowEvent::Key(Key::S, _, action, _) | WindowEvent::Key(Key::Down, _, action, _) if action == Action::Press || action == Action::Repeat => {
          triangle_pos[1] -= 0.1;
        }

        WindowEvent::FramebufferSize(width, height) => {
          back_buffer = Framebuffer::back_buffer([width as u32, height as u32]);
        }

        _ => ()
      }
    }

    let elapsed = start_t.elapsed();
    let t64 = elapsed.as_secs() as f64 + (elapsed.subsec_millis() as f64 * 1e-3);
    let t = t64 as f32;

    surface.pipeline_builder().pipeline(&back_buffer, [0., 0., 0., 0.], |_, shd_gate| {
      match program {
        // if we use the first interface, we just need to pass the time and the triangle position
        ProgramMode::First(ref program) => {
          shd_gate.shade(&program, |rdr_gate, iface| {
            iface.time.update(t);
            iface.triangle_pos.update(triangle_pos);

            rdr_gate.render(RenderState::default(), |tess_gate| {
              tess_gate.render(&mut surface, (&triangle).into());
            });
          });
        }

        // if we use the second interface, we just need to pass the time and we will make the size
        // grow by using the time
        ProgramMode::Second(ref program) => {
          shd_gate.shade(&program, |rdr_gate, iface| {
            iface.time.update(t);
            //iface.triangle_pos.update(triangle_pos); // uncomment this to see a nice error ;)
            iface.triangle_size.update(t.cos().powf(2.));

            rdr_gate.render(RenderState::default(), |tess_gate| {
              tess_gate.render(&mut surface, (&triangle).into());
            });
          });
        }
      }
    });

    surface.swap_buffers();
  }
}
