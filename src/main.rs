use bevy::prelude::*;

mod gen_level_plugin;
pub mod waveform_function;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, gen_level_plugin::GenLevelPlugin))
        .run();
}
