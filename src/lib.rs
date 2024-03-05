use bevy::prelude::*;

mod gen_level_plugin;
pub mod waveform_function;

use wasm_bindgen::prelude::wasm_bindgen;

#[cfg(target_family = "wasm")]
#[wasm_bindgen(inline_js =
    "export function apply_css() {
        let canvas = document.body.querySelector(\"canvas\");
        canvas.style = \"width: 100%; height: 100%;\";
    }
    "
)]
extern "C" {
    fn apply_css();
}

#[wasm_bindgen]
pub fn real_main() {
    let mut app = App::new();
    app
        .add_plugins((
            DefaultPlugins,
            gen_level_plugin::GenLevelPlugin,
        ));
    #[cfg(target_family = "wasm")]
    {
        app.add_systems(Startup, || {
            apply_css();
        });
    }
    app.run();
}
