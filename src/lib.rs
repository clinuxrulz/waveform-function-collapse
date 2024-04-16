use bevy::{asset::AssetMetaCheck, prelude::*, render::extract_resource::ExtractResource};

mod gen_level_plugin;
pub mod waveform_function;

use wasm_bindgen::{convert::FromWasmAbi, prelude::wasm_bindgen, JsValue};
use waveform_function::TileId;

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

#[derive(Resource)]
pub struct MakePropergateFn {
    make_propergate_fn: js_sys::Function,
}

#[derive(Resource)]
pub struct PropergateFn {
    propergate_fn: js_sys::Function,
}

unsafe impl Send for MakePropergateFn {}
unsafe impl Sync for MakePropergateFn {}

unsafe impl Send for PropergateFn {}
unsafe impl Sync for PropergateFn {}

impl MakePropergateFn {
    pub fn call(&self, source_map_rows: usize, source_map_cols: usize, target_map_rows: usize, target_map_columns: usize, num_unique_tiles: usize) -> PropergateFn {
        let propergate_fn = self.make_propergate_fn.bind3(
            &wasm_bindgen::JsValue::null(),
            &source_map_rows.into(),
            &source_map_cols.into(),
            &target_map_rows.into()
        ).call2(
            &wasm_bindgen::JsValue::null(),
            &target_map_columns.into(),
            &num_unique_tiles.into()
        ).unwrap();
        let propergate_fn: js_sys::Function = propergate_fn.into();
        PropergateFn { propergate_fn }
    }
}

impl PropergateFn {
    pub fn call(&self, source_map: &Vec<Vec<usize>>, target_map: &mut Vec<Vec<Vec<usize>>>) {
        let result = self.propergate_fn.call2(
            &JsValue::null(),
            &serde_wasm_bindgen::to_value(source_map).unwrap(),
            &serde_wasm_bindgen::to_value(target_map).unwrap()
        ).unwrap();
        let result: Vec<Vec<Vec<usize>>> = serde_wasm_bindgen::from_value(result).unwrap();
        *target_map = result;
    }
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn real_main_with_make_propergate_fn(make_propergate_fn: js_sys::Function) {
    let make_propergate_fn: MakePropergateFn = MakePropergateFn { make_propergate_fn };
    let mut app = App::new();
    app
        .insert_resource(AssetMetaCheck::Never)
        .add_plugins((
            DefaultPlugins,
            gen_level_plugin::GenLevelPlugin,
        ));
    app.add_systems(Startup, || {
        apply_css();
    });
    app.insert_resource(make_propergate_fn);
    app.run();
}
