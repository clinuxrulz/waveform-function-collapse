use crate::waveform_function::{MapGenerator, TileId, WaveformFunction};
use bevy::{
    asset::LoadState,
    prelude::*,
    utils::{hashbrown::HashSet, HashMap},
    window::PrimaryWindow,
};

#[derive(States, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum PluginState {
    LoadingImage,
    AnalyzeImage,
    GenerateMap,
    NextLevel,
}

#[derive(Resource)]
pub struct Level(usize);

#[derive(Resource)]
pub struct GenerateModelState {
    pub image: Handle<Image>,
}

#[derive(Resource)]
pub struct GenerateModelTextureAtlasLayout {
    pub texture_atlas_layout: Handle<TextureAtlasLayout>,
}

impl std::fmt::Debug for TileId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Resource)]
pub struct GenerateModelMapState {
    pub map: Vec<Vec<TileId>>,
    pub num_rows: usize,
    pub num_cols: usize,
    pub tile_hash_tile_id_map: HashMap<[u8; GEN_MODEL_TILE_SIZE * GEN_MODEL_TILE_SIZE * 4], TileId>,
    pub tile_id_tile_map: HashMap<TileId, Tile>,
    pub seed: u32,
    pub map_generator: MapGenerator,
    pub num_tile_types: usize,
}

pub struct Tile {
    pub location_x: usize,
    pub location_y: usize,
    pub index: usize,
}

#[derive(Resource)]
pub struct GeneratedMap(Vec<Vec<TileId>>);

#[derive(Resource)]
pub struct Tileset {
    pub image: Handle<Image>,
    pub tile_width: u32,
    pub tile_height: u32,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct TilePos {
    row: usize,
    col: usize,
}

#[derive(Component)]
pub struct TileCoordinateText;

#[derive(Component)]
pub struct TileSprite {
    pub row: usize,
    pub col: usize,
    pub tile_id: TileId,
}

#[derive(Component, Event, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    ChangeSeed,
    NextLevel,
}

pub struct GenLevelPlugin;

impl Plugin for GenLevelPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<Action>()
            .insert_resource(Level(0))
            .insert_state(PluginState::LoadingImage)
            // generate model systems
            .add_systems(Startup, startup)
            .add_systems(Update, update.run_if(in_state(PluginState::LoadingImage)))
            .add_systems(
                OnEnter(PluginState::AnalyzeImage),
                generate_model_analyze_image_startup,
            )
            .add_systems(
                Update,
                generate_model_generate_map.run_if(in_state(PluginState::GenerateMap)),
            )
            .add_systems(OnEnter(PluginState::NextLevel), generate_model_next_level)
            .add_systems(Update, button_system);
    }
}

const GEN_MODEL_TILE_SIZE: usize = 16;

const GEN_MODEL_FROM_FILENAMES: [&str; 5] = [
    "super-mario-bros-3-1-1-bg.png",
    "super-mario-bros-3-1-2-bg.png",
    "super-mario-bros-3-1-3-bg.png",
    "super-mario-bros-3-1-5-bg.png",
    "airship.png",
];

const NORMAL_BUTTON: Color = Color::rgb(0.15, 0.15, 0.15);
const HOVERED_BUTTON: Color = Color::rgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::rgb(0.35, 0.75, 0.35);

#[allow(clippy::type_complexity)]
fn button_system(
    mut interaction_query: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &mut BorderColor,
            &Action,
        ),
        (Changed<Interaction>, With<Button>),
    >,
    mut send: EventWriter<Action>,
) {
    for (interaction, mut color, mut border_color, action) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = PRESSED_BUTTON.into();
                border_color.0 = Color::RED;
                send.send(*action);
            }
            Interaction::Hovered => {
                *color = HOVERED_BUTTON.into();
                border_color.0 = Color::WHITE;
            }
            Interaction::None => {
                *color = NORMAL_BUTTON.into();
                border_color.0 = Color::BLACK;
            }
        }
    }
}

fn startup(asset_server: Res<AssetServer>, level: Res<Level>, mut commands: Commands) {
    let filename = GEN_MODEL_FROM_FILENAMES[level.0];
    info!("Loading image \"{filename}\"");
    commands.spawn(Camera2dBundle::default());
    commands.insert_resource(GenerateModelState {
        image: asset_server.load(filename),
    });
    commands
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Row,
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn((ButtonBundle::default(), Action::ChangeSeed))
                .with_children(|parent| {
                    parent.spawn(TextBundle::from_section(
                        "Next Seed (R)",
                        TextStyle {
                            font_size: 20.0,
                            color: Color::rgb(0.9, 0.9, 0.9),
                            ..default()
                        },
                    ));
                });
            parent
                .spawn((
                    ButtonBundle {
                        style: Style {
                            margin: UiRect {
                                left: Val::Px(10.0),
                                ..default()
                            },
                            ..default()
                        },
                        ..default()
                    },
                    Action::NextLevel,
                ))
                .with_children(|parent| {
                    parent.spawn(TextBundle::from_section(
                        "Next Level (N)",
                        TextStyle {
                            font_size: 20.0,
                            color: Color::rgb(0.9, 0.9, 0.9),
                            ..default()
                        },
                    ));
                });
        });
}

fn update(
    asset_server: Res<AssetServer>,
    level: Res<Level>,
    gen_model_state: Res<GenerateModelState>,
    mut app_exit_events: ResMut<Events<bevy::app::AppExit>>,
    mut app_state: ResMut<NextState<PluginState>>,
) {
    let filename = GEN_MODEL_FROM_FILENAMES[level.0];
    let Some(state) = asset_server.get_load_state(&gen_model_state.image) else {
        return;
    };
    match state {
        LoadState::NotLoaded => {}
        LoadState::Loading => {}
        LoadState::Loaded => {
            info!("Image Loaded.");
            app_state.set(PluginState::AnalyzeImage)
        }
        LoadState::Failed => {
            info!("Failed to load image \"{filename}\".");
            app_exit_events.send(bevy::app::AppExit);
        }
    }
}

fn generate_model_analyze_image_startup(
    gen_model_state: Res<GenerateModelState>,
    images: Res<Assets<Image>>,
    mut texture_atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
    mut next_state: ResMut<NextState<PluginState>>,
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let mut window_width: f32 = 0.0;
    let mut window_height: f32 = 0.0;
    for window in &windows {
        window_width = window.width();
        window_height = window.height();
    }
    info!("window {window_width}x{window_height}");
    info!("Analyzing image.");
    let Some(image) = images.get(&gen_model_state.image) else {
        return;
    };
    let image_width = image.width();
    let image_height = image.height();
    info!("image_width: {image_width}");
    info!("image_height: {image_height}");
    let num_cols = (image_width as usize) / GEN_MODEL_TILE_SIZE;
    let num_rows = (image_height as usize) / GEN_MODEL_TILE_SIZE;
    info!("num_cols: {num_cols}");
    info!("num_rows: {num_rows}");
    //
    let layout = TextureAtlasLayout::from_grid(
        Vec2::new(GEN_MODEL_TILE_SIZE as f32, GEN_MODEL_TILE_SIZE as f32),
        num_cols,
        num_rows,
        None,
        None,
    );
    let texture_atlas_layout = texture_atlas_layouts.add(layout);
    commands.insert_resource(GenerateModelTextureAtlasLayout {
        texture_atlas_layout,
    });
    //
    let mut next_id: usize = 0;
    let mut tile_hash_tile_id_map: HashMap<
        [u8; GEN_MODEL_TILE_SIZE * GEN_MODEL_TILE_SIZE * 4],
        TileId,
    > = HashMap::new();
    let mut tile_id_tile_map: HashMap<TileId, Tile> = HashMap::new();
    let mut offset_1 = 0;
    let mut map: Vec<Vec<TileId>> = Vec::with_capacity(num_rows);
    for _i in 0..num_rows {
        let mut row = Vec::with_capacity(num_cols);
        for _j in 0..num_cols {
            row.push(TileId(0));
        }
        map.push(row);
    }
    let mut tile_index: usize = 0;
    let mut tile_hash = [0u8; GEN_MODEL_TILE_SIZE * GEN_MODEL_TILE_SIZE * 4];
    let mut pixel_location_y: usize = 0;
    for map_row in &mut map {
        let mut pixel_location_x: usize = 0;
        let mut offset_2 = offset_1;
        for map_cell in map_row {
            let mut offset_3 = offset_2;
            let mut hash_index: usize = 0;
            for _k in 0..GEN_MODEL_TILE_SIZE {
                let mut offset_4 = offset_3;
                for _l in 0..GEN_MODEL_TILE_SIZE {
                    tile_hash[hash_index] = image.data[offset_4];
                    tile_hash[hash_index + 1] = image.data[offset_4 + 1];
                    tile_hash[hash_index + 2] = image.data[offset_4 + 2];
                    tile_hash[hash_index + 3] = image.data[offset_4 + 3];
                    hash_index += 4;
                    offset_4 += 4;
                }
                offset_3 += (image_width as usize) * 4;
            }
            let tile_id: TileId;
            {
                let tile_id_op = tile_hash_tile_id_map.get(&tile_hash).copied();
                if let Some(tile_id_2) = tile_id_op {
                    tile_id = tile_id_2;
                } else {
                    tile_id = TileId(next_id);
                    next_id += 1;
                    tile_hash_tile_id_map.insert(tile_hash, tile_id);
                    tile_id_tile_map.insert(
                        tile_id,
                        Tile {
                            location_x: pixel_location_x,
                            location_y: pixel_location_y,
                            index: tile_index,
                        },
                    );
                }
            }
            *map_cell = tile_id;
            tile_index += 1;
            offset_2 += GEN_MODEL_TILE_SIZE * 4;
            pixel_location_x += GEN_MODEL_TILE_SIZE;
        }
        offset_1 += GEN_MODEL_TILE_SIZE * (image_width as usize) * 4;
        pixel_location_y += GEN_MODEL_TILE_SIZE;
    }
    /*
    let mut waveform_function_1 = WaveformFunction::new();
    for i in 0..map.len() {
        let row = &map[i];
        for j in 0..row.len() {
            let at_tile_id = map[i][j];
            waveform_function_1.inc_count_for_tile(at_tile_id);
            if j > 1 {
                waveform_function_1.accum_weight(at_tile_id, &[((-1, 0), map[i][j - 1]), ((-2, 0), map[i][j - 2])]);
            }
            if j < row.len() - 2 {
                waveform_function_1.accum_weight(at_tile_id, &[((1, 0), map[i][j + 1]), ((2, 0), map[i][j + 2])]);
            }
            if i > 1 {
                waveform_function_1.accum_weight(at_tile_id, &[((0, -1), map[i - 1][j]), ((0, -2), map[i - 2][j])]);
            }
            if i < map.len() - 2 {
                waveform_function_1.accum_weight(at_tile_id, &[((0, 1), map[i + 1][j]), ((0, 2), map[i + 2][j])]);
            }
        }
    }
    let mut waveform_function_2 = WaveformFunction::new();
    for i in 0..map.len() {
        let row = &map[i];
        for j in 0..row.len() {
            let at_tile_id = map[i][j];
            waveform_function_2.inc_count_for_tile(at_tile_id);
            if j > 0 {
                waveform_function_2.accum_weight(at_tile_id, &[((-1, 0), map[i][j - 1])]);
            }
            if j < row.len() - 1 {
                waveform_function_2.accum_weight(at_tile_id, &[((1, 0), map[i][j + 1])]);
            }
            if i > 0 {
                waveform_function_2.accum_weight(at_tile_id, &[((0, -1), map[i - 1][j])]);
            }
            if i < map.len() - 1 {
                waveform_function_2.accum_weight(at_tile_id, &[((0, 1), map[i + 1][j])]);
            }
        }
    }
    */
    let mut waveform_function_3 = WaveformFunction::new();
    for (i, row) in map.iter().enumerate() {
        for (j, &at_tile_id) in row.iter().enumerate() {
            waveform_function_3.inc_count_for_tile(at_tile_id);
            if j > 0 && i > 0 && j < row.len()-1 && i < map.len() - 1 {
                waveform_function_3.accum_weight(
                    at_tile_id,
                    &([
                        (-1,0),
                        (-1,-1),
                        (0,-1),
                        (1,-1),
                        (1,0),
                        (1,1),
                        (0,1),
                        (-1,1)
                    ] as [(i8,i8);8]).map(|offset| (offset, map[((i as i32) + (offset.1 as i32)) as usize][((j as i32) + (offset.0 as i32)) as usize]))
                );
            }
        }
    }
    info!("{map:?}");
    let map_generator =
        MapGenerator::new(
            vec![/*waveform_function_1,waveform_function_2,*/waveform_function_3],
            (window_height as usize) / GEN_MODEL_TILE_SIZE,
            (window_width as usize) / GEN_MODEL_TILE_SIZE,
        ).with_assigned_random_tiles_from_original_map(&map, 10);
    commands.insert_resource(GenerateModelMapState {
        map,
        num_rows,
        num_cols,
        tile_hash_tile_id_map,
        tile_id_tile_map,
        seed: 1,
        map_generator,
        num_tile_types: next_id,
    });
    next_state.set(PluginState::GenerateMap);
}

#[allow(clippy::too_many_arguments)]
fn generate_model_generate_map(
    generate_model_state: Res<GenerateModelState>,
    mut generate_model_map_state: ResMut<GenerateModelMapState>,
    texture_atlas: Res<GenerateModelTextureAtlasLayout>,
    mut tile_sprites: Query<(
        &mut TileSprite,
        &mut TextureAtlas,
        &mut Transform,
        &mut Visibility,
    )>,
    mut commands: Commands,
    windows: Query<&Window, With<PrimaryWindow>>,
    key: Res<ButtonInput<KeyCode>>,
    mut events: EventReader<Action>,
    mut next_state: ResMut<NextState<PluginState>>,
) {
    generate_model_map_state.map_generator.iterate();
    let mut change_seed = || {
        let seed = generate_model_map_state.seed;
        info!("seed = {seed}");
        generate_model_map_state.seed += 1;
        generate_model_map_state.map_generator.reset(seed);
        let generate_model_map_state = &mut *generate_model_map_state;
        generate_model_map_state.map_generator.assign_random_tiles_from_original_map(&generate_model_map_state.map, 10);
    };
    let mut next_level = || {
        next_state.set(PluginState::NextLevel);
    };
    if key.just_pressed(KeyCode::KeyR) {
        change_seed();
    } else if key.just_pressed(KeyCode::KeyN) {
        next_level();
    }
    for event in events.read() {
        if *event == Action::ChangeSeed {
            change_seed();
        } else if *event == Action::NextLevel {
            next_level();
            return;
        }
    }
    let mut window_width: f32 = 0.0;
    let mut window_height: f32 = 0.0;
    for window in &windows {
        window_width = window.width();
        window_height = window.height();
    }
    let left_x = -0.5 * window_width + 20.0;
    let top_y = 0.5 * window_height - 20.0;
    let map = generate_model_map_state.map_generator.map();
    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    for (mut tile_sprite, mut tile_texture_atlas, mut tile_transform, mut tile_visibility) in
        &mut tile_sprites
    {
        visited.insert((tile_sprite.row, tile_sprite.col));
        let tile_id = map[tile_sprite.row][tile_sprite.col];
        let tile = generate_model_map_state
            .tile_id_tile_map
            .get(&tile_id)
            .unwrap();
        let index = tile.index;
        tile_sprite.tile_id = tile_id;
        tile_texture_atlas.index = index;
        tile_transform.translation.x =
            left_x + (tile_sprite.col as f32) * (GEN_MODEL_TILE_SIZE as f32);
        tile_transform.translation.y =
            top_y - (tile_sprite.row as f32) * (GEN_MODEL_TILE_SIZE as f32);
        *tile_visibility = if generate_model_map_state
            .map_generator
            .is_assigned(tile_sprite.col, tile_sprite.row)
        {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    for (i, row) in map.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            if !visited.contains(&(i, j)) {
                let tile_id = *cell;
                let tile = generate_model_map_state
                    .tile_id_tile_map
                    .get(&tile_id)
                    .unwrap();
                let index = tile.index;
                commands.spawn((
                    TileSprite {
                        row: i,
                        col: j,
                        tile_id,
                    },
                    SpriteSheetBundle {
                        texture: generate_model_state.image.clone(),
                        atlas: TextureAtlas {
                            layout: texture_atlas.texture_atlas_layout.clone(),
                            index,
                        },
                        transform: Transform::from_translation(Vec3::new(
                            left_x + (j as f32) * (GEN_MODEL_TILE_SIZE as f32),
                            top_y - (i as f32) * (GEN_MODEL_TILE_SIZE as f32),
                            0.0,
                        )),
                        visibility: if generate_model_map_state.map_generator.is_assigned(j, i) {
                            Visibility::Inherited
                        } else {
                            Visibility::Hidden
                        },
                        ..default()
                    },
                ));
            }
        }
    }
}

fn generate_model_next_level(
    mut level: ResMut<Level>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<PluginState>>,
    asset_server: Res<AssetServer>,
    tile_sprites: Query<Entity, With<TileSprite>>,
) {
    level.0 = (level.0 + 1) % GEN_MODEL_FROM_FILENAMES.len();
    next_state.set(PluginState::LoadingImage);
    let filename = GEN_MODEL_FROM_FILENAMES[level.0];
    info!("Loading image \"{filename}\"");
    commands.insert_resource(GenerateModelState {
        image: asset_server.load(filename),
    });
    commands.remove_resource::<GenerateModelMapState>();
    commands.remove_resource::<GenerateModelTextureAtlasLayout>();
    commands.remove_resource::<GeneratedMap>();
    for tile_sprite_entity in &tile_sprites {
        commands.entity(tile_sprite_entity).despawn();
    }
}
