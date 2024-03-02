use bevy::utils::{smallvec::SmallVec, HashMap};
use tinybitset::TinyBitSet;

#[derive(Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TileId(pub usize);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Side {
    Up,
    Down,
    Left,
    Right,
}

pub struct WaveformFunction {
    total_tiles: usize,
    num_unique_tiles: usize,
    counts: HashMap<TileId, usize>,
    weights: HashMap<(TileId, Side), Vec<(TileId, usize)>>,
}

impl Default for WaveformFunction {
    fn default() -> Self {
        Self::new()
    }
}

impl WaveformFunction {
    pub fn new() -> WaveformFunction {
        WaveformFunction {
            total_tiles: 0,
            num_unique_tiles: 0,
            counts: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    pub fn inc_count_for_tile(&mut self, tile_id: TileId) {
        self.total_tiles += 1;
        let is_unique;
        {
            let counts = &mut self.counts;
            if let Some(x) = counts.get_mut(&tile_id) {
                *x += 1;
                is_unique = false;
            } else {
                counts.insert(tile_id, 1);
                is_unique = true;
            }
        }
        if is_unique {
            self.num_unique_tiles += 1;
        }
    }

    pub fn accum_weight(&mut self, tile_id: TileId, side: Side, side_tile_id: TileId) {
        let weights = &mut self.weights;
        if let Some(x) = weights.get_mut(&(tile_id, side)) {
            let mut found = false;
            for (tile_id_2, weight) in &mut *x {
                if *tile_id_2 == side_tile_id {
                    *weight += 1;
                    found = true;
                }
            }
            if !found {
                x.push((side_tile_id, 1));
            }
        } else {
            let x = vec![(side_tile_id, 1)];
            weights.insert((tile_id, side), x);
        }
    }
}

struct Prng {
    seed: u32,
}

impl Prng {
    pub fn new() -> Prng {
        Prng { seed: 32 }
    }

    pub fn gen(&mut self) -> u32 {
        let mut random = self.seed;
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        self.seed = random;
        random
    }
}

struct MapStates {
    rows: usize,
    cols: usize,
    num_unique_tiles: usize,
    states: Vec<Vec<TinyBitSet<u64, 2>>>,
    prng: Prng,
}

impl MapStates {
    fn new(rows: usize, cols: usize, num_unique_tiles: usize) -> MapStates {
        let mut bitset: TinyBitSet<u64, 2> = TinyBitSet::new();
        for k in 0..num_unique_tiles.min(64 * 2) {
            bitset.assign(k, true);
        }
        let mut states = Vec::with_capacity(rows);
        for _i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _j in 0..cols {
                row.push(bitset);
            }
            states.push(row);
        }
        MapStates {
            rows,
            cols,
            num_unique_tiles,
            states,
            prng: Prng::new(),
        }
    }

    fn reset(&mut self, seed: u32) {
        self.prng.seed = seed;
        let mut bitset: TinyBitSet<u64, 2> = TinyBitSet::new();
        for k in 0..self.num_unique_tiles.min(64 * 2) {
            bitset.assign(k, true);
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.states[i][j] = bitset;
            }
        }
    }

    fn assign_tile(&mut self, x: usize, y: usize, tile_id: TileId) {
        self.states[y][x] = TinyBitSet::new().assigned(tile_id.0, true);
    }

    fn assign_random(&mut self, waveform_function: &WaveformFunction, x: usize, y: usize) {
        let bitset = self.states[y][x];
        if bitset.is_empty() {
            return;
        }
        let possible_tiles = self.possible_tiles_weighted(waveform_function, x, y);
        let mut total_count: f32 = 0.0;
        for &(_, weight) in &possible_tiles {
            total_count += weight;
        }
        let mut random = (((self.prng.gen() as f64) / (u32::MAX as f64)) as f32) * total_count;
        for &(tile_id, weight) in &possible_tiles {
            if weight >= random {
                self.states[y][x] = TinyBitSet::new().assigned(tile_id.0, true);
                break;
            }
            random -= weight;
        }
    }

    fn possible_tiles_on_side_of(
        &self,
        waveform_function: &WaveformFunction,
        x: usize,
        y: usize,
        side: Side,
    ) -> TinyBitSet<u64, 2> {
        let mut result: TinyBitSet<u64, 2> = TinyBitSet::new();
        for tile_id in &self.states[y][x] {
            let tile_id = TileId(tile_id);
            if let Some(weights) = waveform_function.weights.get(&(tile_id, side)) {
                for (tile_id_2, _) in weights {
                    result |=
                        TinyBitSet::<u64, 2>::new().assigned(tile_id_2.0.min(64 * 2 - 1), true);
                }
            }
        }
        result
    }

    fn possible_tiles_weighted(
        &self,
        waveform_function: &WaveformFunction,
        x: usize,
        y: usize,
    ) -> SmallVec<[(TileId, f32); 128]> {
        let mut result = SmallVec::<[(TileId, f32); 128]>::new();
        let mut add_to_result = |x: &SmallVec<[(TileId, f32); 128]>, first: bool| {
            for &(tile_id, weight) in x {
                if first {
                    result.push((tile_id, weight));
                } else if let Some((_, x2)) = result
                    .iter_mut()
                    .find(|(tile_id_2, _)| *tile_id_2 == tile_id)
                {
                    *x2 += weight;
                }
            }
            if !first {
                for i in (0..result.len()).rev() {
                    let tile_id = result[i].0;
                    let found = x.iter().any(|&(tile_id_2, _)| tile_id_2 == tile_id);
                    if !found {
                        result.remove(i);
                    }
                }
            }
        };
        let mut first = true;
        if x > 0 {
            add_to_result(
                &self.possible_tiles_on_side_of_weighted(waveform_function, x - 1, y, Side::Right),
                first,
            );
            first = false;
        }
        if x < self.cols - 1 {
            add_to_result(
                &self.possible_tiles_on_side_of_weighted(waveform_function, x + 1, y, Side::Left),
                first,
            );
            first = false;
        }
        if y > 0 {
            add_to_result(
                &self.possible_tiles_on_side_of_weighted(waveform_function, x, y - 1, Side::Down),
                first,
            );
            first = false;
        }
        if y < self.rows - 1 {
            add_to_result(
                &self.possible_tiles_on_side_of_weighted(waveform_function, x, y + 1, Side::Up),
                first,
            );
        }
        result
    }

    fn possible_tiles_on_side_of_weighted(
        &self,
        waveform_function: &WaveformFunction,
        x: usize,
        y: usize,
        side: Side,
    ) -> SmallVec<[(TileId, f32); 128]> {
        let mut result = SmallVec::<[(TileId, f32); 128]>::new();
        for tile_id in &self.states[y][x] {
            let tile_id = TileId(tile_id);
            let tile_count = *waveform_function.counts.get(&tile_id).unwrap();
            if let Some(weights) = waveform_function.weights.get(&(tile_id, side)) {
                for &(tile_id_2, weight) in weights {
                    let weight = (weight as f32) / (tile_count as f32);
                    if let Some((_, x)) = result
                        .iter_mut()
                        .find(|(tile_id_3, _)| *tile_id_3 == tile_id_2)
                    {
                        *x += weight;
                    } else {
                        result.push((tile_id_2, weight));
                    }
                }
            }
        }
        result
    }

    fn update_possible_tiles(
        &mut self,
        waveform_function: &WaveformFunction,
        x: usize,
        y: usize,
    ) -> bool {
        let mut new_bitset = self.states[y][x];
        if x > 0 {
            let tmp = new_bitset;
            let bitset = self.possible_tiles_on_side_of(waveform_function, x - 1, y, Side::Right);
            new_bitset &= bitset;
            if new_bitset.is_empty() {
                new_bitset = tmp;
            }
        }
        if x < self.cols - 1 {
            let tmp = new_bitset;
            let bitset = self.possible_tiles_on_side_of(waveform_function, x + 1, y, Side::Left);
            new_bitset &= bitset;
            if new_bitset.is_empty() {
                new_bitset = tmp;
            }
        }
        if y > 0 {
            let tmp = new_bitset;
            let bitset = self.possible_tiles_on_side_of(waveform_function, x, y - 1, Side::Down);
            new_bitset &= bitset;
            if new_bitset.is_empty() {
                new_bitset = tmp;
            }
        }
        if y < self.rows - 1 {
            let tmp = new_bitset;
            let bitset = self.possible_tiles_on_side_of(waveform_function, x, y + 1, Side::Up);
            new_bitset &= bitset;
            if new_bitset.is_empty() {
                new_bitset = tmp;
            }
        }
        let changed = self.states[y][x] != new_bitset;
        self.states[y][x] = new_bitset;
        changed
    }

    fn entropy(&self, waveform_function: &WaveformFunction, x: usize, y: usize) -> f32 {
        let mut total: usize = 0;
        for tile_id in &self.states[y][x] {
            let tile_id = TileId(tile_id);
            let tile_count = waveform_function.counts.get(&tile_id).copied().unwrap_or(0);
            total += tile_count;
        }
        let mut result: f32 = 0.0;
        for tile_id in &self.states[y][x] {
            let tile_id = TileId(tile_id);
            let tile_count = waveform_function.counts.get(&tile_id).copied().unwrap_or(0);
            let p: f32 = (tile_count as f32) / (total as f32);
            result -= p * p.ln();
        }
        result
    }
}

pub struct MapGenerator {
    waveform_function: WaveformFunction,
    rows: usize,
    cols: usize,
    preassigned_tiles: HashMap<(usize, usize), TileId>,
    //
    generation_state: MapGenerationState,
}

struct MapGenerationState {
    map_states: MapStates,
    stack: Vec<(usize, usize)>,
    result: Vec<Vec<TileId>>,
}

impl MapGenerationState {
    pub fn new(rows: usize, cols: usize, num_unique_tiles: usize) -> MapGenerationState {
        let mut result: Vec<Vec<TileId>> = Vec::with_capacity(rows);
        for _i in 0..rows {
            let mut row: Vec<TileId> = Vec::with_capacity(cols);
            for _j in 0..cols {
                row.push(TileId(0));
            }
            result.push(row);
        }
        MapGenerationState {
            map_states: MapStates::new(rows, cols, num_unique_tiles),
            stack: Vec::new(),
            result,
        }
    }

    pub fn reset(&mut self, seed: u32) {
        self.map_states.reset(seed);
        self.stack.clear();
        for i in 0..self.result.len() {
            let row = &mut self.result[i];
            for cell in row {
                *cell = TileId(0);
            }
        }
    }
}

impl MapGenerator {
    pub fn new(waveform_function: WaveformFunction, rows: usize, cols: usize) -> MapGenerator {
        let generation_state =
            MapGenerationState::new(rows, cols, waveform_function.num_unique_tiles);
        MapGenerator {
            waveform_function,
            rows,
            cols,
            preassigned_tiles: HashMap::new(),
            generation_state,
        }
    }

    pub fn reset(&mut self, seed: u32) {
        self.preassigned_tiles.clear();
        self.generation_state.reset(seed);
    }

    pub fn with_assigned_tile(mut self, x: usize, y: usize, tile_id: TileId) -> Self {
        self.assign_tile(x, y, tile_id);
        self
    }

    pub fn assign_tile(&mut self, x: usize, y: usize, tile_id: TileId) {
        self.generation_state.result[y][x] = tile_id;
        self.preassigned_tiles.insert((x, y), tile_id);
        self.generation_state.map_states.assign_tile(x, y, tile_id);
        self.generation_state.stack.push((x, y));
    }

    pub fn with_random_tile_at_random_spot(mut self) -> Self {
        self.assign_random_tile_at_random_spot();
        self
    }

    pub fn assign_random_tile_at_random_spot(&mut self) {
        let x = (self.generation_state.map_states.prng.gen() as usize) % self.cols;
        let y = (self.generation_state.map_states.prng.gen() as usize) % self.rows;
        let tile_id = (self.generation_state.map_states.prng.gen() as usize)
            % self.generation_state.map_states.num_unique_tiles;
        let tile_id = TileId(tile_id);
        self.assign_tile(x, y, tile_id);
    }

    pub fn map(&self) -> &Vec<Vec<TileId>> {
        &self.generation_state.result
    }

    pub fn is_assigned(&self, x: usize, y: usize) -> bool {
        self.generation_state.map_states.states[y][x].len() == 1
    }

    pub fn iterate(&mut self) {
        let map_states = &mut self.generation_state.map_states;
        let stack = &mut self.generation_state.stack;
        let result = &mut self.generation_state.result;
        // Propergate constraints
        loop {
            let Some((at_x, at_y)) = stack.pop() else {
                break;
            };
            if at_x > 0 {
                let next_x = at_x - 1;
                let next_y = at_y;
                let changed =
                    map_states.update_possible_tiles(&self.waveform_function, next_x, next_y);
                if changed {
                    stack.push((next_x, next_y));
                }
            }
            if at_x < self.cols - 1 {
                let next_x = at_x + 1;
                let next_y = at_y;
                let changed =
                    map_states.update_possible_tiles(&self.waveform_function, next_x, next_y);
                if changed {
                    stack.push((next_x, next_y));
                }
            }
            if at_y > 0 {
                let next_x = at_x;
                let next_y = at_y - 1;
                let changed =
                    map_states.update_possible_tiles(&self.waveform_function, next_x, next_y);
                if changed {
                    stack.push((next_x, next_y));
                }
            }
            if at_y < self.rows - 1 {
                let next_x = at_x;
                let next_y = at_y + 1;
                let changed =
                    map_states.update_possible_tiles(&self.waveform_function, next_x, next_y);
                if changed {
                    stack.push((next_x, next_y));
                }
            }
        }
        // Find tile coords with lowest entropy
        let mut min_entropy: Option<(f32, (usize, usize))> = None;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if map_states.states[i][j].len() == 1 {
                    continue;
                }
                let entropy = map_states.entropy(&self.waveform_function, j, i);
                if min_entropy.is_none() || entropy < min_entropy.unwrap().0 {
                    min_entropy = Some((entropy, (j, i)));
                }
            }
        }
        // Pick a tile at random for the lowest entropy cell
        if let Some((_, (at_x, at_y))) = min_entropy {
            map_states.assign_random(&self.waveform_function, at_x, at_y);
            stack.push((at_x, at_y));
        }
        for (i, row) in map_states.states.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                let tile_id = TileId(cell.iter().next().unwrap_or(0));
                result[i][j] = tile_id;
            }
        }
    }
}
