use bevy::utils::{smallvec::SmallVec, HashMap};
use tinybitset::TinyBitSet;
use bevy::utils::HashSet;

#[derive(Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TileId(pub usize);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Side {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct RuleKey {
    neighbour_offset: (i8, i8),
    neighbour: TileId,
}

#[derive(Clone, Copy)]
pub struct Rule {
    result: TileId,
    weight: f32,
    optional: bool,
}

pub struct WaveformFunction {
    total_tiles: usize,
    num_unique_tiles: usize,
    counts: HashMap<TileId, usize>,
    rules: HashMap<RuleKey,Vec<Rule>>,
    rule_offsets: HashSet<(i8,i8)>,
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
            rules: HashMap::new(),
            rule_offsets: HashSet::new(),
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

    pub fn accum_weight(&mut self, tile_id: TileId, neighbour_offset: (i8, i8), neighbour: TileId, optional: bool) {
        self.rule_offsets.insert(neighbour_offset);
        let rule_key = RuleKey {
            neighbour_offset,
            neighbour,
        };
        if let Some(x) = self.rules.get_mut(&rule_key) {
            if let Some(y) = x.iter_mut().find(|rule| rule.result == tile_id) {
                y.weight += 1.0;
            } else {
                x.push(Rule {
                    result: tile_id,
                    weight: 1.0,
                    optional: false,
                });
            }
        } else {
            self.rules.insert(
                rule_key,
                vec![Rule {
                    result: tile_id,
                    weight: 1.0,
                    optional: false,
                }],
            );
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
        self.states[y][x] = TinyBitSet::new().assigned(tile_id.0.min(64*2-1), true);
    }

    fn assign_random(&mut self, waveform_function: &WaveformFunction, x: usize, y: usize) {
        let bitset = self.states[y][x];
        if bitset.is_empty() {
            return;
        }
        let possible_tiles = self.possible_tiles_weighted(waveform_function, x, y);
        let mut total_count: f32 = 0.0;
        let caos = 0.95f32;
        let apply_caos = |x: f32| {
            x*(1.0-caos)+1.0*caos
        };
        for &(tile_id, weight) in &possible_tiles {
            total_count += apply_caos(*waveform_function.counts.get(&tile_id).unwrap() as f32);
        }
        let mut random = (((self.prng.gen() as f64) / (u32::MAX as f64)) as f32) * total_count;
        for &(tile_id, weight) in &possible_tiles {
            let weight = apply_caos(*waveform_function.counts.get(&tile_id).unwrap() as f32);
            if weight >= random {
                self.states[y][x] = TinyBitSet::new().assigned(tile_id.0.min(64*2-1), true);
                break;
            }
            random -= weight;
        }
    }

    fn possible_tiles_weighted(
        &self,
        waveform_function: &WaveformFunction,
        x: usize,
        y: usize,
    ) -> SmallVec<[(TileId, f32); 128]> {
        let mut result = SmallVec::<[((i8, i8), SmallVec::<[(TileId, f32); 128]>); 4]>::new();
        for neighbour_offset in &waveform_function.rule_offsets {
            let mut neighbour_x = (x as i32) + (neighbour_offset.0 as i32);
            if neighbour_x < 0 || neighbour_x >= self.cols as i32 {
                continue;
            }
            if neighbour_x < 0 {
                neighbour_x += self.cols as i32;
            }
            if neighbour_x >= self.cols as i32 {
                neighbour_x -= self.cols as i32;
            }
            let neighbour_x = neighbour_x as usize;
            let mut neighbour_y = (y as i32) + (neighbour_offset.1 as i32);
            if neighbour_y < 0 || neighbour_y >= self.rows as i32 {
                continue;
            }
            if neighbour_y < 0 {
                neighbour_y += self.rows as i32;
            }
            if neighbour_y >= self.rows as i32 {
                neighbour_y -= self.rows as i32;
            }
            let neighbour_y = neighbour_y as usize;
            let possible_neighbours = self.states[neighbour_y][neighbour_x];
            let mut rules = SmallVec::<[Rule; 100]>::new();
            for neighbour in &possible_neighbours {
                let rule_key = RuleKey {
                    neighbour_offset: *neighbour_offset,
                    neighbour: TileId(neighbour),
                };
                if let Some(rules2) = waveform_function.rules.get(&rule_key) {
                    for &rule in rules2 {
                        rules.push(rule);
                    }
                }
            }
            for rule in rules {
                if let Some((_, result)) = result.iter_mut().find(|(offset,_)| *offset == *neighbour_offset) {
                    if let Some((_, weight)) = result.iter_mut().find(|(tile_id, _)| *tile_id == rule.result) {
                        *weight += rule.weight;
                    } else {
                        result.push((rule.result, rule.weight));
                    }
                } else {
                    let mut tmp = SmallVec::<[(TileId,f32);128]>::new();
                    tmp.push((rule.result, rule.weight));
                    result.push((*neighbour_offset, tmp));
                }
            }
        }
        let mut result2 = SmallVec::<[(TileId,f32); 128]>::new();
        if result.is_empty() {
            return result2;
        }
        for &(tile_id, weight) in &result[0].1 {
            result2.push((tile_id, weight));
        }
        for i in 1..result.len() {
            let tiles_weighted = &result[i].1;
            for j in (0..result2.len()).rev() {
                let tile_id = result2[j].0;
                if !tiles_weighted.iter().any(|(tile_id2,_)| *tile_id2 == tile_id) {
                    result2.remove(j);
                }
            }
            for &(tile_id, weight) in tiles_weighted {
                if let Some((_, weight2)) = result2.iter_mut().find(|(tile_id2,_)| *tile_id2 == tile_id) {
                    *weight2 += weight;
                }
            }
        }
        result2
    }

    fn update_possible_tiles(
        &mut self,
        waveform_function: &WaveformFunction,
        x: usize,
        y: usize,
    ) -> bool {
        if self.states[y][x].len() < 2 {
            return false;
        }
        let mut new_bitset = self.states[y][x];
        let mut mask = TinyBitSet::<u64,2>::new();
        for (tile_id, _) in self.possible_tiles_weighted(waveform_function, x, y) {
            if (tile_id.0 < 64*2) {
            mask.insert(tile_id.0);
            }
        }
        new_bitset &= mask;
        let changed = self.states[y][x] != new_bitset;
        self.states[y][x] = new_bitset;
        changed
    }

    fn entropy(&self, waveform_function: &WaveformFunction, x: usize, y: usize) -> f32 {
        let mut result: f32 = 0.0;
        for tile_id in &self.states[y][x] {
            let tile_id = TileId(tile_id);
            let tile_count = waveform_function.counts.get(&tile_id).copied().unwrap_or(0);
            let p: f32 = (tile_count as f32) / ((self.rows * self.cols) as f32);
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
            if stack.is_empty() {
                break;
            }
            let (at_x, at_y) = stack.remove(0);
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
