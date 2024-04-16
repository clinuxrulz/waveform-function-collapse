use std::hash::{Hash, Hasher};

use bevy::utils::{smallvec::SmallVec, AHasher, FixedState, HashMap};
use tinybitset::TinyBitSet;
use bevy::utils::HashSet;

use std::hash::BuildHasher;

use crate::PropergateFn;

#[derive(Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TileId(pub usize);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Neighbour {
    offset: (i8, i8),
    neighbour: TileId,
}

#[derive(Clone)]
pub struct Rule {
    neighbours: SmallVec<[Neighbour;8]>,
    result: TileId,
    weight: f32,
}

pub struct WaveformFunction {
    total_tiles: usize,
    num_unique_tiles: usize,
    counts: HashMap<TileId, usize>,
    rules: Vec<Rule>,
    rule_offsets: HashSet<(i8,i8)>,
    tile_rules_map: HashMap<TileId,Vec<Rule>>,
    original_map: Vec<Vec<TileId>>,
}

impl WaveformFunction {
    pub fn new(original_map: Vec<Vec<TileId>>) -> WaveformFunction {
        WaveformFunction {
            total_tiles: 0,
            num_unique_tiles: 0,
            counts: HashMap::new(),
            rules: Vec::new(),
            rule_offsets: HashSet::new(),
            tile_rules_map: HashMap::new(),
            original_map,
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

    pub fn accum_weight(&mut self, tile_id: TileId, neighbours: &[((i8,i8), TileId)]) {
        let mut neighbours2 = SmallVec::<[Neighbour; 8]>::new();
        for &neighbour in neighbours {
            let neighbour = Neighbour {
                offset: neighbour.0,
                neighbour: neighbour.1,
            };
            neighbours2.push(neighbour);
            self.rule_offsets.insert(neighbour.offset);
        }
        if let Some(x) = self.rules.iter_mut().find(|rule| rule.neighbours == neighbours2 && rule.result == tile_id) {
            x.weight += 1.0;
        } else {
            self.rules.push(Rule {
                neighbours: neighbours2.clone(),
                result: tile_id,
                weight: 1.0,
            });
        }
        if let Some(rules) = self.tile_rules_map.get_mut(&tile_id) {
            if let Some(x) = rules.iter_mut().find(|rule| rule.neighbours == neighbours2 && rule.result == tile_id) {
                x.weight += 1.0;
            } else {
                rules.push(Rule {
                    neighbours: neighbours2,
                    result: tile_id,
                    weight: 1.0,
                });
            }
        } else {
            self.tile_rules_map.insert(
                tile_id,
                vec![Rule {
                    neighbours: neighbours2,
                    result: tile_id,
                    weight: 1.0,
                }]
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

struct SavedState {
    states: Vec<Vec<TinyBitSet<u64, 2>>>,
    current_path: Option<((usize, usize), TileId)>,
}

struct MapStates {
    rows: usize,
    cols: usize,
    num_unique_tiles: usize,
    saved_state_stack: Vec<SavedState>,
    at_deadend: bool,
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
            saved_state_stack: Vec::new(),
            at_deadend: false,
            states,
            prng: Prng::new(),
        }
    }

    fn backtrack(&mut self, waveform_function: &WaveformFunction) -> bool {
        bevy::log::info!("Backtrack");
        if let Some(saved_state) = self.saved_state_stack.pop() {
            self.states = saved_state.states;
            if let Some((location, tile_id)) = saved_state.current_path {
                self.states[location.1][location.0].remove(tile_id.0);
                self.at_deadend = false;
                self.propergate_possible_tiles(location.0, location.1, waveform_function);
            }
            true
        } else {
            false
        }
    }

    fn all_tiles_are_assigned(&self) -> bool {
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.states[i][j].len() != 1 {
                    return false;
                }
            }
        }
        return true;
    }

    fn propergate_2(&mut self, waveform_function: &WaveformFunction, propergate_fn: &PropergateFn) {
        let mut source_map: Vec<Vec<usize>> = Vec::with_capacity(waveform_function.original_map.len());
        for i in 0..waveform_function.original_map.len() {
            let mut source_map_row: Vec<usize> = Vec::with_capacity(waveform_function.original_map[i].len());
            for j in 0..waveform_function.original_map[i].len() {
                source_map_row.push(waveform_function.original_map[i][j].0);
            }
            source_map.push(source_map_row);
        }
        let mut target_map: Vec<Vec<Vec<usize>>> = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut target_map_row: Vec<Vec<usize>> = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                let mut target_map_tiles: Vec<usize> = Vec::with_capacity(waveform_function.num_unique_tiles);
                for k in 0..waveform_function.num_unique_tiles {
                    let has = self.states[i][j].iter().any(|x| x == k);
                    target_map_tiles.push(if has { 1 } else { 0 });
                }
                target_map_row.push(target_map_tiles);
            }
            target_map.push(target_map_row);
        }
        propergate_fn.call(&source_map, &mut target_map);
        let mut hasher = FixedState::default().build_hasher();
        target_map.hash(&mut hasher);
        let mut target_hash = hasher.finish();
        let mut visited_hashes = HashSet::new();
        bevy::log::info!("Start: Propergating on GPU");
        bevy::log::info!("Target hash: {}", target_hash);
        visited_hashes.insert(target_hash);
        loop {
            propergate_fn.call(&source_map, &mut target_map);
            let mut hasher = FixedState::default().build_hasher();
            target_map.hash(&mut hasher);
            let next_hash = hasher.finish();
            if visited_hashes.contains(&next_hash) {
                break;
            }
            visited_hashes.insert(next_hash);
            if next_hash == target_hash {
                break;
            }
            target_hash = next_hash;
            bevy::log::info!("Target hash: {}", target_hash);
        }
        bevy::log::info!("End: Propergating on GPU");
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.states[i][j].len() < 2 {
                    continue;
                }
                let mut bits: TinyBitSet<u64,2> = TinyBitSet::new();
                let mut at_dead_end = true;
                for k in 0..waveform_function.num_unique_tiles {
                    if target_map[i][j][k] != 0 {
                        bits.insert(k);
                        at_dead_end = false;
                    }
                }
                if at_dead_end {
                    self.at_deadend = true;
                }
                self.states[i][j] = bits;
            }
        }
    }

    fn take_random_path_2(&mut self, waveform_function: &WaveformFunction, propergate_fn: &PropergateFn) -> bool {
        // Find tile coords with lowest entropy
        let mut min_entropy: Option<(f32, (usize, usize))> = None;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.states[i][j].len() < 2 {
                    continue;
                }
                let entropy = self.entropy(waveform_function, j, i);
                if min_entropy.is_none() || entropy < min_entropy.unwrap().0 {
                    min_entropy = Some((entropy, (j, i)));
                }
            }
        }
        let Some(min_entropy) = min_entropy else { return false; };
        let Some(tile_id) = self.pick_random(waveform_function, min_entropy.1.0, min_entropy.1.1) else {
            self.at_deadend = true;
            return false;
        };
        bevy::log::info!("Choose tile {} at ({},{})", tile_id.0, min_entropy.1.0, min_entropy.1.1);
        if let Some(saved_state) = self.saved_state_stack.last_mut() {
            saved_state.current_path = Some((min_entropy.1, tile_id));
        }
        self.assign_tile(min_entropy.1.0, min_entropy.1.1, tile_id);
        // Update possible tiles
        self.propergate_2(waveform_function, propergate_fn);
        true
    }

    fn take_random_path(&mut self, waveform_function: &WaveformFunction) -> bool {
        // Find tile coords with lowest entropy
        let mut min_entropy: Option<(f32, (usize, usize))> = None;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.states[i][j].len() < 2 {
                    continue;
                }
                let entropy = self.entropy(waveform_function, j, i);
                if min_entropy.is_none() || entropy < min_entropy.unwrap().0 {
                    min_entropy = Some((entropy, (j, i)));
                }
            }
        }
        let Some(min_entropy) = min_entropy else { return false; };
        let Some(tile_id) = self.pick_random(waveform_function, min_entropy.1.0, min_entropy.1.1) else {
            self.at_deadend = true;
            return false;
        };
        bevy::log::info!("Choose tile {} at ({},{})", tile_id.0, min_entropy.1.0, min_entropy.1.1);
        if let Some(saved_state) = self.saved_state_stack.last_mut() {
            saved_state.current_path = Some((min_entropy.1, tile_id));
        }
        self.assign_tile(min_entropy.1.0, min_entropy.1.1, tile_id);
        // Update possible tiles
        self.propergate_possible_tiles(min_entropy.1.0, min_entropy.1.1, waveform_function);
        true
    }

    fn propergate_possible_tiles(&mut self, start_x: usize, start_y: usize, waveform_function: &WaveformFunction) {
        let mut stack = Vec::<(usize,usize)>::new();
        stack.push((start_x, start_y));
        loop {
            if stack.is_empty() {
                break;
            }
            let (at_x, at_y) = stack.remove(0);
            for &offset in &waveform_function.rule_offsets {
                let next_x = (at_x as i32) - (offset.0 as i32);
                if next_x < 0 || next_x >= self.cols as i32 {
                    continue;
                }
                let next_x = next_x as usize;
                let next_y = (at_y as i32) - (offset.1 as i32);
                if next_y < 0 || next_y >= self.rows as i32 {
                    continue;
                }
                let next_y = next_y as usize;
                let changed =
                    self.update_possible_tiles(waveform_function, next_x, next_y);
                if changed {
                    if self.states[next_y][next_x].is_empty() {
                        self.at_deadend = true;
                        break;
                    }
                    stack.push((next_x, next_y));
                }
            } 
        }
    }

    fn iterate2(&mut self, waveform_function: &WaveformFunction, propergate_fn: &PropergateFn) -> bool {
        bevy::log::info!("Iterate");
        self.saved_state_stack.push(SavedState {
            states: self.states.clone(),
            current_path: None
        });
        let success = self.take_random_path_2(waveform_function, propergate_fn);
        if self.at_deadend {
            while self.at_deadend {
                let success = self.backtrack(waveform_function);
                if !success {
                    bevy::log::info!("Stuck 2");
                    return false;
                }
            }
        }
        true
    }

    fn iterate(&mut self, waveform_function: &WaveformFunction) -> bool {
        bevy::log::info!("Iterate");
        self.saved_state_stack.push(SavedState {
            states: self.states.clone(),
            current_path: None
        });
        let success = self.take_random_path(waveform_function);
        if self.at_deadend {
            while self.at_deadend {
                let success = self.backtrack(waveform_function);
                if !success {
                    bevy::log::info!("Stuck 2");
                    return false;
                }
            }
        }
        true
    }

    fn reset(&mut self, seed: u32) {
        self.at_deadend = false;
        self.saved_state_stack.clear();
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

    fn pick_random(&mut self, waveform_function: &WaveformFunction, x: usize, y: usize) -> Option<TileId> {
        let bitset = self.states[y][x];
        if bitset.is_empty() {
            return None;
        }
        let possible_tiles = self.possible_tiles_weighted(waveform_function, x, y);
        let mut total_count: f32 = 0.0;
        let mut average_weight: f32 = 0.0;
        for &(_, weight) in &possible_tiles {
            average_weight += weight;
        }
        average_weight /= possible_tiles.len() as f32;
        let caos = 0.0f32; // <-- no caos for now
        let apply_caos = |x: f32| {
            x*(1.0-caos) + average_weight*caos
        };
        for &(_, weight) in &possible_tiles {
            total_count += apply_caos(weight);
        }
        let mut random = (((self.prng.gen() as f64) / (u32::MAX as f64)) as f32) * total_count;
        for &(tile_id, weight) in &possible_tiles {
            let weight = apply_caos(weight);
            if weight >= random {
                return Some(tile_id);
            }
            random -= weight;
        }
        possible_tiles.last().map(|x| x.0)
    }

    fn assign_random(&mut self, waveform_function: &WaveformFunction, x: usize, y: usize) {
        let bitset = self.states[y][x];
        if bitset.is_empty() {
            return;
        }
        let possible_tiles = self.possible_tiles_weighted(waveform_function, x, y);
        let mut total_count: f32 = 0.0;
        let mut average_weight: f32 = 0.0;
        for &(_, weight) in &possible_tiles {
            average_weight += weight;
        }
        average_weight /= possible_tiles.len() as f32;
        let caos = 0.0f32; // <-- no caos for now
        let apply_caos = |x: f32| {
            x*(1.0-caos) + average_weight*caos
        };
        for &(_, weight) in &possible_tiles {
            total_count += apply_caos(weight);
        }
        let mut random = (((self.prng.gen() as f64) / (u32::MAX as f64)) as f32) * total_count;
        for &(tile_id, weight) in &possible_tiles {
            let weight = apply_caos(weight);
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
        let mut neighbour_offsets = SmallVec::<[(i8,i8); 8]>::new();
        let mut result = SmallVec::<[(SmallVec<[(i8,i8);8]>,SmallVec::<[(TileId, f32); 128]>); 8]>::new();
        for tile_id in &self.states[y][x] {
            let tile_id = TileId(tile_id);
            let Some(rules) = waveform_function.tile_rules_map.get(&tile_id) else { continue; };
            for rule in rules {
                neighbour_offsets.clear();
                let mut neighbours_match = true;
                for neighbour in &rule.neighbours {
                    neighbour_offsets.push(neighbour.offset);
                    let neighbour_x = (x as i32) + (neighbour.offset.0 as i32);
                    if neighbour_x < 0 || neighbour_x >= self.cols as i32 {
                        //neighbours_match = false;
                        //break;
                        continue;
                    }
                    let neighbour_x = neighbour_x as usize;
                    let neighbour_y = (y as i32) + (neighbour.offset.1 as i32);
                    if neighbour_y < 0 || neighbour_y >= self.rows as i32 {
                        //neighbours_match = false;
                        //break;
                        continue;
                    }
                    let neighbour_y = neighbour_y as usize;
                    if !self.states[neighbour_y][neighbour_x].iter().any(|tile_id| tile_id == neighbour.neighbour.0) {
                        neighbours_match = false;
                        break;
                    }
                }
                if !neighbours_match {
                    continue;
                }
                if let Some((_, x)) = result.iter_mut().find(|(offsets, _)| *offsets == neighbour_offsets) {
                    if let Some((_, weight)) = x.iter_mut().find(|(tile_id, _)| *tile_id == rule.result) {
                        *weight += rule.weight;
                    } else {
                        x.push((rule.result, rule.weight));
                    }
                } else {
                    let mut tmp = SmallVec::<[(TileId, f32); 128]>::new();
                    tmp.push((rule.result, rule.weight));
                    result.push((neighbour_offsets.clone(), tmp));
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
            if tile_id.0 < 64*2 {
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
    waveform_functions: Vec<WaveformFunction>,
    rows: usize,
    cols: usize,
    preassigned_tiles: HashMap<(usize, usize), TileId>,
    //
    generation_state: MapGenerationState,
}

struct MapGenerationState {
    at_wf_idx: usize,
    map_states: MapStates,
    stack: Vec<(usize, usize)>,
    result: Vec<Vec<TileId>>,
    is_stuck: bool,
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
            at_wf_idx: 0,
            map_states: MapStates::new(rows, cols, num_unique_tiles),
            stack: Vec::new(),
            result,
            is_stuck: false,
        }
    }

    pub fn reset(&mut self, seed: u32) {
        self.at_wf_idx = 0;
        self.map_states.reset(seed);
        self.stack.clear();
        for i in 0..self.result.len() {
            let row = &mut self.result[i];
            for cell in row {
                *cell = TileId(0);
            }
        }
        self.is_stuck = false;
    }
}

impl MapGenerator {
    pub fn new(waveform_functions: Vec<WaveformFunction>, rows: usize, cols: usize) -> MapGenerator {
        let generation_state =
            MapGenerationState::new(rows, cols, waveform_functions[0].num_unique_tiles);
        MapGenerator {
            waveform_functions,
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
        //self.generation_state.map_states.propergate_possible_tiles(x, y, &self.waveform_functions[0]);
    }

    pub fn init_propergate(&mut self) {
        let stack = &mut self.generation_state.stack;
        let map_states = &mut self.generation_state.map_states;
        while let Some((x,y)) = stack.pop() {
            map_states.propergate_possible_tiles(x, y, &self.waveform_functions[0]);
        }
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

    pub fn with_assigned_random_tiles_from_original_map(mut self, map: &Vec<Vec<TileId>>, quantity: usize) -> Self {
        self.assign_random_tiles_from_original_map(map, quantity);
        self
    }

    pub fn assign_random_tiles_from_original_map(&mut self, map: &Vec<Vec<TileId>>, quantity: usize) {
        let offset_x: usize;
        if self.cols < map[0].len() {
            offset_x = (self.generation_state.map_states.prng.gen() % ((map[0].len() - self.cols) as u32)) as usize;
        } else {
            offset_x = 0;
        }
        let offset_y: usize;
        if self.rows < map.len() {
            offset_y = (self.generation_state.map_states.prng.gen() % ((map.len() - self.rows) as u32)) as usize;
        } else {
            offset_y = 0;
        }
        for _i in 0..quantity {
            let x = (self.generation_state.map_states.prng.gen() % (self.cols.min(map[0].len()) as u32)) as usize;
            let y = (self.generation_state.map_states.prng.gen() % (self.rows.min(map.len()) as u32)) as usize;
            let t = map[offset_y + y][offset_x + x];
            if t.0 >= 64*2 {
                continue;
            }

            self.assign_tile(x, y, t);
        }
    }

    pub fn map(&self) -> &Vec<Vec<TileId>> {
        &self.generation_state.result
    }

    pub fn is_assigned(&self, x: usize, y: usize) -> bool {
        self.generation_state.map_states.states[y][x].len() == 1
    }

    pub fn iterate(&mut self, propergate_fn: Option<&PropergateFn>) -> bool {
        if !self.generation_state.map_states.all_tiles_are_assigned() {
            if let Some(propergate_fn) = propergate_fn {
                let at_wf_idx = &mut self.generation_state.at_wf_idx;
                let success = self.generation_state.map_states.iterate2(&self.waveform_functions[*at_wf_idx], propergate_fn);
                if !success {
                    if *at_wf_idx < self.waveform_functions.len()-1 {
                        *at_wf_idx += 1;
                    } else {
                        return false;
                    }
                }
                for (i, row) in self.generation_state.map_states.states.iter().enumerate() {
                    for (j, cell) in row.iter().enumerate() {
                        let tile_id = TileId(cell.iter().next().unwrap_or(0));
                        self.generation_state.result[i][j] = tile_id;
                    }
                }
                return true;
            } else {
                let at_wf_idx = &mut self.generation_state.at_wf_idx;
                let success = self.generation_state.map_states.iterate(&self.waveform_functions[*at_wf_idx]);
                if !success {
                    if *at_wf_idx < self.waveform_functions.len()-1 {
                        *at_wf_idx += 1;
                    } else {
                        return false;
                    }
                }
                for (i, row) in self.generation_state.map_states.states.iter().enumerate() {
                    for (j, cell) in row.iter().enumerate() {
                        let tile_id = TileId(cell.iter().next().unwrap_or(0));
                        self.generation_state.result[i][j] = tile_id;
                    }
                }
                return true;
            }
        } else {
            return false;
        }
        /*
        let map_states = &mut self.generation_state.map_states;
        let stack = &mut self.generation_state.stack;
        let result = &mut self.generation_state.result;
        let at_wf_idx = &mut self.generation_state.at_wf_idx;
        let is_stuck = &mut self.generation_state.is_stuck;
        if stack.is_empty() && *is_stuck && *at_wf_idx < self.waveform_functions.len()-1 {
            *is_stuck = false;
            *at_wf_idx += 1;
            bevy::log::info!("Next waveform function.");
            let mut bitset = TinyBitSet::<u64,2>::new();
            for i in 0..map_states.num_unique_tiles.min(64*2) {
                bitset.insert(i);
            }
            for i in 0..self.rows {
                for j in 0..self.cols {
                    if map_states.states[i][j].len() != 1 {
                        map_states.states[i][j] = bitset;
                    } else {
                        stack.push((j,i));
                    }
                }
            }
        }
        let waveform_function = &self.waveform_functions[*at_wf_idx];
        // Propergate constraints
        loop {
            if stack.is_empty() {
                break;
            }
            let (at_x, at_y) = stack.remove(0);
            for &offset in &waveform_function.rule_offsets {
                let next_x = (at_x as i32) - (offset.0 as i32);
                if next_x < 0 || next_x >= self.cols as i32 {
                    continue;
                }
                let next_x = next_x as usize;
                let next_y = (at_y as i32) - (offset.1 as i32);
                if next_y < 0 || next_y >= self.rows as i32 {
                    continue;
                }
                let next_y = next_y as usize;
                let changed =
                    map_states.update_possible_tiles(waveform_function, next_x, next_y);
                if changed {
                    stack.push((next_x, next_y));
                }
            } 
        }
        // Find tile coords with lowest entropy
        let mut min_entropy: Option<(f32, (usize, usize))> = None;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if map_states.states[i][j].len() < 2 {
                    continue;
                }
                let entropy = map_states.entropy(waveform_function, j, i);
                if min_entropy.is_none() || entropy < min_entropy.unwrap().0 {
                    min_entropy = Some((entropy, (j, i)));
                }
            }
        }
        // Pick a tile at random for the lowest entropy cell
        if let Some((_, (at_x, at_y))) = min_entropy {
            map_states.assign_random(waveform_function, at_x, at_y);
            stack.push((at_x, at_y));
        } else {
            *is_stuck = true;
        }
        for (i, row) in map_states.states.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                let tile_id = TileId(cell.iter().next().unwrap_or(0));
                result[i][j] = tile_id;
            }
        }*/
    }
}
