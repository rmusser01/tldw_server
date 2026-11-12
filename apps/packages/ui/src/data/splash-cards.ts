import type { SplashCard } from "../components/Common/SplashScreen/engine/types";

/**
 * Canonical source-fidelity cards mirrored from tldw_chatbook
 * Utils/Splash_Screens/card_definitions.py (name/effect order preserved).
 */
export const SOURCE_CANONICAL_SPLASH_CARDS: SplashCard[] = [
  { name: "default", effect: null, asciiArt: "default_splash" },
  { name: "matrix", effect: "matrix_rain", title: "tldw chatbook", duration: 3000 },
  { name: "glitch", effect: "glitch", asciiArt: "default_splash", effectConfig: { glitch_chars: "!@#$%^&*()_+-=[]{}|;:,.<>?" } },
  { name: "retro", effect: "retro_terminal", asciiArt: "default_splash" },
  { name: "tech_pulse", effect: "pulse", asciiArt: "tech_pulse", effectConfig: { color: [100, 180, 255] } },
  { name: "code_scroll", effect: "code_scroll", title: "TLDW CHATBOOK" },
  { name: "minimal_fade", effect: "typewriter", asciiArt: "minimal_fade" },
  { name: "blueprint", effect: null, asciiArt: "blueprint" },
  { name: "arcade_high_score", effect: "blink", asciiArt: "arcade_high_score", effectConfig: { blink_targets: ["LOADING...", "PRESS ANY KEY TO START!"] } },
  { name: "digital_rain", effect: "digital_rain", title: "TLDW CHATBOOK v2.0" },
  { name: "loading_bar", effect: "loading_bar", asciiArt: "loading_bar_frame", effectConfig: { fill_char: "\u2588", text_above: "SYSTEM INITIALIZATION SEQUENCE" } },
  { name: "starfield", effect: "starfield", title: "Hyperdrive Initializing...", effectConfig: { num_stars: 200, warp_factor: 0.25 } },
  {
    name: "terminal_boot",
    effect: "terminal_boot",
    effectConfig: {
      boot_sequence: [
        { text: "TLDW BIOS v4.2.1 initializing...", typeSpeed: 0.02, pauseAfter: 300, style: "#ffffff" },
        { text: "Memory Test: 65536 KB OK", typeSpeed: 0.01, pauseAfter: 200, style: "#00cc00" },
        { text: "Detecting CPU Type: Quantum Entangled Processor", typeSpeed: 0.02, pauseAfter: 100, style: "#00cc00" },
        { text: "Initializing USB Controllers ... Done.", typeSpeed: 0.015, pauseAfter: 200, style: "#00cc00" },
        { text: "Loading TL-DOS...", typeSpeed: 0.04, pauseAfter: 300, style: "#cccc00" },
        { text: "Starting services:", typeSpeed: 0.02, pauseAfter: 100, style: "#00cc00" },
        { text: "  Network Stack .............. [OK]", typeSpeed: 0.01, style: "#006600" },
        { text: "  AI Core Diagnostics ........ [OK]", typeSpeed: 0.01, style: "#006600" },
        { text: "  Sarcasm Module ............. [ENABLED]", typeSpeed: 0.01, style: "#006600" },
        { text: "Welcome to TLDW Chatbook", typeSpeed: 0.03, style: "#00cccc" },
      ],
    },
  },
  { name: "glitch_reveal", effect: "glitch_reveal", asciiArt: "app_logo_clear", effectConfig: { glitch_chars: "!@#$%^&*\u2593\u2592\u2591\u2588", start_intensity: 0.9 } },
  { name: "ascii_morph", effect: "ascii_morph", effectConfig: { start_art_name: "morph_art_start", end_art_name: "morph_art_end" } },
  { name: "game_of_life", effect: "game_of_life", title: "Cellular Automata Initializing..." },
  {
    name: "scrolling_credits",
    effect: "scrolling_credits",
    title: "TLDW Chatbook",
    effectConfig: {
      credits_list: [
        { role: "Lead Developer", name: "Jules AI" },
        { role: "ASCII Art Design", name: "The Byte Smiths" },
        { role: "Animation Engine", name: "Temporal Mechanics Inc." },
        { line: "" },
        { line: "Special Thanks To:" },
        { line: "All the Electrons" },
        { line: "The Coffee Machine" },
      ],
    },
  },
  { name: "spotlight_reveal", effect: "spotlight", asciiArt: "spotlight_background", effectConfig: { spotlight_radius: 7, path_type: "lissajous" } },
  { name: "sound_bars", effect: "sound_bars", title: "Frequency Analysis Engaged", effectConfig: { num_bars: 25 } },
  { name: "raindrops_pond", effect: "raindrops", title: "TLDW Reflections", effectConfig: { spawn_rate: 2.0, max_concurrent_ripples: 20 } },
  { name: "pixel_zoom", effect: "pixel_zoom", asciiArt: "pixel_art_target", effectConfig: { max_pixel_size: 10 } },
  { name: "text_explosion", effect: "text_explosion", effectConfig: { text_to_animate: "T . L . D . W", effect_direction: "implode", particle_spread: 40 } },
  { name: "old_film", effect: "old_film", effectConfig: { frames_art_names: ["film_generic_frame"], shake_intensity: 1, grain_density: 0.07 } },
  { name: "maze_generator", effect: "maze_generator", title: "Constructing Reality Tunnels..." },
  { name: "dwarf_fortress", effect: "mining", asciiArt: "dwarf_fortress" },
  { name: "neural_network", effect: "neural_network", title: "TLDW Chatbook" },
  { name: "quantum_particles", effect: "quantum_particles", title: "TLDW Chatbook", subtitle: "Quantum Computing Interface" },
  { name: "ascii_wave", effect: "ascii_wave", title: "TLDW Chatbook", subtitle: "Riding the Wave of AI" },
  { name: "binary_matrix", effect: "binary_matrix", title: "TLDW" },
  { name: "constellation_map", effect: "constellation_map", title: "TLDW Chatbook" },
  { name: "typewriter_news", effect: "typewriter_news" },
  { name: "dna_sequence", effect: "dna_sequence", title: "TLDW Chatbook" },
  { name: "circuit_trace", effect: "circuit_trace", title: "TLDW Chatbook" },
  { name: "plasma_field", effect: "plasma_field", title: "TLDW Chatbook" },
  { name: "ascii_fire", effect: "ascii_fire", title: "TLDW Chatbook" },
  { name: "rubiks_cube", effect: "rubiks_cube", title: "TLDW" },
  { name: "data_stream", effect: "data_stream", title: "TLDW Chatbook" },
  { name: "fractal_zoom", effect: "fractal_zoom", title: "TLDW Chatbook" },
  { name: "ascii_spinner", effect: "ascii_spinner", title: "Loading TLDW Chatbook" },
  { name: "hacker_terminal", effect: "hacker_terminal", title: "TLDW Chatbook" },
  { name: "cyberpunk_glitch", effect: "cyberpunk_glitch", title: "tldw chatbook" },
  { name: "ascii_mandala", effect: "ascii_mandala", title: "tldw chatbook" },
  { name: "holographic_interface", effect: "holographic_interface", title: "tldw chatbook" },
  { name: "quantum_tunnel", effect: "quantum_tunnel", title: "tldw chatbook" },
  { name: "chaotic_typewriter", effect: "chaotic_typewriter", title: "tldw chatbook" },
  { name: "spy_vs_spy", effect: "spy_vs_spy" },
  { name: "phonebooths", effect: "phonebooths_dialing" },
  { name: "emoji_face", effect: "emoji_face" },
  { name: "custom_image", effect: "custom_image" },
  { name: "ascii_aquarium", effect: "ascii_aquarium" },
  { name: "bookshelf_browser", effect: "bookshelf_browser" },
  { name: "train_journey", effect: "train_journey" },
  { name: "clock_mechanism", effect: "clock_mechanism" },
  { name: "weather_system", effect: "weather_system" },
  { name: "music_visualizer", effect: "music_visualizer" },
  { name: "origami_folding", effect: "origami_folding" },
  { name: "ant_colony", effect: "ant_colony" },
  { name: "neon_sign_flicker", effect: "neon_sign_flicker" },
  { name: "zen_garden", effect: "zen_garden" },
  { name: "psychedelic_mandala", effect: "psychedelic_mandala" },
  { name: "lava_lamp", effect: "lava_lamp" },
  { name: "kaleidoscope", effect: "kaleidoscope" },
  { name: "deep_dream", effect: "deep_dream" },
  { name: "trippy_tunnel", effect: "trippy_tunnel" },
  { name: "melting_screen", effect: "melting_screen" },
  { name: "shroom_vision", effect: "shroom_vision" },
  { name: "hypno_swirl", effect: "hypno_swirl" },
  { name: "electric_sheep", effect: "electric_sheep" },
  { name: "doom_fire", effect: "doom_fire" },
  { name: "pacman", effect: "pacman" },
  { name: "space_invaders", effect: "space_invaders" },
  { name: "tetris", effect: "tetris" },
  { name: "character_select", effect: "character_select" },
  { name: "achievement_unlocked", effect: "achievement_unlocked" },
  { name: "versus_screen", effect: "versus_screen" },
  { name: "world_map", effect: "world_map" },
  { name: "level_up", effect: "level_up" },
  { name: "retro_gaming_intro", effect: "retro_gaming_intro" },
];

/**
 * Web-only extension cards for effects that exist in web port but are not
 * part of the canonical source `card_definitions.py` set.
 */
export const EXTENDED_SPLASH_CARDS: SplashCard[] = [
  { name: "pixel_dissolve", effect: "pixel_dissolve", asciiArt: "default_splash" },
  { name: "fade", effect: "fade", asciiArt: "default_splash" },
  { name: "ascii_kaleidoscope", effect: "ascii_kaleidoscope", title: "TLDW Chatbook" },
  { name: "spiral_galaxy", effect: "spiral_galaxy", title: "TLDW Chatbook" },
  { name: "morphing_shape", effect: "morphing_shape", title: "TLDW Chatbook" },
  { name: "dna_helix", effect: "dna_helix", title: "TLDW Chatbook" },
  { name: "wave_ripple", effect: "wave_ripple", title: "TLDW Chatbook" },
  { name: "fireworks", effect: "fireworks", title: "TLDW Chatbook" },
  { name: "particle_swarm", effect: "particle_swarm", title: "TLDW Chatbook" },
  { name: "tetris_block", effect: "tetris_block" },
  { name: "circuit_board", effect: "circuit_board", title: "TLDW Chatbook" },
];

/** Canonical source cards + web-only extension cards. */
export const ALL_SPLASH_CARDS: SplashCard[] = [...SOURCE_CANONICAL_SPLASH_CARDS, ...EXTENDED_SPLASH_CARDS];
export const ALL_SPLASH_CARD_NAMES: string[] = ALL_SPLASH_CARDS.map((card) => card.name);
export { DEFAULT_SPLASH_CARD_NAMES } from "./splash-card-names";

/**
 * Default splash pool is canonical source-fidelity cards.
 * Pass `{ includeExtended: true }` to include web-only extension cards.
 */
export const SPLASH_CARDS: SplashCard[] = SOURCE_CANONICAL_SPLASH_CARDS;

type RandomSplashCardOptions = {
  includeExtended?: boolean;
  enabledNames?: string[];
};

/** Resolve a splash card by name from the active pool. */
export function getSplashCardByName(
  name: string,
  options?: { includeExtended?: boolean }
): SplashCard | undefined {
  const pool = options?.includeExtended ? ALL_SPLASH_CARDS : SPLASH_CARDS;
  return pool.find((card) => card.name === name);
}

/** Pick a random card. */
export function randomSplashCard(options?: RandomSplashCardOptions): SplashCard {
  const pool = options?.includeExtended ? ALL_SPLASH_CARDS : SPLASH_CARDS;
  const enabledNames = options?.enabledNames;
  const filteredPool =
    enabledNames && enabledNames.length > 0
      ? pool.filter((card) => enabledNames.includes(card.name))
      : pool;
  const effectivePool = filteredPool.length > 0 ? filteredPool : pool;
  return effectivePool[Math.floor(Math.random() * effectivePool.length)];
}
