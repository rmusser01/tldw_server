import { getDesignSystemState } from "@/design-system";
import { CharGrid } from "../../engine/CharGrid";
import type { SplashEffect } from "../../engine/types";

const SPINNER_FRAMES = ["|", "/", "-", "\\"];
const LOADING_LABEL = getDesignSystemState("loading").label;
const LOADING_DOT_SUFFIXES = ["   ", ".  ", ".. ", "..."] as const;

const BIG_SPINNER: string[][] = [
  [
    "  |||  ",
    "  |||  ",
    "  |||  ",
    "  |||  ",
    "  |||  ",
  ],
  [
    "      /",
    "    /  ",
    "  /    ",
    "/      ",
    "       ",
  ],
  [
    "       ",
    "       ",
    "-------",
    "       ",
    "       ",
  ],
  [
    "\\      ",
    "  \\    ",
    "    \\  ",
    "      \\",
    "       ",
  ],
];

export default class AsciiSpinner implements SplashEffect {
  private grid = new CharGrid(80, 24);
  private w = 0;
  private h = 0;
  private elapsed = 0;

  init(_ctx: CanvasRenderingContext2D, width: number, height: number): void {
    this.w = width;
    this.h = height;
    this.elapsed = 0;
  }

  update(elapsed: number, _dt: number): void {
    this.elapsed = elapsed;
  }

  render(ctx: CanvasRenderingContext2D): void {
    const grid = this.grid;
    grid.clear();

    grid.writeCentered(3, "= = = = = T L D W = = = = =", "#00ccff");
    grid.writeCentered(4, "Too Long; Didn't Watch", "#88aacc");

    // Big spinner
    const frame = Math.floor(this.elapsed / 200) % 4;
    const spinnerLines = BIG_SPINNER[frame];
    const startY = 8;
    for (let i = 0; i < spinnerLines.length; i++) {
      grid.writeCentered(startY + i, spinnerLines[i], "#ffcc00");
    }

    // Small spinning char
    const smallFrame = Math.floor(this.elapsed / 100) % 4;
    const sc = SPINNER_FRAMES[smallFrame];
    grid.writeCentered(15, `[ ${sc} ]`, "#00ff88");

    // Loading dots
    const dots = Math.floor(this.elapsed / 400) % 4;
    const dotStr = LOADING_LABEL + LOADING_DOT_SUFFIXES[dots];
    grid.writeCentered(17, dotStr, "#aaaaaa");

    // Progress bar
    const progress = (this.elapsed % 8000) / 8000;
    const barWidth = 40;
    const filled = Math.floor(progress * barWidth);
    const bar = "[" + "#".repeat(filled) + "-".repeat(barWidth - filled) + "]";
    grid.writeCentered(19, bar, "#00ff00");
    const pct = `${Math.floor(progress * 100)}%`;
    grid.writeCentered(20, pct, "#ffffff");

    // Decorative border
    for (let x = 0; x < 80; x++) {
      const ch = x % 2 === 0 ? "=" : "-";
      grid.setCell(x, 0, ch, "#333366");
      grid.setCell(x, 23, ch, "#333366");
    }

    const cellW = this.w / grid.cols;
    const cellH = this.h / grid.rows;
    grid.renderToCanvas(ctx, cellW, cellH);
  }

  reset(): void {
    this.elapsed = 0;
    this.grid.clear();
  }

  dispose(): void {
    this.grid.clear();
  }
}
