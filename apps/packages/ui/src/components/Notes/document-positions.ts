/** Scalar diagnostics refer to exact source; native selection uses UTF-16. */
export function scalarToUtf16(source: string, offset: number): number {
  let index = 0;
  for (let count = 0; count < offset && index < source.length; count++) {
    index += source.codePointAt(index)! > 0xffff ? 2 : 1;
  }
  return index;
}
export function logicalPosition(
  source: string,
  offset: number,
): { line: number; column: number } {
  let line = 1,
    column = 1;
  for (let index = 0; index < offset; ) {
    const code = source.codePointAt(index)!;
    if (code === 13) {
      line++;
      column = 1;
      index += source[index + 1] === "\n" && index + 1 < offset ? 2 : 1;
    } else {
      if (code === 10) {
        line++;
        column = 1;
      } else column++;
      index += code > 0xffff ? 2 : 1;
    }
  }
  return { line, column };
}
