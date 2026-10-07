/** Translate a UTF-16 parser offset into the editor's Unicode-scalar offset. */
export function scalarOffset(source: string, offset: number): number {
  let scalars = 0;
  for (let i = 0; i < Math.min(offset, source.length); scalars++) {
    i += source.codePointAt(i)! > 0xffff ? 2 : 1;
  }
  return scalars;
}
