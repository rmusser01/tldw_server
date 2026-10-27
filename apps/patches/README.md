# jsdom stylesheet prefilter

`jsdom@28.1.0.patch` backports the conservative stylesheet subject prefilter from [jsdom v29.1.1](https://github.com/jsdom/jsdom/blob/v29.1.1/lib/jsdom/living/css/helpers/computed-style.js). It uses the pinned selector 7.1.1 extractor, retains full matching for XML and quirks documents, and keys extraction by selector text so CSSOM edits remain current. Bun applies the patch through `patchedDependencies` during frozen installation.

Keep this backport until a compatible unpatched jsdom passes the stylesheet prefilter, accessibility, and full coverage regressions. The patch preserves jsdom 28's CSS implementation and existing Node 20.19 requirement. Regression coverage lives in `packages/ui/src/utils/__tests__/jsdom-stylesheet-prefilter.test.ts`.

The upstream code is distributed under the following notice:

Copyright (c) 2010 Elijah Insua

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
