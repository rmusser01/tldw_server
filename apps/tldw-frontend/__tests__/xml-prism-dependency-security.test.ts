import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { describe, expect, it } from 'vitest';

// Resolve through the real consumers: a safe top-level copy must not mask an
// older parser/highlighter nested beneath EPUB.js or OpenUI.
const require = createRequire(import.meta.url);
const epubRequire = createRequire(require.resolve('epubjs'));
const openUiRequire = createRequire(require.resolve('@openuidev/react-ui'));
const syntaxRequire = createRequire(openUiRequire.resolve('react-syntax-highlighter'));
const refractorRequire = createRequire(syntaxRequire.resolve('refractor'));
const { DOMImplementation, DOMParser, XMLSerializer } = epubRequire('@xmldom/xmldom');
const { JSDOM } = require('jsdom');

describe('EPUB.js XML dependency security (CVE-2026-83610)', () => {
  it.each(['safe; <injected/> &x', 'x<injected', 'x y'])(
    'rejects an invalid entity name at creation: %s',
    (name) => {
      const doc = new DOMImplementation().createDocument(null, 'root', null);
      expect(() => doc.createEntityReference(name)).toThrow();
    }
  );

  it.each(['safe; <injected/> &x', 'x<injected', 'x y'])(
    'rejects a name mutated after creation in strict serialization: %s',
    (name) => {
      const doc = new DOMImplementation().createDocument(null, 'root', null);
      const ref = doc.createEntityReference('valid');
      ref.nodeName = name;
      // The maintained 0.8 branch takes options as the FOURTH argument.
      expect(() =>
        new XMLSerializer().serializeToString(ref, false, null, { requireWellFormed: true })
      ).toThrow();
    }
  );

  it('preserves valid references in strict serialization', () => {
    const doc = new DOMImplementation().createDocument(null, 'root', null);
    expect(
      new XMLSerializer().serializeToString(doc.createEntityReference('valid'), false, null, {
        requireWellFormed: true,
      })
    ).toBe('&valid;');
  });

  it('round-trips namespaced EPUB metadata and escaped text without adding elements', () => {
    const xml =
      '<package xmlns="http://www.idpf.org/2007/opf"><metadata>' +
      '<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Research &amp; &lt;injected/&gt;</dc:title>' +
      '</metadata></package>';
    const parser = new DOMParser();
    const doc = epubRequire('./utils/core').parse(xml, 'application/xml', true);
    const roundTrip = parser.parseFromString(
      new XMLSerializer().serializeToString(doc, false, null, { requireWellFormed: true }),
      'application/xml'
    );
    expect(
      roundTrip.getElementsByTagNameNS('http://purl.org/dc/elements/1.1/', 'title')[0].textContent
    ).toBe('Research & <injected/>');
    expect(roundTrip.getElementsByTagName('injected').length).toBe(0);
  });
});

describe('nested Prism dependency security (CVE-2024-53382)', () => {
  it.each([
    ['img', 'https://attacker.invalid/', 'components/'],
    ['form', 'https://attacker.invalid/', 'components/'],
    ['script', 'https://app.example/languages/', 'https://app.example/languages/'],
  ])(
    'accepts an autoloader location only from a genuine script element: %s',
    (tag, source, expected) => {
      const dom = new JSDOM('<!doctype html><html><body></body></html>', {
        url: 'https://app.example/',
        runScripts: 'outside-only',
      });
      try {
        const impostor = dom.window.document.createElement(tag);
        impostor.setAttribute('data-autoloader-path', source);
        Object.defineProperty(dom.window.document, 'currentScript', { value: impostor });
        dom.window.Prism = { manual: true, disableWorkerMessageHandler: true };
        // Evaluate only installed, trusted package code; JSDOM cannot fetch or
        // execute injected document scripts in outside-only mode.
        dom.window.eval(
          readFileSync(refractorRequire.resolve('prismjs/components/prism-core'), 'utf8')
        );
        dom.window.eval(
          readFileSync(
            refractorRequire.resolve('prismjs/plugins/autoloader/prism-autoloader'),
            'utf8'
          )
        );
        expect(dom.window.Prism.plugins.autoloader.languages_path).toBe(expected);
      } finally {
        dom.window.close();
      }
    }
  );

  it('preserves JavaScript highlighting through the real Refractor consumer', () => {
    const refractor = syntaxRequire('refractor');
    const tree = refractor.highlight('const answer = 42;', 'javascript');
    expect(tree).toContainEqual({
      type: 'element',
      tagName: 'span',
      properties: { className: ['token', 'keyword'] },
      children: [{ type: 'text', value: 'const' }],
    });
    expect(tree).toContainEqual({
      type: 'element',
      tagName: 'span',
      properties: { className: ['token', 'number'] },
      children: [{ type: 'text', value: '42' }],
    });
  });
});
