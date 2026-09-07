/* Candidate-only controls for Debian's uint16_t XML_UNICODE ABI.
 * Upstream runtests deliberately does not support this character mode.
 * All XML/entity data is local and bounded; no file or network entity loading.
 */
#include <expat.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct control {
  int mode, chunk, seen, child, failed;
};

static int equal_ascii(const XML_Char *value, const char *expected) {
  while (*expected) {
    if (*value++ != (unsigned char)*expected++) return 0;
  }
  return *value == 0;
}

static enum XML_Status parse(XML_Parser parser, const char *text, int chunk) {
  size_t length = strlen(text), offset = 0;
  if (chunk == 0) return XML_Parse(parser, text, (int)length, XML_TRUE);
  while (offset < length) {
    size_t size = length - offset;
    if (size > (size_t)chunk) size = (size_t)chunk;
    if (XML_Parse(parser, text + offset, (int)size,
                  offset + size == length) != XML_STATUS_OK)
      return XML_STATUS_ERROR;
    offset += size;
  }
  return XML_STATUS_OK;
}

static void XMLCALL start(void *data, const XML_Char *name,
                          const XML_Char **attributes) {
  struct control *state = data;
  int first = 0, second = 0;
  if (equal_ascii(name, "doc")) return;
  if (!equal_ascii(name, state->mode == 3 ? "urn:test|tag" : "tag")) {
    state->failed = 1;
    return;
  }
  for (int i = 0; attributes[i]; i += 2) {
    if (equal_ascii(attributes[i], "first"))
      first = equal_ascii(attributes[i + 1], " a  b ");
    else if (equal_ascii(attributes[i], state->mode == 3 ? "urn:test|second" : "second"))
      second = equal_ascii(attributes[i + 1], "a b");
    else
      state->failed = 1;
  }
  state->seen++;
  if (!second || (state->mode == 1 && !first)) state->failed = 1;
}

static int XMLCALL external(XML_Parser parent, const XML_Char *context,
                            const XML_Char *base, const XML_Char *system,
                            const XML_Char *public_id) {
  struct control *state = XML_GetUserData(parent);
  XML_Parser child;
  enum XML_Status status;
  (void)base; (void)system; (void)public_id;
  /* Non-NULL context selects dtdCopy, the security-fix lifecycle regression. */
  if (!context) return XML_STATUS_ERROR;
  child = XML_ExternalEntityParserCreate(parent, context, NULL);
  if (!child) return XML_STATUS_ERROR;
  state->child++;
  status = parse(child, "<tag second=' a  b '/>", state->chunk);
  XML_ParserFree(child);
  return status;
}

int main(int argc, char **argv) {
  const char *documents[] = {
    "<!DOCTYPE tag [<!ELEMENT tag EMPTY>"
    "<!ATTLIST tag first CDATA #IMPLIED>"
    "<!ATTLIST tag first NMTOKENS #IMPLIED>"
    "<!ATTLIST tag second NMTOKENS ' a  b '>]><tag first=' a  b '/>",
    "<!DOCTYPE doc [<!ENTITY e SYSTEM 'entity.ent'><!ELEMENT doc ANY>"
    "<!ELEMENT tag EMPTY><!ATTLIST tag first CDATA #IMPLIED>"
    "<!ATTLIST tag second NMTOKENS #IMPLIED>]><doc>&e;</doc>",
    "<n:tag xmlns:n='urn:test' n:second='a b'/>"
  };
  const int chunks[] = {0, 1, 7};
  if (sizeof(XML_Char) != sizeof(uint16_t) || argc != 2
      || strcmp(XML_ExpatVersion(), argv[1]) != 0) {
    fprintf(stderr, "unexpected wide ABI or version: %s\n", XML_ExpatVersion());
    return EXIT_FAILURE;
  }
  for (int mode = 1; mode <= 3; mode++) {
    for (size_t n = 0; n < sizeof(chunks) / sizeof(chunks[0]); n++) {
      struct control state = {mode, chunks[n], 0, 0, 0};
      XML_Parser parser = XML_ParserCreateNS(NULL, '|');
      if (!parser) return EXIT_FAILURE;
      XML_SetUserData(parser, &state);
      XML_SetElementHandler(parser, start, NULL);
      XML_SetExternalEntityRefHandler(parser, external);
      enum XML_Status status = parse(parser, documents[mode - 1], chunks[n]);
      XML_ParserFree(parser);
      if (status != XML_STATUS_OK || state.failed || state.seen != 1
          || state.child != (mode == 2)) {
        fprintf(stderr, "wide control failed: mode=%d chunk=%d\n", mode, chunks[n]);
        return EXIT_FAILURE;
      }
      printf("PASS: wide mode=%d chunk=%d\n", mode, chunks[n]);
    }
  }
  puts("PASS: wide XML controls");
  return EXIT_SUCCESS;
}
