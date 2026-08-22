const isAsciiAlpha = (code: number): boolean =>
  (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a)

const isAsciiDigit = (code: number): boolean => code >= 0x30 && code <= 0x39

const isAsciiAlphaNumeric = (code: number): boolean =>
  isAsciiAlpha(code) || isAsciiDigit(code)

const isSchemeCharacter = (code: number): boolean =>
  isAsciiAlphaNumeric(code) || code === 0x2b || code === 0x2d || code === 0x2e

const isAsciiWhitespaceOrControl = (code: number): boolean =>
  code <= 0x20 || code === 0x7f

const isReferenceBoundary = (value: string, index: number): boolean => {
  if (index === 0) return true
  const previous = value.charCodeAt(index - 1)
  return (
    isAsciiWhitespaceOrControl(previous) ||
    previous === 0x22 ||
    previous === 0x27 ||
    previous === 0x28 ||
    previous === 0x3c ||
    previous === 0x5b ||
    previous === 0x7b ||
    previous === 0x2c ||
    previous === 0x3b ||
    previous === 0x3d
  )
}

const hasImmediateContent = (value: string, index: number): boolean =>
  index < value.length && !isAsciiWhitespaceOrControl(value.charCodeAt(index))

const hasGenericScheme = (value: string): boolean => {
  let index = 0
  while (index < value.length) {
    const code = value.charCodeAt(index)
    const previousIsSchemeCharacter =
      index > 0 && isSchemeCharacter(value.charCodeAt(index - 1))
    if (!isAsciiAlpha(code) || previousIsSchemeCharacter) {
      index += 1
      continue
    }
    let end = index + 1
    while (end < value.length && isSchemeCharacter(value.charCodeAt(end))) end += 1
    if (
      end < value.length &&
      value.charCodeAt(end) === 0x3a &&
      hasImmediateContent(value, end + 1)
    ) {
      return true
    }
    index = end
  }
  return false
}

const hasPathReference = (value: string): boolean => {
  for (let index = 0; index < value.length; index += 1) {
    if (!isReferenceBoundary(value, index)) continue
    const code = value.charCodeAt(index)
    if (code === 0x2f && hasImmediateContent(value, index + 1)) return true
    if (code !== 0x2e) continue
    if (
      value.charCodeAt(index + 1) === 0x2f &&
      hasImmediateContent(value, index + 2)
    ) {
      return true
    }
    if (
      value.charCodeAt(index + 1) === 0x2e &&
      value.charCodeAt(index + 2) === 0x2f &&
      hasImmediateContent(value, index + 3)
    ) {
      return true
    }
  }
  return false
}

const isHostCharacter = (code: number): boolean =>
  isAsciiAlphaNumeric(code) || code === 0x2d || code === 0x2e

const isIdnaDot = (code: number): boolean =>
  code === 0x2e || code === 0x3002 || code === 0xff0e || code === 0xff61

const isUnicodeHostLabelCharacter = (code: number): boolean =>
  isAsciiAlphaNumeric(code) || code === 0x2d || (code > 0x7f && !isIdnaDot(code))

const isHostSuffixDelimiter = (code: number): boolean =>
  code === 0x2f || code === 0x3a || code === 0x3f || code === 0x23

const hasUnicodeHostReference = (value: string): boolean => {
  let index = 0
  while (index < value.length) {
    if (!isUnicodeHostLabelCharacter(value.charCodeAt(index))) {
      index += 1
      continue
    }

    let cursor = index
    let labelStart = index
    let sawDot = false
    let sawNonAscii = false
    let validLabels = true
    while (cursor < value.length) {
      const code = value.charCodeAt(cursor)
      if (isUnicodeHostLabelCharacter(code)) {
        if (code > 0x7f) sawNonAscii = true
        cursor += 1
        continue
      }
      if (!isIdnaDot(code)) break
      if (
        cursor === labelStart ||
        value.charCodeAt(labelStart) === 0x2d ||
        value.charCodeAt(cursor - 1) === 0x2d
      ) {
        validLabels = false
      }
      sawDot = true
      cursor += 1
      labelStart = cursor
    }
    if (
      cursor === labelStart ||
      value.charCodeAt(labelStart) === 0x2d ||
      value.charCodeAt(cursor - 1) === 0x2d
    ) {
      validLabels = false
    }
    if (
      validLabels &&
      sawDot &&
      sawNonAscii &&
      isHostSuffixDelimiter(value.charCodeAt(cursor)) &&
      hasImmediateContent(value, cursor + 1)
    ) {
      return true
    }
    index = cursor > index ? cursor : index + 1
  }
  return false
}

const isValidHostLabel = (label: string): boolean => {
  if (
    label.length === 0 ||
    !isAsciiAlphaNumeric(label.charCodeAt(0)) ||
    !isAsciiAlphaNumeric(label.charCodeAt(label.length - 1))
  ) {
    return false
  }
  for (let index = 1; index < label.length - 1; index += 1) {
    const code = label.charCodeAt(index)
    if (!isAsciiAlphaNumeric(code) && code !== 0x2d) return false
  }
  return true
}

const isIpv4Host = (labels: string[]): boolean => {
  if (labels.length !== 4) return false
  for (const label of labels) {
    if (label.length === 0 || label.length > 3) return false
    let octet = 0
    for (let index = 0; index < label.length; index += 1) {
      const code = label.charCodeAt(index)
      if (!isAsciiDigit(code)) return false
      octet = octet * 10 + code - 0x30
    }
    if (octet > 255) return false
  }
  return true
}

const isDomainHost = (labels: string[]): boolean => {
  if (labels.length < 2 || !labels.every(isValidHostLabel)) return false
  const finalLabel = labels[labels.length - 1]
  if (finalLabel.length < 2) return false
  for (let index = 0; index < finalLabel.length; index += 1) {
    if (isAsciiAlpha(finalLabel.charCodeAt(index))) return true
  }
  return false
}

const isBracketedIpHostAt = (value: string, index: number): boolean => {
  if (value.charCodeAt(index) !== 0x5b || !isReferenceBoundary(value, index)) return false
  let cursor = index + 1
  let sawColon = false
  while (cursor < value.length && value.charCodeAt(cursor) !== 0x5d) {
    const code = value.charCodeAt(cursor)
    const isHex = isAsciiDigit(code) ||
      (code >= 0x41 && code <= 0x46) ||
      (code >= 0x61 && code <= 0x66)
    if (code === 0x3a) sawColon = true
    else if (!isHex && code !== 0x2e) return false
    cursor += 1
  }
  if (!sawColon || cursor >= value.length || cursor === index + 1) return false
  const next = value.charCodeAt(cursor + 1)
  return (
    cursor + 1 === value.length ||
    isAsciiWhitespaceOrControl(next) ||
    next === 0x3a ||
    next === 0x2f ||
    next === 0x3f ||
    next === 0x23 ||
    next === 0x2c ||
    next === 0x2e ||
    next === 0x29 ||
    next === 0x5d
  )
}

const hasHostReference = (value: string): boolean => {
  let index = 0
  while (index < value.length) {
    if (isBracketedIpHostAt(value, index)) return true
    const code = value.charCodeAt(index)
    const previousIsHostCharacter = index > 0 && isHostCharacter(value.charCodeAt(index - 1))
    if (!isAsciiAlphaNumeric(code) || previousIsHostCharacter) {
      index += 1
      continue
    }
    let end = index + 1
    while (end < value.length && isHostCharacter(value.charCodeAt(end))) end += 1
    let hostEnd = end
    while (hostEnd > index && value.charCodeAt(hostEnd - 1) === 0x2e) hostEnd -= 1
    const host = value.slice(index, hostEnd)
    const labels = host.split(".")
    if (
      host.toLowerCase() === "localhost" ||
      isIpv4Host(labels) ||
      isDomainHost(labels)
    ) {
      return true
    }
    index = end
  }
  return false
}

export const hasDirectUrlLikeText = (value: string): boolean =>
  value.includes("//") ||
  hasGenericScheme(value) ||
  hasPathReference(value) ||
  hasHostReference(value) ||
  hasUnicodeHostReference(value)
