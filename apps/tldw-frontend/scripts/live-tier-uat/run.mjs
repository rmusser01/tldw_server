#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process"
import {
  appendFileSync,
  chmodSync,
  closeSync,
  constants,
  fstatSync,
  ftruncateSync,
  openSync,
  writeSync,
  existsSync,
  lstatSync,
  realpathSync,
  renameSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { createRequire } from "node:module"
import { createHash, randomBytes } from "node:crypto"
import path from "node:path"
import { tmpdir } from "node:os"
import { fileURLToPath } from "node:url"
import { redactText } from "../onboarding-uat/artifacts.mjs"
import { reservePorts } from "../onboarding-uat/ports.mjs"
import {
  spawnLoggedProcess,
  stopProcessTree,
  waitForHttpOk,
} from "../onboarding-uat/processes.mjs"
import { resolvePythonCommand } from "../onboarding-uat/run.mjs"
import { buildCollectionCommand } from "../list-release-tests.mjs"
import { inventoryProjects } from "./inventory-api-mocks.mjs"
import { buildLiveTierBackendEnv, buildLiveTierProfile } from "./profile.mjs"
import {
  assertProjectAccounting,
  collectSkippedTests,
  collectPlaywrightCases,
  renderMarkdownReport,
  summarizePlaywrightReport,
} from "./report.mjs"

// Fixed rows contain numeric enums and opaque realm-local identities only.
export function deferred43Recorder(capacity, realm, source) {
  const tape = new Float64Array(capacity * 24)
  const objects = new WeakMap(), nodes = new WeakMap(), events = new WeakMap()
  const declarations = new WeakMap(), owners = new WeakMap(), links = new WeakMap()
  const forwardOwners = new WeakMap(), callbackRefs = new WeakMap(), parentCallbacks = new WeakMap(), exportParents = new WeakMap(), menuParents = new WeakMap(), menuCallbacks = new WeakMap(), aliases = new WeakMap(), callbackOwners = new WeakMap(), contexts = new WeakMap(), parentOwners = new WeakMap(), popupOwners = new WeakMap(), delaySerials = new WeakMap(), reasons = new WeakMap()
  const origins = new WeakMap(), rootNodes = new WeakMap(), activeOwners = new WeakMap(), timerOrigins = new Uint32Array(513)
  const timers = new Uint32Array(513), timerOwners = new Uint32Array(513), timerKinds = new Uint32Array(513), timerRoles = new Uint32Array(513), timerGenerations = new Uint32Array(513), timerReturnTypes = new Uint32Array(513), timerReturns = new Float64Array(513), timerDelays = new Float64Array(513)
  let objectCount = 0, nodeCount = 0, eventCount = 0, timerCount = 0
  let count = 0, faults = 0, first = 0, drained = false, activeOrigin = 0
  const allowed = new Uint32Array(71)
  for (const site of [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,40,41,42,43,44,45,46,50,51,52,53,54,55,56]) allowed[site] = 1
  const origin = performance.timeOrigin
  const objectLike = value => value !== null && (typeof value === "object" || typeof value === "function")
  function id(value, kind = 0) {
    if (!objectLike(value)) return 0
    const map = kind === 1 ? nodes : kind === 2 ? events : objects
    let result = map.get(value)
    if (result) return result
    const current = kind === 1 ? nodeCount : kind === 2 ? eventCount : objectCount
    if (current >= (kind === 0 ? 512 : 256)) { faults |= 1; return 0 }
    result = current + 1
    map.set(value, result)
    if (kind === 1) nodeCount = result
    else if (kind === 2) eventCount = result
    else objectCount = result
    return result
  }
  function row(site, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0, t = 0, u = 0) {
    try {
      if (!Number.isInteger(site) || site < 1 || site > 70 || !allowed[site] || arguments.length > 22) { faults |= 2; return 0 }
      for (let index = 1; index < arguments.length; index++) {
        if (typeof arguments[index] !== "number" || !Number.isFinite(arguments[index])) { faults |= 2; return 0 }
      }
      if (drained || count === capacity) { faults |= 1; return 0 }
      const offset = count++ * 24
      tape[offset] = site; tape[offset + 1] = count; tape[offset + 2] = performance.now()
      tape[offset + 3] = a; tape[offset + 4] = b; tape[offset + 5] = c
      tape[offset + 6] = d; tape[offset + 7] = e; tape[offset + 8] = f
      tape[offset + 9] = g; tape[offset + 10] = h; tape[offset + 11] = i
      tape[offset + 12] = j; tape[offset + 13] = k; tape[offset + 14] = l
      tape[offset + 15] = m; tape[offset + 16] = n; tape[offset + 17] = o
      tape[offset + 18] = p; tape[offset + 19] = q; tape[offset + 20] = r
      tape[offset + 21] = s; tape[offset + 22] = t; tape[offset + 23] = u
      return count
    } catch { faults |= 4; return 0 }
  }
  function declare(role, ref, upstream, downstream) {
    if (!objectLike(ref) || (role !== 1 && role !== 2)) { faults |= 8; return }
    const old = declarations.get(ref)
    if (old && old[0] !== role) { old[0] = 0; faults |= 8; return }
    const value = old || [role, 0, 0]
    value[1] = objectLike(upstream) ? upstream : 0; value[2] = objectLike(downstream) ? downstream : 0
    declarations.set(ref, value)
  }
  function owner(value, ref, downstream, popupChange, parentContext, context) {
    if (objectLike(ref)) forwardOwners.set(ref, value)
    if (objectLike(context)) contexts.set(context, value)
    if (objectLike(parentContext)) parentOwners.set(value, contexts.get(parentContext))
    if (objectLike(popupChange)) callbackOwners.set(popupChange, value)
    if (!objectLike(value)) { faults |= 8; return }
    let state = owners.get(value)
    if (!state) { state = [value, 0, 0, 0, 0, 0, 0, 0]; owners.set(value, state) }
    const declaration = objectLike(ref) && declarations.get(ref)
    const link = objectLike(popupChange) && links.get(popupChange)
    const nextRole = declaration ? declaration[0] : link ? link[0] : 0
    if (state[2] !== (objectLike(ref) ? ref : 0) || state[1] !== nextRole) { state[4] = 0; state[5]++ }
    state[1] = nextRole
    state[2] = objectLike(ref) ? ref : 0; state[3] = objectLike(downstream) ? downstream : 0; state[6] = objectLike(popupChange) ? popupChange : 0
    state[7] = link ? link[1] : 0
    const menuCallback = objectLike(popupChange) && exportParents.get(popupChange)
    const menuOwner = objectLike(menuCallback) && menuCallbacks.get(menuCallback)
    if (menuOwner && objectLike(parentContext) && contexts.get(parentContext)) menuParents.set(menuOwner, contexts.get(parentContext))
    if (declaration && declaration[2] !== (objectLike(downstream) ? downstream : 0)) { state[1] = 0; faults |= 8 }
  }
  function link(callback, role, parent) {
    if (!objectLike(callback) || !Number.isInteger(role) || role < 3 || role > 5) { faults |= 8; return }
    const previous = links.get(callback), parentRef = objectLike(parent) ? parent : 0
    if (previous && (previous[0] !== role || previous[1] !== parentRef)) { previous[0] = 0; faults |= 8; return }
    links.set(callback, [role, parentRef])
  }
  function resolve(value) {
    for (let depth = 0; depth < 4 && objectLike(value); depth++) {
      const next = aliases.get(value) || callbackOwners.get(value)
      if (!next) return value
      value = next
    }
    return value
  }
  function role(value) {
    value = resolve(value)
    const state = objectLike(value) && owners.get(value)
    if (state && state[1] === 3 && role(parentOwners.get(value)) !== 1) return 0
    if (state && state[1] === 4 && role(menuParents.get(value)) !== 1) return 0
    return state && state[4] && state[1] ? state[1] : 0
  }
  function target(value, node, marked) {
    const state = objectLike(value) && owners.get(value)
    if (!state) { faults |= 8; return }
    const nodeId = id(node, 1), previousNode = state[4]
    if (marked && nodeId && (state[1] === 1 || state[1] === 2)) {
      let pair = rootNodes.get(node)
      if (!pair) { pair = [0, 0]; rootNodes.set(node, pair) }
      const old = pair[state[1] - 1]
      if (old && old !== state[0] && activeOwners.get(old) === nodeId) { state[1] = 0; faults |= 8 }
      else pair[state[1] - 1] = state[0]
    }
    if (state[4] !== nodeId) state[5]++
    state[4] = marked || state[1] === 3 ? nodeId : 0
    if (marked && !state[1]) faults |= 8
    activeOwners.set(value, state[4])
    if (previousNode || state[4]) row(1, id(value), state[1], state[5], nodeId, id(state[2]), id(state[3]), id(state[6]), id(state[7]), id(parentOwners.get(value) || menuParents.get(value)))
  }
  function observe(site, value, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0) {
    value = resolve(value)
    const state = objectLike(value) && owners.get(value)
    if (state && role(value)) {
      const index = row(site, id(value), state[1], state[5], a, b, c, d, e, f, origins.get(value) || activeOrigin)
      if (site === 5 || site === 7 || site === 8 || site === 9 || site === 14 || site === 15) origins.set(value, index)
      return index
    }
  }
  function callback(site, callback, a = 0, b = 0) {
    const data = objectLike(callback) && links.get(callback)
    const ref = objectLike(callback) && callbackRefs.get(callback)
    const owner = objectLike(ref) ? forwardOwners.get(ref) : callbackOwners.get(callback)
    if (data && data[0] && role(owner)) row(site, id(callback), data[0], id(data[1]), a, b, id(owner))
  }
  function eventType(value) {
    const result = value === "pointerdown" ? 1 : value === "mousedown" ? 2 : value === "mouseup" ? 3 : value === "click" ? 4 : value === "pointerenter" ? 5 : value === "mouseenter" ? 6 : value === "pointerleave" ? 7 : value === "mouseleave" ? 8 : value === "contextmenu" ? 9 : value === "focus" ? 10 : value === "blur" ? 11 : value === "touchstart" ? 12 : value === "touchend" ? 13 : value === "pointerup" ? 14 : value === "touchcancel" ? 15 : value === "mousemove" ? 16 : 0
    if (!result) faults |= 2
    return result
  }
  function event(site, value, event) {
    if (!role(value) || !objectLike(event)) return
    observe(site, value, id(event.nativeEvent || event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, typeof event.clientX === "number" ? event.clientX : -1, typeof event.clientY === "number" ? event.clientY : -1)
  }
  function native(site, key, node, result) {
    if (!key || key.length !== 3 || !key.every(Number.isSafeInteger)) { faults |= 2; return }
    row(site, key[0], key[1], key[2], id(node, 1), result)
  }
  function nativeEvent(key, node, event, result, fresh, preliminary = false) {
    if (!key || !fresh) return
    const index = row(51, key[0], key[1], key[2], id(node, 1), result, id(event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, event.clientX, event.clientY, typeof event.button === "number" ? event.button : -1, event.timeStamp)
    if (result === 2 && preliminary && !first) first = index
  }
  function alias(value, parent) { if (objectLike(value) && objectLike(parent)) aliases.set(value, parent) }
  function parent(openCallback, menuCallback) {
    link(openCallback, 5, openCallback); link(menuCallback, 5, openCallback)
    if (objectLike(openCallback)) parentCallbacks.set(openCallback, menuCallback)
  }
  function child(upstream, downstream, ref) {
    const parent = objectLike(upstream) && links.get(upstream)
    if (parent && parent[0] === 5) {
      link(downstream, 5, ref); callbackRefs.set(upstream, ref); callbackRefs.set(downstream, ref)
      const close = parentCallbacks.get(upstream); if (objectLike(close)) callbackRefs.set(close, ref)
    }
  }
  function menu(value, callback) {
    if (!owners.has(value)) { owner(value, null, callback, null); owners.get(value)[1] = 4 }
    menuCallbacks.set(callback, value)
    link(callback, 4, value)
  }
  function menuCommit(value, callback) {
    const state = owners.get(value)
    if (!state || !objectLike(value.current)) { faults |= 8; return }
    target(value, value.current, true); callbackOwners.set(callback, value)
  }
  function submenu(callback, popupRef, parentCallback) {
    const parent = objectLike(parentCallback) && links.get(parentCallback)
    if (!parent || parent[0] !== 4) { faults |= 8; return }
    link(callback, 3, parentCallback); exportParents.set(callback, parentCallback)
  }
  function popup(value, callback) { alias(value, callback) }
  function bindPopup(value, node) { if (objectLike(node)) popupOwners.set(node, value) }
  function timerRecord(site, serial) {
    row(site, timerOwners[serial], timerRoles[serial], timerGenerations[serial], serial, timerKinds[serial], timerReturnTypes[serial], timerReturns[serial], timerDelays[serial], timerOrigins[serial])
  }
  function installed(ref, callback, kind, returned, delay) {
    const value = resolve(callback), state = objectLike(value) && owners.get(value)
    if (!state || !role(value)) return 0
    if (kind !== 1 && kind !== 2 || !Number.isFinite(delay)) { faults |= 2; return 0 }
    const type = typeof returned === "number" ? 1 : objectLike(returned) ? 2 : 0
    if (!type || type === 1 && (!Number.isSafeInteger(returned) || returned < 0)) { faults |= 2; return 0 }
    if (timerCount === 512) { faults |= 1; return 0 }
    const old = objectLike(ref) && delaySerials.get(ref)
    if (old && timers[old] === 1) faults |= 16
    const serial = ++timerCount
    timers[serial] = 1; timerOwners[serial] = id(value); timerRoles[serial] = state[1]; timerGenerations[serial] = state[5]
    timerKinds[serial] = kind; timerReturnTypes[serial] = type; timerReturns[serial] = type === 1 ? returned : id(returned)
    timerDelays[serial] = delay; timerOrigins[serial] = origins.get(value) || 0
    if (objectLike(ref)) delaySerials.set(ref, serial)
    timerRecord(10, serial)
    return serial
  }
  function cleared(ref, kind) {
    const serial = objectLike(ref) && delaySerials.get(ref)
    if (!serial) { if (role(ref)) faults |= 16; return }
    if (timerKinds[serial] !== kind) { faults |= 16; return }
    if (timers[serial] === 1) { timers[serial] = 2; timerRecord(11, serial) }
    else if (timers[serial] === 3) { timers[serial] = 4; timerRecord(31, serial) }
    else { faults |= 16; return }
    delaySerials.delete(ref)
  }
  function fired(ref, callback, serial, kind) {
    if (!serial) { if (role(callback)) faults |= 16; return }
    const value = resolve(callback)
    if (!role(value) || timerOwners[serial] !== id(value) || timerKinds[serial] !== kind || timers[serial] !== 1) { faults |= 16; return }
    timers[serial] = 3; origins.set(value, timerOrigins[serial]); timerRecord(12, serial)
  }
  function immediate(callback, phase) { observe(phase === 1 ? 29 : 30, callback, 0, phase) }
  function reason(value, reason) {
    if (reason !== 3 || reasons.get(value) !== 4) reasons.set(value, reason)
    observe(19, value, reason)
  }
  function menuKeys(value, keys) {
    observe(20, value, reasons.get(value) || 0, keys.length)
    reasons.delete(value)
  }
  function geometry(popup, target, targetRect, popupRect, next) {
    const value = objectLike(popup) && popupOwners.get(popup), state = value && owners.get(value)
    if (!state || !role(value)) return
    row(25, id(value), state[1], state[5], id(popup, 1), id(target, 1), targetRect.x, targetRect.y, targetRect.width, targetRect.height, popupRect.x, popupRect.y, popupRect.width, popupRect.height, next.offsetX, next.offsetY, next.offsetR, next.offsetB, next.arrowX, next.arrowY, next.scaleX, next.scaleY)
  }
  function hit(key, target, top, point) {
    const tag = top?.nodeName
    const descriptor = tag === "TEXTAREA" ? 2 : tag === "BUTTON" ? 3 : tag === "LI" ? 4 : tag === "SPAN" ? 5 : tag === "DIV" ? 6 : 7
    row(54, key[0], key[1], key[2], id(target, 1), id(top, 1), descriptor, -1, point.x, point.y)
  }
  function drain() {
    if (drained) throw new Error("already drained")
    drained = true
    const rows = []
    for (let index = 0; index < count; index++) rows.push(Array.from(tape.subarray(index * 24, (index + 1) * 24)))
    return { schema: 43, realm, source, timeOrigin: origin, count, faults, first, counts: [objectCount, nodeCount, eventCount, timerCount], rows, order: 0 }
  }
  return Object.freeze({ source, realm, hit, alias, parent, child, menu, menuCommit, submenu, popup, bindPopup, installed, cleared, fired, immediate, reason, menuKeys, geometry, row, id, declare, owner, link, role, target, observe, callback, event, native, nativeEvent, drain, fault: flag => { faults |= flag }, origin: value => { activeOrigin = value }, node: value => id(value, 1) })
}
// END DEFERRED43 RECORDER

const deferred43Source = "aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d"
const deferred43AppSource = "try { if (globalThis.window === globalThis.top) { let df43Owner = globalThis; while (df43Owner && !Object.getOwnPropertyDescriptor(df43Owner, \"__df43App\")) df43Owner = Object.getPrototypeOf(df43Owner); if (!df43Owner) Object.defineProperty(globalThis, \"__df43App\", { value: (function deferred43Recorder(capacity, realm, source) {\n  const tape = new Float64Array(capacity * 24)\n  const objects = new WeakMap(), nodes = new WeakMap(), events = new WeakMap()\n  const declarations = new WeakMap(), owners = new WeakMap(), links = new WeakMap()\n  const forwardOwners = new WeakMap(), callbackRefs = new WeakMap(), parentCallbacks = new WeakMap(), exportParents = new WeakMap(), menuParents = new WeakMap(), menuCallbacks = new WeakMap(), aliases = new WeakMap(), callbackOwners = new WeakMap(), contexts = new WeakMap(), parentOwners = new WeakMap(), popupOwners = new WeakMap(), delaySerials = new WeakMap(), reasons = new WeakMap()\n  const origins = new WeakMap(), rootNodes = new WeakMap(), activeOwners = new WeakMap(), timerOrigins = new Uint32Array(513)\n  const timers = new Uint32Array(513), timerOwners = new Uint32Array(513), timerKinds = new Uint32Array(513), timerRoles = new Uint32Array(513), timerGenerations = new Uint32Array(513), timerReturnTypes = new Uint32Array(513), timerReturns = new Float64Array(513), timerDelays = new Float64Array(513)\n  let objectCount = 0, nodeCount = 0, eventCount = 0, timerCount = 0\n  let count = 0, faults = 0, first = 0, drained = false, activeOrigin = 0\n  const allowed = new Uint32Array(71)\n  for (const site of [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,40,41,42,43,44,45,46,50,51,52,53,54,55,56]) allowed[site] = 1\n  const origin = performance.timeOrigin\n  const objectLike = value => value !== null && (typeof value === \"object\" || typeof value === \"function\")\n  function id(value, kind = 0) {\n    if (!objectLike(value)) return 0\n    const map = kind === 1 ? nodes : kind === 2 ? events : objects\n    let result = map.get(value)\n    if (result) return result\n    const current = kind === 1 ? nodeCount : kind === 2 ? eventCount : objectCount\n    if (current >= (kind === 0 ? 512 : 256)) { faults |= 1; return 0 }\n    result = current + 1\n    map.set(value, result)\n    if (kind === 1) nodeCount = result\n    else if (kind === 2) eventCount = result\n    else objectCount = result\n    return result\n  }\n  function row(site, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0, t = 0, u = 0) {\n    try {\n      if (!Number.isInteger(site) || site < 1 || site > 70 || !allowed[site] || arguments.length > 22) { faults |= 2; return 0 }\n      for (let index = 1; index < arguments.length; index++) {\n        if (typeof arguments[index] !== \"number\" || !Number.isFinite(arguments[index])) { faults |= 2; return 0 }\n      }\n      if (drained || count === capacity) { faults |= 1; return 0 }\n      const offset = count++ * 24\n      tape[offset] = site; tape[offset + 1] = count; tape[offset + 2] = performance.now()\n      tape[offset + 3] = a; tape[offset + 4] = b; tape[offset + 5] = c\n      tape[offset + 6] = d; tape[offset + 7] = e; tape[offset + 8] = f\n      tape[offset + 9] = g; tape[offset + 10] = h; tape[offset + 11] = i\n      tape[offset + 12] = j; tape[offset + 13] = k; tape[offset + 14] = l\n      tape[offset + 15] = m; tape[offset + 16] = n; tape[offset + 17] = o\n      tape[offset + 18] = p; tape[offset + 19] = q; tape[offset + 20] = r\n      tape[offset + 21] = s; tape[offset + 22] = t; tape[offset + 23] = u\n      return count\n    } catch { faults |= 4; return 0 }\n  }\n  function declare(role, ref, upstream, downstream) {\n    if (!objectLike(ref) || (role !== 1 && role !== 2)) { faults |= 8; return }\n    const old = declarations.get(ref)\n    if (old && old[0] !== role) { old[0] = 0; faults |= 8; return }\n    const value = old || [role, 0, 0]\n    value[1] = objectLike(upstream) ? upstream : 0; value[2] = objectLike(downstream) ? downstream : 0\n    declarations.set(ref, value)\n  }\n  function owner(value, ref, downstream, popupChange, parentContext, context) {\n    if (objectLike(ref)) forwardOwners.set(ref, value)\n    if (objectLike(context)) contexts.set(context, value)\n    if (objectLike(parentContext)) parentOwners.set(value, contexts.get(parentContext))\n    if (objectLike(popupChange)) callbackOwners.set(popupChange, value)\n    if (!objectLike(value)) { faults |= 8; return }\n    let state = owners.get(value)\n    if (!state) { state = [value, 0, 0, 0, 0, 0, 0, 0]; owners.set(value, state) }\n    const declaration = objectLike(ref) && declarations.get(ref)\n    const link = objectLike(popupChange) && links.get(popupChange)\n    const nextRole = declaration ? declaration[0] : link ? link[0] : 0\n    if (state[2] !== (objectLike(ref) ? ref : 0) || state[1] !== nextRole) { state[4] = 0; state[5]++ }\n    state[1] = nextRole\n    state[2] = objectLike(ref) ? ref : 0; state[3] = objectLike(downstream) ? downstream : 0; state[6] = objectLike(popupChange) ? popupChange : 0\n    state[7] = link ? link[1] : 0\n    const menuCallback = objectLike(popupChange) && exportParents.get(popupChange)\n    const menuOwner = objectLike(menuCallback) && menuCallbacks.get(menuCallback)\n    if (menuOwner && objectLike(parentContext) && contexts.get(parentContext)) menuParents.set(menuOwner, contexts.get(parentContext))\n    if (declaration && declaration[2] !== (objectLike(downstream) ? downstream : 0)) { state[1] = 0; faults |= 8 }\n  }\n  function link(callback, role, parent) {\n    if (!objectLike(callback) || !Number.isInteger(role) || role < 3 || role > 5) { faults |= 8; return }\n    const previous = links.get(callback), parentRef = objectLike(parent) ? parent : 0\n    if (previous && (previous[0] !== role || previous[1] !== parentRef)) { previous[0] = 0; faults |= 8; return }\n    links.set(callback, [role, parentRef])\n  }\n  function resolve(value) {\n    for (let depth = 0; depth < 4 && objectLike(value); depth++) {\n      const next = aliases.get(value) || callbackOwners.get(value)\n      if (!next) return value\n      value = next\n    }\n    return value\n  }\n  function role(value) {\n    value = resolve(value)\n    const state = objectLike(value) && owners.get(value)\n    if (state && state[1] === 3 && role(parentOwners.get(value)) !== 1) return 0\n    if (state && state[1] === 4 && role(menuParents.get(value)) !== 1) return 0\n    return state && state[4] && state[1] ? state[1] : 0\n  }\n  function target(value, node, marked) {\n    const state = objectLike(value) && owners.get(value)\n    if (!state) { faults |= 8; return }\n    const nodeId = id(node, 1), previousNode = state[4]\n    if (marked && nodeId && (state[1] === 1 || state[1] === 2)) {\n      let pair = rootNodes.get(node)\n      if (!pair) { pair = [0, 0]; rootNodes.set(node, pair) }\n      const old = pair[state[1] - 1]\n      if (old && old !== state[0] && activeOwners.get(old) === nodeId) { state[1] = 0; faults |= 8 }\n      else pair[state[1] - 1] = state[0]\n    }\n    if (state[4] !== nodeId) state[5]++\n    state[4] = marked || state[1] === 3 ? nodeId : 0\n    if (marked && !state[1]) faults |= 8\n    activeOwners.set(value, state[4])\n    if (previousNode || state[4]) row(1, id(value), state[1], state[5], nodeId, id(state[2]), id(state[3]), id(state[6]), id(state[7]), id(parentOwners.get(value) || menuParents.get(value)))\n  }\n  function observe(site, value, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0) {\n    value = resolve(value)\n    const state = objectLike(value) && owners.get(value)\n    if (state && role(value)) {\n      const index = row(site, id(value), state[1], state[5], a, b, c, d, e, f, origins.get(value) || activeOrigin)\n      if (site === 5 || site === 7 || site === 8 || site === 9 || site === 14 || site === 15) origins.set(value, index)\n      return index\n    }\n  }\n  function callback(site, callback, a = 0, b = 0) {\n    const data = objectLike(callback) && links.get(callback)\n    const ref = objectLike(callback) && callbackRefs.get(callback)\n    const owner = objectLike(ref) ? forwardOwners.get(ref) : callbackOwners.get(callback)\n    if (data && data[0] && role(owner)) row(site, id(callback), data[0], id(data[1]), a, b, id(owner))\n  }\n  function eventType(value) {\n    const result = value === \"pointerdown\" ? 1 : value === \"mousedown\" ? 2 : value === \"mouseup\" ? 3 : value === \"click\" ? 4 : value === \"pointerenter\" ? 5 : value === \"mouseenter\" ? 6 : value === \"pointerleave\" ? 7 : value === \"mouseleave\" ? 8 : value === \"contextmenu\" ? 9 : value === \"focus\" ? 10 : value === \"blur\" ? 11 : value === \"touchstart\" ? 12 : value === \"touchend\" ? 13 : value === \"pointerup\" ? 14 : value === \"touchcancel\" ? 15 : value === \"mousemove\" ? 16 : 0\n    if (!result) faults |= 2\n    return result\n  }\n  function event(site, value, event) {\n    if (!role(value) || !objectLike(event)) return\n    observe(site, value, id(event.nativeEvent || event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, typeof event.clientX === \"number\" ? event.clientX : -1, typeof event.clientY === \"number\" ? event.clientY : -1)\n  }\n  function native(site, key, node, result) {\n    if (!key || key.length !== 3 || !key.every(Number.isSafeInteger)) { faults |= 2; return }\n    row(site, key[0], key[1], key[2], id(node, 1), result)\n  }\n  function nativeEvent(key, node, event, result, fresh, preliminary = false) {\n    if (!key || !fresh) return\n    const index = row(51, key[0], key[1], key[2], id(node, 1), result, id(event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, event.clientX, event.clientY, typeof event.button === \"number\" ? event.button : -1, event.timeStamp)\n    if (result === 2 && preliminary && !first) first = index\n  }\n  function alias(value, parent) { if (objectLike(value) && objectLike(parent)) aliases.set(value, parent) }\n  function parent(openCallback, menuCallback) {\n    link(openCallback, 5, openCallback); link(menuCallback, 5, openCallback)\n    if (objectLike(openCallback)) parentCallbacks.set(openCallback, menuCallback)\n  }\n  function child(upstream, downstream, ref) {\n    const parent = objectLike(upstream) && links.get(upstream)\n    if (parent && parent[0] === 5) {\n      link(downstream, 5, ref); callbackRefs.set(upstream, ref); callbackRefs.set(downstream, ref)\n      const close = parentCallbacks.get(upstream); if (objectLike(close)) callbackRefs.set(close, ref)\n    }\n  }\n  function menu(value, callback) {\n    if (!owners.has(value)) { owner(value, null, callback, null); owners.get(value)[1] = 4 }\n    menuCallbacks.set(callback, value)\n    link(callback, 4, value)\n  }\n  function menuCommit(value, callback) {\n    const state = owners.get(value)\n    if (!state || !objectLike(value.current)) { faults |= 8; return }\n    target(value, value.current, true); callbackOwners.set(callback, value)\n  }\n  function submenu(callback, popupRef, parentCallback) {\n    const parent = objectLike(parentCallback) && links.get(parentCallback)\n    if (!parent || parent[0] !== 4) { faults |= 8; return }\n    link(callback, 3, parentCallback); exportParents.set(callback, parentCallback)\n  }\n  function popup(value, callback) { alias(value, callback) }\n  function bindPopup(value, node) { if (objectLike(node)) popupOwners.set(node, value) }\n  function timerRecord(site, serial) {\n    row(site, timerOwners[serial], timerRoles[serial], timerGenerations[serial], serial, timerKinds[serial], timerReturnTypes[serial], timerReturns[serial], timerDelays[serial], timerOrigins[serial])\n  }\n  function installed(ref, callback, kind, returned, delay) {\n    const value = resolve(callback), state = objectLike(value) && owners.get(value)\n    if (!state || !role(value)) return 0\n    if (kind !== 1 && kind !== 2 || !Number.isFinite(delay)) { faults |= 2; return 0 }\n    const type = typeof returned === \"number\" ? 1 : objectLike(returned) ? 2 : 0\n    if (!type || type === 1 && (!Number.isSafeInteger(returned) || returned < 0)) { faults |= 2; return 0 }\n    if (timerCount === 512) { faults |= 1; return 0 }\n    const old = objectLike(ref) && delaySerials.get(ref)\n    if (old && timers[old] === 1) faults |= 16\n    const serial = ++timerCount\n    timers[serial] = 1; timerOwners[serial] = id(value); timerRoles[serial] = state[1]; timerGenerations[serial] = state[5]\n    timerKinds[serial] = kind; timerReturnTypes[serial] = type; timerReturns[serial] = type === 1 ? returned : id(returned)\n    timerDelays[serial] = delay; timerOrigins[serial] = origins.get(value) || 0\n    if (objectLike(ref)) delaySerials.set(ref, serial)\n    timerRecord(10, serial)\n    return serial\n  }\n  function cleared(ref, kind) {\n    const serial = objectLike(ref) && delaySerials.get(ref)\n    if (!serial) { if (role(ref)) faults |= 16; return }\n    if (timerKinds[serial] !== kind) { faults |= 16; return }\n    if (timers[serial] === 1) { timers[serial] = 2; timerRecord(11, serial) }\n    else if (timers[serial] === 3) { timers[serial] = 4; timerRecord(31, serial) }\n    else { faults |= 16; return }\n    delaySerials.delete(ref)\n  }\n  function fired(ref, callback, serial, kind) {\n    if (!serial) { if (role(callback)) faults |= 16; return }\n    const value = resolve(callback)\n    if (!role(value) || timerOwners[serial] !== id(value) || timerKinds[serial] !== kind || timers[serial] !== 1) { faults |= 16; return }\n    timers[serial] = 3; origins.set(value, timerOrigins[serial]); timerRecord(12, serial)\n  }\n  function immediate(callback, phase) { observe(phase === 1 ? 29 : 30, callback, 0, phase) }\n  function reason(value, reason) {\n    if (reason !== 3 || reasons.get(value) !== 4) reasons.set(value, reason)\n    observe(19, value, reason)\n  }\n  function menuKeys(value, keys) {\n    observe(20, value, reasons.get(value) || 0, keys.length)\n    reasons.delete(value)\n  }\n  function geometry(popup, target, targetRect, popupRect, next) {\n    const value = objectLike(popup) && popupOwners.get(popup), state = value && owners.get(value)\n    if (!state || !role(value)) return\n    row(25, id(value), state[1], state[5], id(popup, 1), id(target, 1), targetRect.x, targetRect.y, targetRect.width, targetRect.height, popupRect.x, popupRect.y, popupRect.width, popupRect.height, next.offsetX, next.offsetY, next.offsetR, next.offsetB, next.arrowX, next.arrowY, next.scaleX, next.scaleY)\n  }\n  function hit(key, target, top, point) {\n    const tag = top?.nodeName\n    const descriptor = tag === \"TEXTAREA\" ? 2 : tag === \"BUTTON\" ? 3 : tag === \"LI\" ? 4 : tag === \"SPAN\" ? 5 : tag === \"DIV\" ? 6 : 7\n    row(54, key[0], key[1], key[2], id(target, 1), id(top, 1), descriptor, -1, point.x, point.y)\n  }\n  function drain() {\n    if (drained) throw new Error(\"already drained\")\n    drained = true\n    const rows = []\n    for (let index = 0; index < count; index++) rows.push(Array.from(tape.subarray(index * 24, (index + 1) * 24)))\n    return { schema: 43, realm, source, timeOrigin: origin, count, faults, first, counts: [objectCount, nodeCount, eventCount, timerCount], rows, order: 0 }\n  }\n  return Object.freeze({ source, realm, hit, alias, parent, child, menu, menuCommit, submenu, popup, bindPopup, installed, cleared, fired, immediate, reason, menuKeys, geometry, row, id, declare, owner, link, role, target, observe, callback, event, native, nativeEvent, drain, fault: flag => { faults |= flag }, origin: value => { activeOrigin = value }, node: value => id(value, 1) })\n})(4096,1,\"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\") }); } } catch { /* Unavailable observer. */ }"
const deferred43Patches = [
  {
    "sha256": "40ca22a328c30896466ff76a7501aa7c6ad24048afb4c413e120915d575b643c",
    "candidateHash": "ebb43c0f3d0b1d9994cdaeb0da2955c9661d087f1952ed14f0d797e294e16941",
    "edits": [
      {
        "from": "    const externalPopupRef = React.useRef(null);",
        "to": "    const externalPopupRef = React.useRef(null);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.owner?.value === \"function\"\n  ) {\n    try {\n      df43Observer.owner(\n        externalPopupRef,\n        ref,\n        onOpenChange,\n        onPopupVisibleChange,\n        parentContext,\n        context,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      externalPopupRef.current = node;",
        "to": "      externalPopupRef.current = node;\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\" &&\n    typeof df43Shape.node?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(2, externalPopupRef, df43Observer.node(node) || 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      const domNode = getDOM(node);",
        "to": "      const domNode = getDOM(node);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.target?.value === \"function\"\n  ) {\n    try {\n      df43Observer.target(\n        externalPopupRef,\n        domNode,\n        domNode?.getAttribute?.(\"data-testid\") === \"notes-overflow-menu-button\",\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    const internalTriggerOpen = useEvent(nextOpen => {",
        "to": "    const internalTriggerOpen = useEvent(nextOpen => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(\n        3,\n        externalPopupRef,\n        nextOpen ? 1 : 0,\n        mergedOpen ? 1 : 0,\n        mergedOpen !== nextOpen ? 1 : 0,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "          setInternalOpen(nextOpen);",
        "to": "          setInternalOpen(nextOpen);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(4, externalPopupRef, nextOpen ? 1 : 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    const delayInvoke = useDelay();",
        "to": "    const delayInvoke = useDelay();\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.alias?.value === \"function\"\n  ) {\n    try {\n      df43Observer.alias(delayInvoke, externalPopupRef)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    const triggerOpen = (nextOpen, delay = 0) => {",
        "to": "    const triggerOpen = (nextOpen, delay = 0) => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(5, externalPopupRef, nextOpen ? 1 : 0, delay)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    useLayoutEffect(firstMount => {",
        "to": "    useLayoutEffect(firstMount => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(6, externalPopupRef, mergedOpen ? 1 : 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      cloneProps[eventName] = (event, ...args) => {",
        "to": "      cloneProps[eventName] = (event, ...args) => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.event?.value === \"function\"\n  ) {\n    try {\n      df43Observer.event(7, externalPopupRef, event)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      onPopupMouseEnter = event => {",
        "to": "      onPopupMouseEnter = event => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.event?.value === \"function\"\n  ) {\n    try {\n      df43Observer.event(8, externalPopupRef, event)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      onPopupMouseLeave = () => {",
        "to": "      onPopupMouseLeave = () => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(9, externalPopupRef)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    function onEsc({",
        "to": "    function onEsc({\n// df43 source boundary",
        "count": 1
      },
      {
        "from": "    function onEsc({\n// df43 source boundary",
        "to": "try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.alias?.value === \"function\"\n  ) {\n    try {\n      df43Observer.alias(triggerOpen, externalPopupRef)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n    function onEsc({",
        "count": 1
      },
      {
        "from": "      externalPopupRef.current = node;",
        "to": "      externalPopupRef.current = node;\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.bindPopup?.value === \"function\"\n  ) {\n    try {\n      df43Observer.bindPopup(externalPopupRef, node)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "          onOpenChange?.(nextOpen);",
        "to": "          onOpenChange?.(nextOpen);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(27, externalPopupRef, nextOpen ? 1 : 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "          onPopupVisibleChange?.(nextOpen);",
        "to": "          onPopupVisibleChange?.(nextOpen);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(28, externalPopupRef, nextOpen ? 1 : 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      }
    ]
  },
  {
    "sha256": "e5b93ef5317033ab95ca76018ff8ea149c86b9f48bbc416c4f567530f043f1d9",
    "candidateHash": "81b5c3094f7f020a285ed54d8da23b53214f1f64223921835bda4aef2938e38e",
    "edits": [
      {
        "from": "import * as React from 'react';\nexport default function useDelay() {\n  const delayRef = React.useRef(null);\n  const clearDelay = () => {\n    if (delayRef.current) {\n      clearTimeout(delayRef.current);\n      delayRef.current = null;\n    }\n  };\n  const delayInvoke = (callback, delay) => {\n    clearDelay();\n    if (delay === 0) {\n      callback();\n    } else {\n      delayRef.current = setTimeout(() => {\n        callback();\n      }, delay * 1000);\n    }\n  };\n\n  // Clean up on unmount\n  React.useEffect(() => {\n    return () => {\n      clearDelay();\n    };\n  }, []);\n  return delayInvoke;\n}",
        "to": "import * as React from 'react';\nexport default function useDelay() {\n  const delayRef = React.useRef(null);\n  const clearDelay = () => {\n    if (delayRef.current) {\n      clearTimeout(delayRef.current);\n      try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.cleared?.value === \"function\"\n  ) {\n    try {\n      df43Observer.cleared(delayRef, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n      delayRef.current = null;\n    }\n  };\n  const delayInvoke = (callback, delay) => {\n    clearDelay();\n    if (delay === 0) {\n      try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.immediate?.value === \"function\"\n  ) {\n    try {\n      df43Observer.immediate(delayInvoke, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n      callback();\n      try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.immediate?.value === \"function\"\n  ) {\n    try {\n      df43Observer.immediate(delayInvoke, 2)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n    } else {\n      let df43Serial = 0;\n      delayRef.current = setTimeout(() => {\n        try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.fired?.value === \"function\"\n  ) {\n    try {\n      df43Observer.fired(delayRef, delayInvoke, df43Serial, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n        callback();\n      }, delay * 1000);\n      try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.installed?.value === \"function\"\n  ) {\n    try {\n      df43Serial =\n        df43Observer.installed(\n          delayRef,\n          delayInvoke,\n          1,\n          delayRef.current,\n          delay,\n        ) || 0\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n    }\n  };\n\n  // Clean up on unmount\n  React.useEffect(() => {\n    return () => {\n      clearDelay();\n    };\n  }, []);\n  return delayInvoke;\n}",
        "count": 1
      }
    ]
  },
  {
    "sha256": "6f89c7d63b5de65600b9204ee3e3272c9c2309f4bbd4575afa55a4a11f9626d5",
    "candidateHash": "f4510feb20e4ecadb5382caa4a0029529685df3d0135e0dac399c21f16711ac0",
    "edits": [
      {
        "from": "  const popupPointerDownRef = React.useRef(false);",
        "to": "  const popupPointerDownRef = React.useRef(false);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.alias?.value === \"function\"\n  ) {\n    try {\n      df43Observer.alias(popupPointerDownRef, triggerOpen)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "        popupPointerDownRef.current = false;",
        "to": "        popupPointerDownRef.current = false;\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(13, popupPointerDownRef, 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      const onTriggerClose = e => {",
        "to": "      const onTriggerClose = e => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.event?.value === \"function\"\n  ) {\n    try {\n      df43Observer.event(14, popupPointerDownRef, e)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "          triggerOpen(false);",
        "to": "          triggerOpen(false);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(15, popupPointerDownRef, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    popupPointerDownRef.current = true;",
        "to": "    popupPointerDownRef.current = true;\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(13, popupPointerDownRef, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "        if (openRef.current && !inPopupOrChild(e.composedPath?.()?.[0] || e.target) && !popupPointerDownRef.current) {",
        "to": "        let df43Open = -1, df43Inside = -1, df43Popup = -1;\n        if ((df43Open = openRef.current) && !(df43Inside = inPopupOrChild(e.composedPath?.()?.[0] || e.target)) && !(df43Popup = popupPointerDownRef.current)) {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(\n        16,\n        popupPointerDownRef,\n        df43Open ? 1 : 0,\n        df43Inside === -1 ? -1 : df43Inside ? 1 : 0,\n        df43Popup === -1 ? -1 : df43Popup ? 1 : 0,\n        1,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "          triggerOpen(false);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(15, popupPointerDownRef, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n        }",
        "to": "          triggerOpen(false);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(15, popupPointerDownRef, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n        } else {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.observe(\n        16,\n        popupPointerDownRef,\n        df43Open ? 1 : 0,\n        df43Inside === -1 ? -1 : df43Inside ? 1 : 0,\n        df43Popup === -1 ? -1 : df43Popup ? 1 : 0,\n        0,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n        }",
        "count": 1
      }
    ]
  },
  {
    "sha256": "101bddb95b2611c922e305204984c7d6ac8892f8d110e20d18970b31902777cb",
    "candidateHash": "279be574fa0b9a68f4c562b52e4971ca43da09be7297d98802efa7e907c26901",
    "edits": [
      {
        "from": "      setOffsetInfo(nextOffsetInfo);",
        "to": "try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.geometry?.value === \"function\"\n  ) {\n    try {\n      df43Observer.geometry(\n        popupElement,\n        target,\n        targetRect,\n        popupRect,\n        nextOffsetInfo,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n      setOffsetInfo(nextOffsetInfo);",
        "count": 1
      }
    ]
  },
  {
    "sha256": "eb471f0390b1ac43c8cab87832e4a6134010794e52c799dbdff0cbd302da9551",
    "candidateHash": "44f97a12f0016d05daa8c099bf35b3f5a9e409519d76d573dd0df25036b1629c",
    "edits": [
      {
        "from": "import * as React from 'react';\nimport Trigger from '@rc-component/trigger';\nimport { clsx } from 'clsx';\nimport raf from \"@rc-component/util/es/raf\";\nimport { MenuContext } from \"../context/MenuContext\";\nimport { placements, placementsRtl } from \"../placements\";\nimport { getMotion } from \"../utils/motionUtil\";\nconst popupPlacementMap = {\n  horizontal: 'bottomLeft',\n  vertical: 'rightTop',\n  'vertical-left': 'rightTop',\n  'vertical-right': 'leftTop'\n};\nexport default function PopupTrigger({\n  prefixCls,\n  visible,\n  children,\n  popup,\n  popupStyle,\n  popupClassName,\n  popupOffset,\n  disabled,\n  mode,\n  onVisibleChange\n}) {\n  const {\n    getPopupContainer,\n    rtl,\n    subMenuOpenDelay,\n    subMenuCloseDelay,\n    builtinPlacements,\n    triggerSubMenuAction,\n    forceSubMenuRender,\n    rootClassName,\n    // Motion\n    motion,\n    defaultMotions\n  } = React.useContext(MenuContext);\n  const [innerVisible, setInnerVisible] = React.useState(false);\n  const placement = rtl ? {\n    ...placementsRtl,\n    ...builtinPlacements\n  } : {\n    ...placements,\n    ...builtinPlacements\n  };\n  const popupPlacement = popupPlacementMap[mode];\n  const targetMotion = getMotion(mode, motion, defaultMotions);\n  const targetMotionRef = React.useRef(targetMotion);\n  if (mode !== 'inline') {\n    /**\n     * PopupTrigger is only used for vertical and horizontal types.\n     * When collapsed is unfolded, the inline animation will destroy the vertical animation.\n     */\n    targetMotionRef.current = targetMotion;\n  }\n  const mergedMotion = {\n    ...targetMotionRef.current,\n    leavedClassName: `${prefixCls}-hidden`,\n    removeOnLeave: false,\n    motionAppear: true\n  };\n\n  // Delay to change visible\n  const visibleRef = React.useRef();\n  React.useEffect(() => {\n    visibleRef.current = raf(() => {\n      setInnerVisible(visible);\n    });\n    return () => {\n      raf.cancel(visibleRef.current);\n    };\n  }, [visible]);\n  return /*#__PURE__*/React.createElement(Trigger, {\n    prefixCls: prefixCls,\n    popupClassName: clsx(`${prefixCls}-popup`, {\n      [`${prefixCls}-rtl`]: rtl\n    }, popupClassName, rootClassName),\n    stretch: mode === 'horizontal' ? 'minWidth' : null,\n    getPopupContainer: getPopupContainer,\n    builtinPlacements: placement,\n    popupPlacement: popupPlacement,\n    popupVisible: innerVisible,\n    popup: popup,\n    popupStyle: popupStyle,\n    popupAlign: popupOffset && {\n      offset: popupOffset\n    },\n    action: disabled ? [] : [triggerSubMenuAction],\n    mouseEnterDelay: subMenuOpenDelay,\n    mouseLeaveDelay: subMenuCloseDelay,\n    onPopupVisibleChange: onVisibleChange,\n    forceRender: forceSubMenuRender,\n    popupMotion: mergedMotion,\n    fresh: true\n  }, children);\n}",
        "to": "import * as React from 'react';\nimport Trigger from '@rc-component/trigger';\nimport { clsx } from 'clsx';\nimport raf from \"@rc-component/util/es/raf\";\nimport { MenuContext } from \"../context/MenuContext\";\nimport { placements, placementsRtl } from \"../placements\";\nimport { getMotion } from \"../utils/motionUtil\";\nconst popupPlacementMap = {\n  horizontal: 'bottomLeft',\n  vertical: 'rightTop',\n  'vertical-left': 'rightTop',\n  'vertical-right': 'leftTop'\n};\nexport default function PopupTrigger({\n  prefixCls,\n  visible,\n  children,\n  popup,\n  popupStyle,\n  popupClassName,\n  popupOffset,\n  disabled,\n  mode,\n  onVisibleChange\n}) {\n  const {\n    getPopupContainer,\n    rtl,\n    subMenuOpenDelay,\n    subMenuCloseDelay,\n    builtinPlacements,\n    triggerSubMenuAction,\n    forceSubMenuRender,\n    rootClassName,\n    // Motion\n    motion,\n    defaultMotions\n  } = React.useContext(MenuContext);\n  const [innerVisible, setInnerVisible] = React.useState(false);\n  const placement = rtl ? {\n    ...placementsRtl,\n    ...builtinPlacements\n  } : {\n    ...placements,\n    ...builtinPlacements\n  };\n  const popupPlacement = popupPlacementMap[mode];\n  const targetMotion = getMotion(mode, motion, defaultMotions);\n  const targetMotionRef = React.useRef(targetMotion);\n  if (mode !== 'inline') {\n    /**\n     * PopupTrigger is only used for vertical and horizontal types.\n     * When collapsed is unfolded, the inline animation will destroy the vertical animation.\n     */\n    targetMotionRef.current = targetMotion;\n  }\n  const mergedMotion = {\n    ...targetMotionRef.current,\n    leavedClassName: `${prefixCls}-hidden`,\n    removeOnLeave: false,\n    motionAppear: true\n  };\n\n  // Delay to change visible\n  const visibleRef = React.useRef();\n  try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.popup?.value === \"function\"\n  ) {\n    try {\n      df43Observer.popup(visibleRef, onVisibleChange)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n  React.useEffect(() => {\n    let df43Serial = 0;\n    visibleRef.current = raf(() => {\n      try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.fired?.value === \"function\" &&\n    typeof df43Shape.observe?.value === \"function\"\n  ) {\n    try {\n      df43Observer.fired(visibleRef, visibleRef, df43Serial, 2)\n      df43Observer.observe(\n        17,\n        visibleRef,\n        visible ? 1 : 0,\n        innerVisible ? 1 : 0,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n      setInnerVisible(visible);\n    });\n    try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.installed?.value === \"function\"\n  ) {\n    try {\n      df43Serial =\n        df43Observer.installed(\n          visibleRef,\n          visibleRef,\n          2,\n          visibleRef.current,\n          0,\n        ) || 0\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n    return () => {\n      raf.cancel(visibleRef.current);\n      try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.cleared?.value === \"function\"\n  ) {\n    try {\n      df43Observer.cleared(visibleRef, 2)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n    };\n  }, [visible]);\n  return /*#__PURE__*/React.createElement(Trigger, {\n    prefixCls: prefixCls,\n    popupClassName: clsx(`${prefixCls}-popup`, {\n      [`${prefixCls}-rtl`]: rtl\n    }, popupClassName, rootClassName),\n    stretch: mode === 'horizontal' ? 'minWidth' : null,\n    getPopupContainer: getPopupContainer,\n    builtinPlacements: placement,\n    popupPlacement: popupPlacement,\n    popupVisible: innerVisible,\n    popup: popup,\n    popupStyle: popupStyle,\n    popupAlign: popupOffset && {\n      offset: popupOffset\n    },\n    action: disabled ? [] : [triggerSubMenuAction],\n    mouseEnterDelay: subMenuOpenDelay,\n    mouseLeaveDelay: subMenuCloseDelay,\n    onPopupVisibleChange: onVisibleChange,\n    forceRender: forceSubMenuRender,\n    popupMotion: mergedMotion,\n    fresh: true\n  }, children);\n}",
        "count": 1
      }
    ]
  },
  {
    "sha256": "1eb6e0f39e3343c05e92f3d237d6fdf19b00a7fd80ed6f38ab4876c80219c5c4",
    "candidateHash": "5ee9b6071b084c76c8bc9699c785e712f523b7147549175fea5c00575d114b20",
    "edits": [
      {
        "from": "  const onPopupVisibleChange = newVisible => {",
        "to": "  const onPopupVisibleChange = newVisible => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.callback?.value === \"function\"\n  ) {\n    try {\n      df43Observer.callback(\n        18,\n        onPopupVisibleChange,\n        newVisible ? 1 : 0,\n        originOpen ? 1 : 0,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "  // =============================== Render ===============================",
        "to": "  // =============================== Render ===============================\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.submenu?.value === \"function\"\n  ) {\n    try {\n      if (eventKey === \"export-submenu\")\n        df43Observer.submenu(onPopupVisibleChange, popupRef, onOpenChange)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      }
    ]
  },
  {
    "sha256": "d3360b9af12eb4c0c553878960e0becfbd3f9e2691587c933134761f34246b22",
    "candidateHash": "52057727c3b158128e3a56346b739f0ba756bfe2c19717f8e3effa60800461b1",
    "edits": [
      {
        "from": "      setMergedOpenKeys(keys);",
        "to": "      setMergedOpenKeys(keys);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.menuKeys?.value === \"function\"\n  ) {\n    try {\n      df43Observer.menuKeys(containerRef, keys)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      // Trigger open event in case its in control\n      triggerOpenKeys(EMPTY_LIST);",
        "to": "      // Trigger open event in case its in control\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.reason?.value === \"function\"\n  ) {\n    try {\n      df43Observer.reason(containerRef, 1)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n      triggerOpenKeys(EMPTY_LIST);",
        "count": 1
      },
      {
        "from": "    if (!multiple && mergedOpenKeys.length && internalMode !== 'inline') {\n      triggerOpenKeys(EMPTY_LIST);",
        "to": "    if (!multiple && mergedOpenKeys.length && internalMode !== 'inline') {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.reason?.value === \"function\"\n  ) {\n    try {\n      df43Observer.reason(containerRef, 2)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n      triggerOpenKeys(EMPTY_LIST);",
        "count": 1
      },
      {
        "from": "  const onInternalOpenChange = useMemoCallback((key, open) => {",
        "to": "  const onInternalOpenChange = useMemoCallback((key, open) => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.reason?.value === \"function\"\n  ) {\n    try {\n      df43Observer.reason(containerRef, 3)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "  const triggerAccessibilityOpen = (key, open) => {",
        "to": "  const triggerAccessibilityOpen = (key, open) => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.reason?.value === \"function\"\n  ) {\n    try {\n      df43Observer.reason(containerRef, 4)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    setMounted(true);",
        "to": "    setMounted(true);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.menuCommit?.value === \"function\"\n  ) {\n    try {\n      df43Observer.menuCommit(containerRef, onInternalOpenChange)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "  // ======================= Context ========================",
        "to": "  // ======================= Context ========================\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.menu?.value === \"function\"\n  ) {\n    try {\n      df43Observer.menu(containerRef, onInternalOpenChange)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    onInternalOpenChange(key, nextOpen);",
        "to": "    onInternalOpenChange(key, nextOpen);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.reason?.value === \"function\"\n  ) {\n    try {\n      df43Observer.reason(containerRef, 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "      setMergedOpenKeys(inlineCacheOpenKeys);",
        "to": "      setMergedOpenKeys(inlineCacheOpenKeys);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.reason?.value === \"function\" &&\n    typeof df43Shape.menuKeys?.value === \"function\"\n  ) {\n    try {\n      df43Observer.reason(containerRef, 1)\n      df43Observer.menuKeys(containerRef, inlineCacheOpenKeys)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      }
    ]
  },
  {
    "sha256": "768889a22182941a21c17e9073f2119b4833f6c2824d794d956f609bbbe3663a",
    "candidateHash": "0675640bf049520510a369dabc0288d4749a4620d7400f4852ba711bb442fb7a",
    "edits": [
      {
        "from": "  const onInnerOpenChange = useEvent(nextOpen => {",
        "to": "  const onInnerOpenChange = useEvent(nextOpen => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.callback?.value === \"function\"\n  ) {\n    try {\n      df43Observer.callback(\n        21,\n        onInnerOpenChange,\n        nextOpen ? 1 : 0,\n        mergedOpen ? 1 : 0,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "    setOpen(false);",
        "to": "    setOpen(false);\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.callback?.value === \"function\"\n  ) {\n    try {\n      df43Observer.callback(22, onMenuClick, 0, mergedOpen ? 1 : 0)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "  let renderNode = /*#__PURE__*/React.createElement(RcDropdown, {",
        "to": "  let renderNode = /*#__PURE__*/React.createElement(RcDropdown, {\n// df43 source boundary",
        "count": 1
      },
      {
        "from": "  let renderNode = /*#__PURE__*/React.createElement(RcDropdown, {\n// df43 source boundary",
        "to": "try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.parent?.value === \"function\"\n  ) {\n    try {\n      df43Observer.parent(onInnerOpenChange, onMenuClick)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n  let renderNode = /*#__PURE__*/React.createElement(RcDropdown, {",
        "count": 1
      }
    ]
  },
  {
    "sha256": "9a4caff3ba328f928ad873b5aa856b7bce34923c7950503ae0e19320f45f06ea",
    "candidateHash": "a0fe32091feefd9b24ffc9fe44b43104f5d646005d2adab00aed0685b9779e0b",
    "edits": [
      {
        "from": "  const handleVisibleChange = newVisible => {",
        "to": "  const handleVisibleChange = newVisible => {\ntry {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.callback?.value === \"function\"\n  ) {\n    try {\n      df43Observer.callback(\n        23,\n        handleVisibleChange,\n        newVisible ? 1 : 0,\n        mergedVisible ? 1 : 0,\n      )\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      },
      {
        "from": "  return /*#__PURE__*/React.createElement(Trigger, _extends({",
        "to": "try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.declare?.value === \"function\" &&\n    typeof df43Shape.child?.value === \"function\"\n  ) {\n    try {\n      df43Observer.declare(1, triggerRef, onVisibleChange, handleVisibleChange)\n      df43Observer.child(onVisibleChange, handleVisibleChange, triggerRef)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n  return /*#__PURE__*/React.createElement(Trigger, _extends({",
        "count": 1
      }
    ]
  },
  {
    "sha256": "5015d8c15f096d4ac2ab023b73fe51f78cb80501a504dfa72b9bbb5c44e7645a",
    "candidateHash": "21dce203daa13d12fec76dc1230af8a9c0a50646a190f8fc7e6c30ee98a2bf36",
    "edits": [
      {
        "from": "  return /*#__PURE__*/React.createElement(Trigger, _extends({",
        "to": "try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.declare?.value === \"function\"\n  ) {\n    try {\n      df43Observer.declare(2, triggerRef, null, onVisibleChange)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}\n  return /*#__PURE__*/React.createElement(Trigger, _extends({",
        "count": 1
      }
    ]
  },
  {
    "sha256": "ba8315311db5df7dc9c5b42abc4a58a7293a6746b6e274c3360e56fefa53feee",
    "candidateHash": "e4551fee267bc900c8c85e2bcf88141cfc1d1afe5efe9adb395cc89abd589c9a",
    "edits": [
      {
        "from": "      case 'export-print':",
        "to": "      case 'export-print':\n        try {\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43App\")\n  const df43Observer = df43Entry?.value\n  const df43Shape =\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\n  if (\n    df43Entry &&\n    df43Entry.writable === false &&\n    df43Entry.configurable === false &&\n    Object.isFrozen(df43Observer) &&\n    df43Shape?.source?.value ===\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\n    df43Shape.realm?.value === 1 &&\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\n    typeof df43Shape.fault?.value === \"function\" &&\n    typeof df43Shape.row?.value === \"function\"\n  ) {\n    try {\n      df43Observer.row(24)\n    } catch {\n      try {\n        df43Observer.fault(4)\n      } catch {\n        /* Observer failure cannot change product behavior. */\n      }\n    }\n  }\n} catch {\n  /* Unavailable observer. */\n}",
        "count": 1
      }
    ]
  },
  {
    "sha256": "33f502d6319a4e2539a4020c4983045f30d3d1cbed5d7610a32d9f89cbf8823d",
    "candidateHash": "90d5420fce037454c3d64d6b24e13a030882c1abeae2036251cdf8c35b4e1a21",
    "edits": [
      {
        "from": "\"use strict\";",
        "to": "\"use strict\";\n\n// Temporary deferred ledger belongs to this actual owning server process.\nconst df43Server = (() => {\n  try {\n    const fs = require(\"node:fs\"), path = require(\"node:path\"), crypto = require(\"node:crypto\");\n    const recorder = (function deferred43Recorder(capacity, realm, source) {\n  const tape = new Float64Array(capacity * 24)\n  const objects = new WeakMap(), nodes = new WeakMap(), events = new WeakMap()\n  const declarations = new WeakMap(), owners = new WeakMap(), links = new WeakMap()\n  const forwardOwners = new WeakMap(), callbackRefs = new WeakMap(), parentCallbacks = new WeakMap(), exportParents = new WeakMap(), menuParents = new WeakMap(), menuCallbacks = new WeakMap(), aliases = new WeakMap(), callbackOwners = new WeakMap(), contexts = new WeakMap(), parentOwners = new WeakMap(), popupOwners = new WeakMap(), delaySerials = new WeakMap(), reasons = new WeakMap()\n  const origins = new WeakMap(), rootNodes = new WeakMap(), activeOwners = new WeakMap(), timerOrigins = new Uint32Array(513)\n  const timers = new Uint32Array(513), timerOwners = new Uint32Array(513), timerKinds = new Uint32Array(513), timerRoles = new Uint32Array(513), timerGenerations = new Uint32Array(513), timerReturnTypes = new Uint32Array(513), timerReturns = new Float64Array(513), timerDelays = new Float64Array(513)\n  let objectCount = 0, nodeCount = 0, eventCount = 0, timerCount = 0\n  let count = 0, faults = 0, first = 0, drained = false, activeOrigin = 0\n  const allowed = new Uint32Array(71)\n  for (const site of [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,40,41,42,43,44,45,46,50,51,52,53,54,55,56]) allowed[site] = 1\n  const origin = performance.timeOrigin\n  const objectLike = value => value !== null && (typeof value === \"object\" || typeof value === \"function\")\n  function id(value, kind = 0) {\n    if (!objectLike(value)) return 0\n    const map = kind === 1 ? nodes : kind === 2 ? events : objects\n    let result = map.get(value)\n    if (result) return result\n    const current = kind === 1 ? nodeCount : kind === 2 ? eventCount : objectCount\n    if (current >= (kind === 0 ? 512 : 256)) { faults |= 1; return 0 }\n    result = current + 1\n    map.set(value, result)\n    if (kind === 1) nodeCount = result\n    else if (kind === 2) eventCount = result\n    else objectCount = result\n    return result\n  }\n  function row(site, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0, t = 0, u = 0) {\n    try {\n      if (!Number.isInteger(site) || site < 1 || site > 70 || !allowed[site] || arguments.length > 22) { faults |= 2; return 0 }\n      for (let index = 1; index < arguments.length; index++) {\n        if (typeof arguments[index] !== \"number\" || !Number.isFinite(arguments[index])) { faults |= 2; return 0 }\n      }\n      if (drained || count === capacity) { faults |= 1; return 0 }\n      const offset = count++ * 24\n      tape[offset] = site; tape[offset + 1] = count; tape[offset + 2] = performance.now()\n      tape[offset + 3] = a; tape[offset + 4] = b; tape[offset + 5] = c\n      tape[offset + 6] = d; tape[offset + 7] = e; tape[offset + 8] = f\n      tape[offset + 9] = g; tape[offset + 10] = h; tape[offset + 11] = i\n      tape[offset + 12] = j; tape[offset + 13] = k; tape[offset + 14] = l\n      tape[offset + 15] = m; tape[offset + 16] = n; tape[offset + 17] = o\n      tape[offset + 18] = p; tape[offset + 19] = q; tape[offset + 20] = r\n      tape[offset + 21] = s; tape[offset + 22] = t; tape[offset + 23] = u\n      return count\n    } catch { faults |= 4; return 0 }\n  }\n  function declare(role, ref, upstream, downstream) {\n    if (!objectLike(ref) || (role !== 1 && role !== 2)) { faults |= 8; return }\n    const old = declarations.get(ref)\n    if (old && old[0] !== role) { old[0] = 0; faults |= 8; return }\n    const value = old || [role, 0, 0]\n    value[1] = objectLike(upstream) ? upstream : 0; value[2] = objectLike(downstream) ? downstream : 0\n    declarations.set(ref, value)\n  }\n  function owner(value, ref, downstream, popupChange, parentContext, context) {\n    if (objectLike(ref)) forwardOwners.set(ref, value)\n    if (objectLike(context)) contexts.set(context, value)\n    if (objectLike(parentContext)) parentOwners.set(value, contexts.get(parentContext))\n    if (objectLike(popupChange)) callbackOwners.set(popupChange, value)\n    if (!objectLike(value)) { faults |= 8; return }\n    let state = owners.get(value)\n    if (!state) { state = [value, 0, 0, 0, 0, 0, 0, 0]; owners.set(value, state) }\n    const declaration = objectLike(ref) && declarations.get(ref)\n    const link = objectLike(popupChange) && links.get(popupChange)\n    const nextRole = declaration ? declaration[0] : link ? link[0] : 0\n    if (state[2] !== (objectLike(ref) ? ref : 0) || state[1] !== nextRole) { state[4] = 0; state[5]++ }\n    state[1] = nextRole\n    state[2] = objectLike(ref) ? ref : 0; state[3] = objectLike(downstream) ? downstream : 0; state[6] = objectLike(popupChange) ? popupChange : 0\n    state[7] = link ? link[1] : 0\n    const menuCallback = objectLike(popupChange) && exportParents.get(popupChange)\n    const menuOwner = objectLike(menuCallback) && menuCallbacks.get(menuCallback)\n    if (menuOwner && objectLike(parentContext) && contexts.get(parentContext)) menuParents.set(menuOwner, contexts.get(parentContext))\n    if (declaration && declaration[2] !== (objectLike(downstream) ? downstream : 0)) { state[1] = 0; faults |= 8 }\n  }\n  function link(callback, role, parent) {\n    if (!objectLike(callback) || !Number.isInteger(role) || role < 3 || role > 5) { faults |= 8; return }\n    const previous = links.get(callback), parentRef = objectLike(parent) ? parent : 0\n    if (previous && (previous[0] !== role || previous[1] !== parentRef)) { previous[0] = 0; faults |= 8; return }\n    links.set(callback, [role, parentRef])\n  }\n  function resolve(value) {\n    for (let depth = 0; depth < 4 && objectLike(value); depth++) {\n      const next = aliases.get(value) || callbackOwners.get(value)\n      if (!next) return value\n      value = next\n    }\n    return value\n  }\n  function role(value) {\n    value = resolve(value)\n    const state = objectLike(value) && owners.get(value)\n    if (state && state[1] === 3 && role(parentOwners.get(value)) !== 1) return 0\n    if (state && state[1] === 4 && role(menuParents.get(value)) !== 1) return 0\n    return state && state[4] && state[1] ? state[1] : 0\n  }\n  function target(value, node, marked) {\n    const state = objectLike(value) && owners.get(value)\n    if (!state) { faults |= 8; return }\n    const nodeId = id(node, 1), previousNode = state[4]\n    if (marked && nodeId && (state[1] === 1 || state[1] === 2)) {\n      let pair = rootNodes.get(node)\n      if (!pair) { pair = [0, 0]; rootNodes.set(node, pair) }\n      const old = pair[state[1] - 1]\n      if (old && old !== state[0] && activeOwners.get(old) === nodeId) { state[1] = 0; faults |= 8 }\n      else pair[state[1] - 1] = state[0]\n    }\n    if (state[4] !== nodeId) state[5]++\n    state[4] = marked || state[1] === 3 ? nodeId : 0\n    if (marked && !state[1]) faults |= 8\n    activeOwners.set(value, state[4])\n    if (previousNode || state[4]) row(1, id(value), state[1], state[5], nodeId, id(state[2]), id(state[3]), id(state[6]), id(state[7]), id(parentOwners.get(value) || menuParents.get(value)))\n  }\n  function observe(site, value, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0) {\n    value = resolve(value)\n    const state = objectLike(value) && owners.get(value)\n    if (state && role(value)) {\n      const index = row(site, id(value), state[1], state[5], a, b, c, d, e, f, origins.get(value) || activeOrigin)\n      if (site === 5 || site === 7 || site === 8 || site === 9 || site === 14 || site === 15) origins.set(value, index)\n      return index\n    }\n  }\n  function callback(site, callback, a = 0, b = 0) {\n    const data = objectLike(callback) && links.get(callback)\n    const ref = objectLike(callback) && callbackRefs.get(callback)\n    const owner = objectLike(ref) ? forwardOwners.get(ref) : callbackOwners.get(callback)\n    if (data && data[0] && role(owner)) row(site, id(callback), data[0], id(data[1]), a, b, id(owner))\n  }\n  function eventType(value) {\n    const result = value === \"pointerdown\" ? 1 : value === \"mousedown\" ? 2 : value === \"mouseup\" ? 3 : value === \"click\" ? 4 : value === \"pointerenter\" ? 5 : value === \"mouseenter\" ? 6 : value === \"pointerleave\" ? 7 : value === \"mouseleave\" ? 8 : value === \"contextmenu\" ? 9 : value === \"focus\" ? 10 : value === \"blur\" ? 11 : value === \"touchstart\" ? 12 : value === \"touchend\" ? 13 : value === \"pointerup\" ? 14 : value === \"touchcancel\" ? 15 : value === \"mousemove\" ? 16 : 0\n    if (!result) faults |= 2\n    return result\n  }\n  function event(site, value, event) {\n    if (!role(value) || !objectLike(event)) return\n    observe(site, value, id(event.nativeEvent || event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, typeof event.clientX === \"number\" ? event.clientX : -1, typeof event.clientY === \"number\" ? event.clientY : -1)\n  }\n  function native(site, key, node, result) {\n    if (!key || key.length !== 3 || !key.every(Number.isSafeInteger)) { faults |= 2; return }\n    row(site, key[0], key[1], key[2], id(node, 1), result)\n  }\n  function nativeEvent(key, node, event, result, fresh, preliminary = false) {\n    if (!key || !fresh) return\n    const index = row(51, key[0], key[1], key[2], id(node, 1), result, id(event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, event.clientX, event.clientY, typeof event.button === \"number\" ? event.button : -1, event.timeStamp)\n    if (result === 2 && preliminary && !first) first = index\n  }\n  function alias(value, parent) { if (objectLike(value) && objectLike(parent)) aliases.set(value, parent) }\n  function parent(openCallback, menuCallback) {\n    link(openCallback, 5, openCallback); link(menuCallback, 5, openCallback)\n    if (objectLike(openCallback)) parentCallbacks.set(openCallback, menuCallback)\n  }\n  function child(upstream, downstream, ref) {\n    const parent = objectLike(upstream) && links.get(upstream)\n    if (parent && parent[0] === 5) {\n      link(downstream, 5, ref); callbackRefs.set(upstream, ref); callbackRefs.set(downstream, ref)\n      const close = parentCallbacks.get(upstream); if (objectLike(close)) callbackRefs.set(close, ref)\n    }\n  }\n  function menu(value, callback) {\n    if (!owners.has(value)) { owner(value, null, callback, null); owners.get(value)[1] = 4 }\n    menuCallbacks.set(callback, value)\n    link(callback, 4, value)\n  }\n  function menuCommit(value, callback) {\n    const state = owners.get(value)\n    if (!state || !objectLike(value.current)) { faults |= 8; return }\n    target(value, value.current, true); callbackOwners.set(callback, value)\n  }\n  function submenu(callback, popupRef, parentCallback) {\n    const parent = objectLike(parentCallback) && links.get(parentCallback)\n    if (!parent || parent[0] !== 4) { faults |= 8; return }\n    link(callback, 3, parentCallback); exportParents.set(callback, parentCallback)\n  }\n  function popup(value, callback) { alias(value, callback) }\n  function bindPopup(value, node) { if (objectLike(node)) popupOwners.set(node, value) }\n  function timerRecord(site, serial) {\n    row(site, timerOwners[serial], timerRoles[serial], timerGenerations[serial], serial, timerKinds[serial], timerReturnTypes[serial], timerReturns[serial], timerDelays[serial], timerOrigins[serial])\n  }\n  function installed(ref, callback, kind, returned, delay) {\n    const value = resolve(callback), state = objectLike(value) && owners.get(value)\n    if (!state || !role(value)) return 0\n    if (kind !== 1 && kind !== 2 || !Number.isFinite(delay)) { faults |= 2; return 0 }\n    const type = typeof returned === \"number\" ? 1 : objectLike(returned) ? 2 : 0\n    if (!type || type === 1 && (!Number.isSafeInteger(returned) || returned < 0)) { faults |= 2; return 0 }\n    if (timerCount === 512) { faults |= 1; return 0 }\n    const old = objectLike(ref) && delaySerials.get(ref)\n    if (old && timers[old] === 1) faults |= 16\n    const serial = ++timerCount\n    timers[serial] = 1; timerOwners[serial] = id(value); timerRoles[serial] = state[1]; timerGenerations[serial] = state[5]\n    timerKinds[serial] = kind; timerReturnTypes[serial] = type; timerReturns[serial] = type === 1 ? returned : id(returned)\n    timerDelays[serial] = delay; timerOrigins[serial] = origins.get(value) || 0\n    if (objectLike(ref)) delaySerials.set(ref, serial)\n    timerRecord(10, serial)\n    return serial\n  }\n  function cleared(ref, kind) {\n    const serial = objectLike(ref) && delaySerials.get(ref)\n    if (!serial) { if (role(ref)) faults |= 16; return }\n    if (timerKinds[serial] !== kind) { faults |= 16; return }\n    if (timers[serial] === 1) { timers[serial] = 2; timerRecord(11, serial) }\n    else if (timers[serial] === 3) { timers[serial] = 4; timerRecord(31, serial) }\n    else { faults |= 16; return }\n    delaySerials.delete(ref)\n  }\n  function fired(ref, callback, serial, kind) {\n    if (!serial) { if (role(callback)) faults |= 16; return }\n    const value = resolve(callback)\n    if (!role(value) || timerOwners[serial] !== id(value) || timerKinds[serial] !== kind || timers[serial] !== 1) { faults |= 16; return }\n    timers[serial] = 3; origins.set(value, timerOrigins[serial]); timerRecord(12, serial)\n  }\n  function immediate(callback, phase) { observe(phase === 1 ? 29 : 30, callback, 0, phase) }\n  function reason(value, reason) {\n    if (reason !== 3 || reasons.get(value) !== 4) reasons.set(value, reason)\n    observe(19, value, reason)\n  }\n  function menuKeys(value, keys) {\n    observe(20, value, reasons.get(value) || 0, keys.length)\n    reasons.delete(value)\n  }\n  function geometry(popup, target, targetRect, popupRect, next) {\n    const value = objectLike(popup) && popupOwners.get(popup), state = value && owners.get(value)\n    if (!state || !role(value)) return\n    row(25, id(value), state[1], state[5], id(popup, 1), id(target, 1), targetRect.x, targetRect.y, targetRect.width, targetRect.height, popupRect.x, popupRect.y, popupRect.width, popupRect.height, next.offsetX, next.offsetY, next.offsetR, next.offsetB, next.arrowX, next.arrowY, next.scaleX, next.scaleY)\n  }\n  function hit(key, target, top, point) {\n    const tag = top?.nodeName\n    const descriptor = tag === \"TEXTAREA\" ? 2 : tag === \"BUTTON\" ? 3 : tag === \"LI\" ? 4 : tag === \"SPAN\" ? 5 : tag === \"DIV\" ? 6 : 7\n    row(54, key[0], key[1], key[2], id(target, 1), id(top, 1), descriptor, -1, point.x, point.y)\n  }\n  function drain() {\n    if (drained) throw new Error(\"already drained\")\n    drained = true\n    const rows = []\n    for (let index = 0; index < count; index++) rows.push(Array.from(tape.subarray(index * 24, (index + 1) * 24)))\n    return { schema: 43, realm, source, timeOrigin: origin, count, faults, first, counts: [objectCount, nodeCount, eventCount, timerCount], rows, order: 0 }\n  }\n  return { source, realm, hit, alias, parent, child, menu, menuCommit, submenu, popup, bindPopup, installed, cleared, fired, immediate, reason, menuKeys, geometry, row, id, declare, owner, link, role, target, observe, callback, event, native, nativeEvent, drain, fault: flag => { faults |= flag }, origin: value => { activeOrigin = value }, node: value => id(value, 1) }\n})(1024,3,\"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\");\n    let calls = 0, browserMetadata = null, metadataAttempted = false;\n    const select = (progress, handle, action, retry) => {\n      try {\n        if (process.env.TLDW_DEFERRED43_SOURCE !== \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" || (action !== \"click\" && action !== \"hover\") || progress.metadata.params?.selector !== \"internal:role=menuitem[name=/print \\\\/ save as pdf/i]\") return null;\n        const text = progress.metadata.id;\n        if (typeof text !== \"string\" || !/^call@\\d+$/.test(text) || !Number.isSafeInteger(retry)) { recorder.fault(2); return null; }\n        const call = Number(text.slice(5));\n        if (!Number.isSafeInteger(call) || call < 1 || retry < 0 || ++calls > 512) { recorder.fault(1); return null; }\n        const key = [call,recorder.id(handle),retry];\n        if (!metadataAttempted) {\n          metadataAttempted = true;\n          try {\n            const browser = handle._page.browserContext._browser;\n            const child = browser.options.browserProcess.process;\n            const executable = child.spawnfile, version = browser._version;\n            const raw = child.spawnargs;\n            if (typeof executable !== \"string\" || Buffer.byteLength(executable) > 128 || typeof version !== \"string\" || Buffer.byteLength(version) > 128 || !Number.isSafeInteger(child.pid) || child.pid < 1 || !Array.isArray(raw) || raw.length > 64) { recorder.fault(2048); }\n            else {\n              const flags = raw.slice(1).filter(value => typeof value === \"string\" && (/^--(?:headless(?:=(?:new|old))?|enable-automation|no-sandbox|disable-dev-shm-usage)$/.test(value) || /^--window-size=\\d{1,5},\\d{1,5}$/.test(value) || /^--force-device-scale-factor=\\d{1,2}(?:\\.\\d{1,4})?$/.test(value)));\n              browserMetadata = { pid: child.pid, executable, version, flags, omittedArgs: raw.length - 1 - flags.length, binary: null, readback: 0 };\n            }\n          } catch { recorder.fault(2048); }\n        }\n        recorder.row(40,key[0],key[1],key[2],recorder.id(handle._frame));\n        return key;\n      } catch { recorder.fault(4); return null; }\n    };\n    const result = value => value === \"done\" ? 1 : value === \"error:notconnected\" ? 3 : value === \"error:notvisible\" ? 4 : value === \"error:notinviewport\" ? 5 : value?.missingState === \"stable\" ? 6 : value?.missingState ? 7 : value && typeof value === \"object\" && \"hitTargetDescription\" in value ? 2 : 0;\n    process.once(\"exit\", () => {\n      try {\n        if (!calls) return;\n        const reporter = process.env.PLAYWRIGHT_JSON_OUTPUT_NAME;\n        if (typeof reporter !== \"string\") return;\n        const parent = fs.realpathSync(path.dirname(reporter));\n        const cwd = fs.realpathSync(process.cwd());\n        if (!parent.startsWith(path.join(cwd,\"test-results\",\"live-tier-uat\") + path.sep) || path.basename(reporter) !== \"playwright-results.json\") return;\n        if (browserMetadata) {\n          try {\n            const stat = fs.statSync(browserMetadata.executable);\n            if (!stat.isFile() || stat.size > 536870912) recorder.fault(2048);\n            else { browserMetadata.binary = crypto.createHash(\"sha256\").update(fs.readFileSync(browserMetadata.executable)).digest(\"hex\"); browserMetadata.readback = 1; }\n          } catch { recorder.fault(2048); }\n        } else recorder.fault(2048);\n        const payload = recorder.drain();\n        const bytes = JSON.stringify({ ...payload, browser: browserMetadata, pid: process.pid, platform: process.platform, version: process.version, binary: crypto.createHash(\"sha256\").update(fs.readFileSync(process.execPath)).digest(\"hex\") });\n        if (Buffer.byteLength(bytes) > 262144) return;\n        fs.writeFileSync(path.join(parent,\"deferred43-server-\"+process.pid+\".json\"),bytes,{flag:\"wx\",mode:0o600});\n      } catch { /* Missing exit receipt is SERVER_LEDGER_LOST, never a native result. */ }\n    });\n    return { recorder, select, result, returned(key, site, value) { try { if(key)recorder.row(site,key[0],key[1],key[2],result(value)); } catch { recorder.fault(4); } return value; } };\n  } catch { return null; }\n})();\n",
        "count": 1
      },
      {
        "from": "return await this._performPointerAction(progress, actionName, waitForEnabled, action, forceScrollOptions, options);",
        "to": "return await this._performPointerAction(progress, actionName, waitForEnabled, action, forceScrollOptions, options, retry);",
        "count": 1
      },
      {
        "from": "async _performPointerAction(progress, actionName, waitForEnabled, action, forceScrollOptions, options) {",
        "to": "async _performPointerAction(progress, actionName, waitForEnabled, action, forceScrollOptions, options, df43Retry) {\n    const df43Key = df43Server?.select(progress, this, actionName, df43Retry);",
        "count": 1
      },
      {
        "from": "      const result = await progress.race(this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2 }]) => {\n        return await injected.checkElementStates(node, elementStates2);\n      }, { elementStates }));",
        "to": "      const result = await progress.race(df43Key ? this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2, df43Key }]) => {\n        return await injected.checkElementStates(node, elementStates2, df43Key);\n      }, { elementStates, df43Key }) : this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2 }]) => {\n        return await injected.checkElementStates(node, elementStates2);\n      }, { elementStates }));",
        "count": 1
      },
      {
        "from": "      const handle = await progress.race(this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial), { actionType, hitPoint, trial: !!options.trial }));",
        "to": "      const handle = await progress.race(df43Key ? this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial, df43Key }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial, df43Key), { actionType, hitPoint, trial: !!options.trial, df43Key }) : this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial), { actionType, hitPoint, trial: !!options.trial }));",
        "count": 1
      },
      {
        "from": "    const scrolled = await progress.race(doScrollIntoView());",
        "to": "    const scrolled = await progress.race(doScrollIntoView());\n    try { if (df43Key) df43Server.recorder.row(42, df43Key[0], df43Key[1], df43Key[2], df43Server.result(scrolled)); } catch { df43Server?.recorder.fault(4); }",
        "count": 1
      },
      {
        "from": "    const point = roundPoint(maybePoint);",
        "to": "    const point = roundPoint(maybePoint);\n    try { if (df43Key) df43Server.recorder.row(43, df43Key[0], df43Key[1], df43Key[2], point.x, point.y); } catch { df43Server?.recorder.fault(4); }",
        "count": 1
      },
      {
        "from": "      const frameCheckResult = await progress.race(this._checkFrameIsHitTarget(point));",
        "to": "      const frameCheckResult = await progress.race(this._checkFrameIsHitTarget(point));\n      try { if (df43Key) df43Server.recorder.row(44, df43Key[0], df43Key[1], df43Key[2], df43Server.result(frameCheckResult), frameCheckResult.framePoint?.x ?? -1, frameCheckResult.framePoint?.y ?? -1); } catch { df43Server?.recorder.fault(4); }",
        "count": 1
      },
      {
        "from": "      await action(point);",
        "to": "      await action(point);\n      try { if (df43Key) df43Server.recorder.row(45, df43Key[0], df43Key[1], df43Key[2]); } catch { df43Server?.recorder.fault(4); }",
        "count": 1
      },
      {
        "from": "async _clickablePoint() {",
        "to": "async _clickablePoint(df43Key) {",
        "count": 1
      },
      {
        "from": "await progress.race(this._clickablePoint());",
        "to": "await progress.race(this._clickablePoint(df43Key));",
        "count": 1
      },
      {
        "from": "    if (quads === \"error:notconnected\")",
        "to": "    try { if (df43Key) {\n      if (Array.isArray(quads) && quads.length > 4) df43Server.recorder.fault(128);\n      if (Array.isArray(quads)) for (let i = 0; i < Math.min(quads.length,4); i++) { const q = quads[i]; df43Server.recorder.row(41,df43Key[0],df43Key[1],df43Key[2],i,metrics.width,metrics.height,q[0].x,q[0].y,q[1].x,q[1].y,q[2].x,q[2].y,q[3].x,q[3].y); }\n    } } catch { df43Server?.recorder.fault(4); }\n    if (quads === \"error:notconnected\")",
        "count": 1
      },
      {
        "from": "  async _performPointerAction(progress, actionName, waitForEnabled, action, forceScrollOptions, options, df43Retry) {\n    const df43Key = df43Server?.select(progress, this, actionName, df43Retry);\n    const { force = false, position } = options;\n    const doScrollIntoView = async () => {\n      if (forceScrollOptions) {\n        return await this.evaluateInUtility(([injected, node, options2]) => {\n          if (node.nodeType === 1)\n            node.scrollIntoView(options2);\n          return \"done\";\n        }, forceScrollOptions);\n      }\n      return await this._scrollRectIntoViewIfNeeded(progress, position ? { x: position.x, y: position.y, width: 0, height: 0 } : void 0);\n    };\n    if (this._frame.parentFrame()) {\n      await progress.race(doScrollIntoView().catch(() => {\n      }));\n    }\n    if (options.__testHookBeforeStable)\n      await progress.race(options.__testHookBeforeStable());\n    if (!force) {\n      const elementStates = waitForEnabled ? [\"visible\", \"enabled\", \"stable\"] : [\"visible\", \"stable\"];\n      progress.log(`  waiting for element to be ${waitForEnabled ? \"visible, enabled and stable\" : \"visible and stable\"}`);\n      const result = await progress.race(df43Key ? this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2, df43Key }]) => {\n        return await injected.checkElementStates(node, elementStates2, df43Key);\n      }, { elementStates, df43Key }) : this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2 }]) => {\n        return await injected.checkElementStates(node, elementStates2);\n      }, { elementStates }));\n      if (result)\n        return result;\n      progress.log(`  element is ${waitForEnabled ? \"visible, enabled and stable\" : \"visible and stable\"}`);\n    }\n    if (options.__testHookAfterStable)\n      await progress.race(options.__testHookAfterStable());\n    progress.log(\"  scrolling into view if needed\");\n    const scrolled = await progress.race(doScrollIntoView());\n    try { if (df43Key) df43Server.recorder.row(42, df43Key[0], df43Key[1], df43Key[2], df43Server.result(scrolled)); } catch { df43Server?.recorder.fault(4); }\n    if (scrolled !== \"done\")\n      return scrolled;\n    progress.log(\"  done scrolling\");\n    const maybePoint = position ? await progress.race(this._offsetPoint(position)) : await progress.race(this._clickablePoint(df43Key));\n    if (typeof maybePoint === \"string\")\n      return maybePoint;\n    const point = roundPoint(maybePoint);\n    try { if (df43Key) df43Server.recorder.row(43, df43Key[0], df43Key[1], df43Key[2], point.x, point.y); } catch { df43Server?.recorder.fault(4); }\n    progress.metadata.point = point;\n    await progress.race(this.instrumentation.onBeforeInputAction(this, progress.metadata));\n    let hitTargetInterceptionHandle;\n    if (force) {\n      progress.log(`  forcing action`);\n    } else {\n      if (options.__testHookBeforeHitTarget)\n        await progress.race(options.__testHookBeforeHitTarget());\n      const frameCheckResult = await progress.race(this._checkFrameIsHitTarget(point));\n      try { if (df43Key) df43Server.recorder.row(44, df43Key[0], df43Key[1], df43Key[2], df43Server.result(frameCheckResult), frameCheckResult.framePoint?.x ?? -1, frameCheckResult.framePoint?.y ?? -1); } catch { df43Server?.recorder.fault(4); }\n      if (frameCheckResult === \"error:notconnected\" || \"hitTargetDescription\" in frameCheckResult)\n        return frameCheckResult;\n      const hitPoint = frameCheckResult.framePoint;\n      const actionType = actionName === \"move and up\" ? \"drag\" : actionName === \"hover\" || actionName === \"tap\" ? actionName : \"mouse\";\n      const handle = await progress.race(df43Key ? this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial, df43Key }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial, df43Key), { actionType, hitPoint, trial: !!options.trial, df43Key }) : this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial), { actionType, hitPoint, trial: !!options.trial }));\n      if (handle === \"error:notconnected\")\n        return handle;\n      if (!handle._objectId) {\n        const error = handle.rawValue();\n        if (error === \"error:notconnected\")\n          return error;\n        return { hitTargetDescription: error };\n      }\n      hitTargetInterceptionHandle = handle;\n    }\n    const actionResult = await this._page.frameManager.waitForSignalsCreatedBy(progress, options.waitAfter === true, async () => {\n      if (options.__testHookBeforePointerAction)\n        await progress.race(options.__testHookBeforePointerAction());\n      let restoreModifiers;\n      if (options && options.modifiers)\n        restoreModifiers = await this._page.keyboard.ensureModifiers(progress, options.modifiers);\n      progress.log(`  performing ${actionName} action`);\n      await action(point);\n      try { if (df43Key) df43Server.recorder.row(45, df43Key[0], df43Key[1], df43Key[2]); } catch { df43Server?.recorder.fault(4); }\n      if (restoreModifiers)\n        await this._page.keyboard.ensureModifiers(progress, restoreModifiers);\n      if (hitTargetInterceptionHandle) {\n        const stopHitTargetInterception = this._frame.raceAgainstEvaluationStallingEvents(() => {\n          return hitTargetInterceptionHandle.evaluate((h) => h.stop());\n        }).catch((e) => \"done\").finally(() => {\n          hitTargetInterceptionHandle?.dispose();\n        });\n        if (options.waitAfter !== false) {\n          const hitTargetResult = await progress.race(stopHitTargetInterception);\n          if (hitTargetResult !== \"done\")\n            return hitTargetResult;\n        }\n      }\n      progress.log(`  ${options.trial ? \"trial \" : \"\"}${actionName} action done`);\n      progress.log(\"  waiting for scheduled navigations to finish\");\n      if (options.__testHookAfterPointerAction)\n        await progress.race(options.__testHookAfterPointerAction());\n      return \"done\";\n    }).finally(() => {\n      const stopPromise = hitTargetInterceptionHandle?.evaluate((h) => h.stop()).catch(() => {\n      });\n      stopPromise?.then(() => hitTargetInterceptionHandle?.dispose());\n    });\n    if (actionResult !== \"done\")\n      return actionResult;\n    progress.log(\"  navigations have finished\");\n    return \"done\";\n  }\n",
        "to": "  async _performPointerAction(progress, actionName, waitForEnabled, action, forceScrollOptions, options, df43Retry) {\n    const df43Key = df43Server?.select(progress, this, actionName, df43Retry);\n    const { force = false, position } = options;\n    const doScrollIntoView = async () => {\n      if (forceScrollOptions) {\n        return await this.evaluateInUtility(([injected, node, options2]) => {\n          if (node.nodeType === 1)\n            node.scrollIntoView(options2);\n          return \"done\";\n        }, forceScrollOptions);\n      }\n      return await this._scrollRectIntoViewIfNeeded(progress, position ? { x: position.x, y: position.y, width: 0, height: 0 } : void 0);\n    };\n    if (this._frame.parentFrame()) {\n      await progress.race(doScrollIntoView().catch(() => {\n      }));\n    }\n    if (options.__testHookBeforeStable)\n      await progress.race(options.__testHookBeforeStable());\n    if (!force) {\n      const elementStates = waitForEnabled ? [\"visible\", \"enabled\", \"stable\"] : [\"visible\", \"stable\"];\n      progress.log(`  waiting for element to be ${waitForEnabled ? \"visible, enabled and stable\" : \"visible and stable\"}`);\n      const result = await progress.race(df43Key ? this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2, df43Key }]) => {\n        return await injected.checkElementStates(node, elementStates2, df43Key);\n      }, { elementStates, df43Key }) : this.evaluateInUtility(async ([injected, node, { elementStates: elementStates2 }]) => {\n        return await injected.checkElementStates(node, elementStates2);\n      }, { elementStates }));\n      if (result)\n        return df43Server ? df43Server.returned(df43Key,46,result) : result;\n      progress.log(`  element is ${waitForEnabled ? \"visible, enabled and stable\" : \"visible and stable\"}`);\n    }\n    if (options.__testHookAfterStable)\n      await progress.race(options.__testHookAfterStable());\n    progress.log(\"  scrolling into view if needed\");\n    const scrolled = await progress.race(doScrollIntoView());\n    try { if (df43Key) df43Server.recorder.row(42, df43Key[0], df43Key[1], df43Key[2], df43Server.result(scrolled)); } catch { df43Server?.recorder.fault(4); }\n    if (scrolled !== \"done\")\n      return df43Server ? df43Server.returned(df43Key,46,scrolled) : scrolled;\n    progress.log(\"  done scrolling\");\n    const maybePoint = position ? await progress.race(this._offsetPoint(position)) : await progress.race(this._clickablePoint(df43Key));\n    if (typeof maybePoint === \"string\")\n      return df43Server ? df43Server.returned(df43Key,46,maybePoint) : maybePoint;\n    const point = roundPoint(maybePoint);\n    try { if (df43Key) df43Server.recorder.row(43, df43Key[0], df43Key[1], df43Key[2], point.x, point.y); } catch { df43Server?.recorder.fault(4); }\n    progress.metadata.point = point;\n    await progress.race(this.instrumentation.onBeforeInputAction(this, progress.metadata));\n    let hitTargetInterceptionHandle;\n    if (force) {\n      progress.log(`  forcing action`);\n    } else {\n      if (options.__testHookBeforeHitTarget)\n        await progress.race(options.__testHookBeforeHitTarget());\n      const frameCheckResult = await progress.race(this._checkFrameIsHitTarget(point));\n      try { if (df43Key) df43Server.recorder.row(44, df43Key[0], df43Key[1], df43Key[2], df43Server.result(frameCheckResult), frameCheckResult.framePoint?.x ?? -1, frameCheckResult.framePoint?.y ?? -1); } catch { df43Server?.recorder.fault(4); }\n      if (frameCheckResult === \"error:notconnected\" || \"hitTargetDescription\" in frameCheckResult)\n        return df43Server ? df43Server.returned(df43Key,46,frameCheckResult) : frameCheckResult;\n      const hitPoint = frameCheckResult.framePoint;\n      const actionType = actionName === \"move and up\" ? \"drag\" : actionName === \"hover\" || actionName === \"tap\" ? actionName : \"mouse\";\n      const handle = await progress.race(df43Key ? this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial, df43Key }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial, df43Key), { actionType, hitPoint, trial: !!options.trial, df43Key }) : this.evaluateHandleInUtility(([injected, node, { actionType: actionType2, hitPoint: hitPoint2, trial }]) => injected.setupHitTargetInterceptor(node, actionType2, hitPoint2, trial), { actionType, hitPoint, trial: !!options.trial }));\n      if (handle === \"error:notconnected\")\n        return df43Server ? df43Server.returned(df43Key,46,handle) : handle;\n      if (!handle._objectId) {\n        const error = handle.rawValue();\n        if (error === \"error:notconnected\")\n          return df43Server ? df43Server.returned(df43Key,46,error) : error;\n        return df43Server ? df43Server.returned(df43Key,46,{ hitTargetDescription: error }) : { hitTargetDescription: error };\n      }\n      hitTargetInterceptionHandle = handle;\n    }\n    const actionResult = await this._page.frameManager.waitForSignalsCreatedBy(progress, options.waitAfter === true, async () => {\n      if (options.__testHookBeforePointerAction)\n        await progress.race(options.__testHookBeforePointerAction());\n      let restoreModifiers;\n      if (options && options.modifiers)\n        restoreModifiers = await this._page.keyboard.ensureModifiers(progress, options.modifiers);\n      progress.log(`  performing ${actionName} action`);\n      await action(point);\n      try { if (df43Key) df43Server.recorder.row(45, df43Key[0], df43Key[1], df43Key[2]); } catch { df43Server?.recorder.fault(4); }\n      if (restoreModifiers)\n        await this._page.keyboard.ensureModifiers(progress, restoreModifiers);\n      if (hitTargetInterceptionHandle) {\n        const stopHitTargetInterception = this._frame.raceAgainstEvaluationStallingEvents(() => {\n          return hitTargetInterceptionHandle.evaluate((h) => h.stop());\n        }).catch((e) => \"done\").finally(() => {\n          hitTargetInterceptionHandle?.dispose();\n        });\n        if (options.waitAfter !== false) {\n          const hitTargetResult = await progress.race(stopHitTargetInterception);\n          if (hitTargetResult !== \"done\")\n            return df43Server ? df43Server.returned(df43Key,46,hitTargetResult) : hitTargetResult;\n        }\n      }\n      progress.log(`  ${options.trial ? \"trial \" : \"\"}${actionName} action done`);\n      progress.log(\"  waiting for scheduled navigations to finish\");\n      if (options.__testHookAfterPointerAction)\n        await progress.race(options.__testHookAfterPointerAction());\n      return df43Server ? df43Server.returned(df43Key,46,\"done\") : \"done\";\n    }).finally(() => {\n      const stopPromise = hitTargetInterceptionHandle?.evaluate((h) => h.stop()).catch(() => {\n      });\n      stopPromise?.then(() => hitTargetInterceptionHandle?.dispose());\n    });\n    if (actionResult !== \"done\")\n      return df43Server ? df43Server.returned(df43Key,46,actionResult) : actionResult;\n    progress.log(\"  navigations have finished\");\n    return df43Server ? df43Server.returned(df43Key,46,\"done\") : \"done\";\n  }\n",
        "count": 1
      }
    ]
  },
  {
    "sha256": "ea0889cc1b0efba61511dd640e658ca965aa8d5ae20eef8e608fd651b85390ca",
    "candidateHash": "6f5a45e6dfe810ff74bd0c10e780bed81be8c2f54b3f6309d0c2daaed930af1b",
    "edits": [
      {
        "from": "async checkElementStates(node, states) {",
        "to": "async checkElementStates(node, states, df43Key) {\\n    try { if (df43Key) { let df43Owner = globalThis; while (df43Owner && !Object.getOwnPropertyDescriptor(df43Owner, \"__df43Utility\")) df43Owner = Object.getPrototypeOf(df43Owner); if (!df43Owner) Object.defineProperty(globalThis, \"__df43Utility\", { value: (function deferred43Recorder(capacity, realm, source) {\\n  const tape = new Float64Array(capacity * 24)\\n  const objects = new WeakMap(), nodes = new WeakMap(), events = new WeakMap()\\n  const declarations = new WeakMap(), owners = new WeakMap(), links = new WeakMap()\\n  const forwardOwners = new WeakMap(), callbackRefs = new WeakMap(), parentCallbacks = new WeakMap(), exportParents = new WeakMap(), menuParents = new WeakMap(), menuCallbacks = new WeakMap(), aliases = new WeakMap(), callbackOwners = new WeakMap(), contexts = new WeakMap(), parentOwners = new WeakMap(), popupOwners = new WeakMap(), delaySerials = new WeakMap(), reasons = new WeakMap()\\n  const origins = new WeakMap(), rootNodes = new WeakMap(), activeOwners = new WeakMap(), timerOrigins = new Uint32Array(513)\\n  const timers = new Uint32Array(513), timerOwners = new Uint32Array(513), timerKinds = new Uint32Array(513), timerRoles = new Uint32Array(513), timerGenerations = new Uint32Array(513), timerReturnTypes = new Uint32Array(513), timerReturns = new Float64Array(513), timerDelays = new Float64Array(513)\\n  let objectCount = 0, nodeCount = 0, eventCount = 0, timerCount = 0\\n  let count = 0, faults = 0, first = 0, drained = false, activeOrigin = 0\\n  const allowed = new Uint32Array(71)\\n  for (const site of [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,40,41,42,43,44,45,46,50,51,52,53,54,55,56]) allowed[site] = 1\\n  const origin = performance.timeOrigin\\n  const objectLike = value => value !== null && (typeof value === \"object\" || typeof value === \"function\")\\n  function id(value, kind = 0) {\\n    if (!objectLike(value)) return 0\\n    const map = kind === 1 ? nodes : kind === 2 ? events : objects\\n    let result = map.get(value)\\n    if (result) return result\\n    const current = kind === 1 ? nodeCount : kind === 2 ? eventCount : objectCount\\n    if (current >= (kind === 0 ? 512 : 256)) { faults |= 1; return 0 }\\n    result = current + 1\\n    map.set(value, result)\\n    if (kind === 1) nodeCount = result\\n    else if (kind === 2) eventCount = result\\n    else objectCount = result\\n    return result\\n  }\\n  function row(site, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0, t = 0, u = 0) {\\n    try {\\n      if (!Number.isInteger(site) || site < 1 || site > 70 || !allowed[site] || arguments.length > 22) { faults |= 2; return 0 }\\n      for (let index = 1; index < arguments.length; index++) {\\n        if (typeof arguments[index] !== \"number\" || !Number.isFinite(arguments[index])) { faults |= 2; return 0 }\\n      }\\n      if (drained || count === capacity) { faults |= 1; return 0 }\\n      const offset = count++ * 24\\n      tape[offset] = site; tape[offset + 1] = count; tape[offset + 2] = performance.now()\\n      tape[offset + 3] = a; tape[offset + 4] = b; tape[offset + 5] = c\\n      tape[offset + 6] = d; tape[offset + 7] = e; tape[offset + 8] = f\\n      tape[offset + 9] = g; tape[offset + 10] = h; tape[offset + 11] = i\\n      tape[offset + 12] = j; tape[offset + 13] = k; tape[offset + 14] = l\\n      tape[offset + 15] = m; tape[offset + 16] = n; tape[offset + 17] = o\\n      tape[offset + 18] = p; tape[offset + 19] = q; tape[offset + 20] = r\\n      tape[offset + 21] = s; tape[offset + 22] = t; tape[offset + 23] = u\\n      return count\\n    } catch { faults |= 4; return 0 }\\n  }\\n  function declare(role, ref, upstream, downstream) {\\n    if (!objectLike(ref) || (role !== 1 && role !== 2)) { faults |= 8; return }\\n    const old = declarations.get(ref)\\n    if (old && old[0] !== role) { old[0] = 0; faults |= 8; return }\\n    const value = old || [role, 0, 0]\\n    value[1] = objectLike(upstream) ? upstream : 0; value[2] = objectLike(downstream) ? downstream : 0\\n    declarations.set(ref, value)\\n  }\\n  function owner(value, ref, downstream, popupChange, parentContext, context) {\\n    if (objectLike(ref)) forwardOwners.set(ref, value)\\n    if (objectLike(context)) contexts.set(context, value)\\n    if (objectLike(parentContext)) parentOwners.set(value, contexts.get(parentContext))\\n    if (objectLike(popupChange)) callbackOwners.set(popupChange, value)\\n    if (!objectLike(value)) { faults |= 8; return }\\n    let state = owners.get(value)\\n    if (!state) { state = [value, 0, 0, 0, 0, 0, 0, 0]; owners.set(value, state) }\\n    const declaration = objectLike(ref) && declarations.get(ref)\\n    const link = objectLike(popupChange) && links.get(popupChange)\\n    const nextRole = declaration ? declaration[0] : link ? link[0] : 0\\n    if (state[2] !== (objectLike(ref) ? ref : 0) || state[1] !== nextRole) { state[4] = 0; state[5]++ }\\n    state[1] = nextRole\\n    state[2] = objectLike(ref) ? ref : 0; state[3] = objectLike(downstream) ? downstream : 0; state[6] = objectLike(popupChange) ? popupChange : 0\\n    state[7] = link ? link[1] : 0\\n    const menuCallback = objectLike(popupChange) && exportParents.get(popupChange)\\n    const menuOwner = objectLike(menuCallback) && menuCallbacks.get(menuCallback)\\n    if (menuOwner && objectLike(parentContext) && contexts.get(parentContext)) menuParents.set(menuOwner, contexts.get(parentContext))\\n    if (declaration && declaration[2] !== (objectLike(downstream) ? downstream : 0)) { state[1] = 0; faults |= 8 }\\n  }\\n  function link(callback, role, parent) {\\n    if (!objectLike(callback) || !Number.isInteger(role) || role < 3 || role > 5) { faults |= 8; return }\\n    const previous = links.get(callback), parentRef = objectLike(parent) ? parent : 0\\n    if (previous && (previous[0] !== role || previous[1] !== parentRef)) { previous[0] = 0; faults |= 8; return }\\n    links.set(callback, [role, parentRef])\\n  }\\n  function resolve(value) {\\n    for (let depth = 0; depth < 4 && objectLike(value); depth++) {\\n      const next = aliases.get(value) || callbackOwners.get(value)\\n      if (!next) return value\\n      value = next\\n    }\\n    return value\\n  }\\n  function role(value) {\\n    value = resolve(value)\\n    const state = objectLike(value) && owners.get(value)\\n    if (state && state[1] === 3 && role(parentOwners.get(value)) !== 1) return 0\\n    if (state && state[1] === 4 && role(menuParents.get(value)) !== 1) return 0\\n    return state && state[4] && state[1] ? state[1] : 0\\n  }\\n  function target(value, node, marked) {\\n    const state = objectLike(value) && owners.get(value)\\n    if (!state) { faults |= 8; return }\\n    const nodeId = id(node, 1), previousNode = state[4]\\n    if (marked && nodeId && (state[1] === 1 || state[1] === 2)) {\\n      let pair = rootNodes.get(node)\\n      if (!pair) { pair = [0, 0]; rootNodes.set(node, pair) }\\n      const old = pair[state[1] - 1]\\n      if (old && old !== state[0] && activeOwners.get(old) === nodeId) { state[1] = 0; faults |= 8 }\\n      else pair[state[1] - 1] = state[0]\\n    }\\n    if (state[4] !== nodeId) state[5]++\\n    state[4] = marked || state[1] === 3 ? nodeId : 0\\n    if (marked && !state[1]) faults |= 8\\n    activeOwners.set(value, state[4])\\n    if (previousNode || state[4]) row(1, id(value), state[1], state[5], nodeId, id(state[2]), id(state[3]), id(state[6]), id(state[7]), id(parentOwners.get(value) || menuParents.get(value)))\\n  }\\n  function observe(site, value, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0) {\\n    value = resolve(value)\\n    const state = objectLike(value) && owners.get(value)\\n    if (state && role(value)) {\\n      const index = row(site, id(value), state[1], state[5], a, b, c, d, e, f, origins.get(value) || activeOrigin)\\n      if (site === 5 || site === 7 || site === 8 || site === 9 || site === 14 || site === 15) origins.set(value, index)\\n      return index\\n    }\\n  }\\n  function callback(site, callback, a = 0, b = 0) {\\n    const data = objectLike(callback) && links.get(callback)\\n    const ref = objectLike(callback) && callbackRefs.get(callback)\\n    const owner = objectLike(ref) ? forwardOwners.get(ref) : callbackOwners.get(callback)\\n    if (data && data[0] && role(owner)) row(site, id(callback), data[0], id(data[1]), a, b, id(owner))\\n  }\\n  function eventType(value) {\\n    const result = value === \"pointerdown\" ? 1 : value === \"mousedown\" ? 2 : value === \"mouseup\" ? 3 : value === \"click\" ? 4 : value === \"pointerenter\" ? 5 : value === \"mouseenter\" ? 6 : value === \"pointerleave\" ? 7 : value === \"mouseleave\" ? 8 : value === \"contextmenu\" ? 9 : value === \"focus\" ? 10 : value === \"blur\" ? 11 : value === \"touchstart\" ? 12 : value === \"touchend\" ? 13 : value === \"pointerup\" ? 14 : value === \"touchcancel\" ? 15 : value === \"mousemove\" ? 16 : 0\\n    if (!result) faults |= 2\\n    return result\\n  }\\n  function event(site, value, event) {\\n    if (!role(value) || !objectLike(event)) return\\n    observe(site, value, id(event.nativeEvent || event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, typeof event.clientX === \"number\" ? event.clientX : -1, typeof event.clientY === \"number\" ? event.clientY : -1)\\n  }\\n  function native(site, key, node, result) {\\n    if (!key || key.length !== 3 || !key.every(Number.isSafeInteger)) { faults |= 2; return }\\n    row(site, key[0], key[1], key[2], id(node, 1), result)\\n  }\\n  function nativeEvent(key, node, event, result, fresh, preliminary = false) {\\n    if (!key || !fresh) return\\n    const index = row(51, key[0], key[1], key[2], id(node, 1), result, id(event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, event.clientX, event.clientY, typeof event.button === \"number\" ? event.button : -1, event.timeStamp)\\n    if (result === 2 && preliminary && !first) first = index\\n  }\\n  function alias(value, parent) { if (objectLike(value) && objectLike(parent)) aliases.set(value, parent) }\\n  function parent(openCallback, menuCallback) {\\n    link(openCallback, 5, openCallback); link(menuCallback, 5, openCallback)\\n    if (objectLike(openCallback)) parentCallbacks.set(openCallback, menuCallback)\\n  }\\n  function child(upstream, downstream, ref) {\\n    const parent = objectLike(upstream) && links.get(upstream)\\n    if (parent && parent[0] === 5) {\\n      link(downstream, 5, ref); callbackRefs.set(upstream, ref); callbackRefs.set(downstream, ref)\\n      const close = parentCallbacks.get(upstream); if (objectLike(close)) callbackRefs.set(close, ref)\\n    }\\n  }\\n  function menu(value, callback) {\\n    if (!owners.has(value)) { owner(value, null, callback, null); owners.get(value)[1] = 4 }\\n    menuCallbacks.set(callback, value)\\n    link(callback, 4, value)\\n  }\\n  function menuCommit(value, callback) {\\n    const state = owners.get(value)\\n    if (!state || !objectLike(value.current)) { faults |= 8; return }\\n    target(value, value.current, true); callbackOwners.set(callback, value)\\n  }\\n  function submenu(callback, popupRef, parentCallback) {\\n    const parent = objectLike(parentCallback) && links.get(parentCallback)\\n    if (!parent || parent[0] !== 4) { faults |= 8; return }\\n    link(callback, 3, parentCallback); exportParents.set(callback, parentCallback)\\n  }\\n  function popup(value, callback) { alias(value, callback) }\\n  function bindPopup(value, node) { if (objectLike(node)) popupOwners.set(node, value) }\\n  function timerRecord(site, serial) {\\n    row(site, timerOwners[serial], timerRoles[serial], timerGenerations[serial], serial, timerKinds[serial], timerReturnTypes[serial], timerReturns[serial], timerDelays[serial], timerOrigins[serial])\\n  }\\n  function installed(ref, callback, kind, returned, delay) {\\n    const value = resolve(callback), state = objectLike(value) && owners.get(value)\\n    if (!state || !role(value)) return 0\\n    if (kind !== 1 && kind !== 2 || !Number.isFinite(delay)) { faults |= 2; return 0 }\\n    const type = typeof returned === \"number\" ? 1 : objectLike(returned) ? 2 : 0\\n    if (!type || type === 1 && (!Number.isSafeInteger(returned) || returned < 0)) { faults |= 2; return 0 }\\n    if (timerCount === 512) { faults |= 1; return 0 }\\n    const old = objectLike(ref) && delaySerials.get(ref)\\n    if (old && timers[old] === 1) faults |= 16\\n    const serial = ++timerCount\\n    timers[serial] = 1; timerOwners[serial] = id(value); timerRoles[serial] = state[1]; timerGenerations[serial] = state[5]\\n    timerKinds[serial] = kind; timerReturnTypes[serial] = type; timerReturns[serial] = type === 1 ? returned : id(returned)\\n    timerDelays[serial] = delay; timerOrigins[serial] = origins.get(value) || 0\\n    if (objectLike(ref)) delaySerials.set(ref, serial)\\n    timerRecord(10, serial)\\n    return serial\\n  }\\n  function cleared(ref, kind) {\\n    const serial = objectLike(ref) && delaySerials.get(ref)\\n    if (!serial) { if (role(ref)) faults |= 16; return }\\n    if (timerKinds[serial] !== kind) { faults |= 16; return }\\n    if (timers[serial] === 1) { timers[serial] = 2; timerRecord(11, serial) }\\n    else if (timers[serial] === 3) { timers[serial] = 4; timerRecord(31, serial) }\\n    else { faults |= 16; return }\\n    delaySerials.delete(ref)\\n  }\\n  function fired(ref, callback, serial, kind) {\\n    if (!serial) { if (role(callback)) faults |= 16; return }\\n    const value = resolve(callback)\\n    if (!role(value) || timerOwners[serial] !== id(value) || timerKinds[serial] !== kind || timers[serial] !== 1) { faults |= 16; return }\\n    timers[serial] = 3; origins.set(value, timerOrigins[serial]); timerRecord(12, serial)\\n  }\\n  function immediate(callback, phase) { observe(phase === 1 ? 29 : 30, callback, 0, phase) }\\n  function reason(value, reason) {\\n    if (reason !== 3 || reasons.get(value) !== 4) reasons.set(value, reason)\\n    observe(19, value, reason)\\n  }\\n  function menuKeys(value, keys) {\\n    observe(20, value, reasons.get(value) || 0, keys.length)\\n    reasons.delete(value)\\n  }\\n  function geometry(popup, target, targetRect, popupRect, next) {\\n    const value = objectLike(popup) && popupOwners.get(popup), state = value && owners.get(value)\\n    if (!state || !role(value)) return\\n    row(25, id(value), state[1], state[5], id(popup, 1), id(target, 1), targetRect.x, targetRect.y, targetRect.width, targetRect.height, popupRect.x, popupRect.y, popupRect.width, popupRect.height, next.offsetX, next.offsetY, next.offsetR, next.offsetB, next.arrowX, next.arrowY, next.scaleX, next.scaleY)\\n  }\\n  function hit(key, target, top, point) {\\n    const tag = top?.nodeName\\n    const descriptor = tag === \"TEXTAREA\" ? 2 : tag === \"BUTTON\" ? 3 : tag === \"LI\" ? 4 : tag === \"SPAN\" ? 5 : tag === \"DIV\" ? 6 : 7\\n    row(54, key[0], key[1], key[2], id(target, 1), id(top, 1), descriptor, -1, point.x, point.y)\\n  }\\n  function drain() {\\n    if (drained) throw new Error(\"already drained\")\\n    drained = true\\n    const rows = []\\n    for (let index = 0; index < count; index++) rows.push(Array.from(tape.subarray(index * 24, (index + 1) * 24)))\\n    return { schema: 43, realm, source, timeOrigin: origin, count, faults, first, counts: [objectCount, nodeCount, eventCount, timerCount], rows, order: 0 }\\n  }\\n  return Object.freeze({ source, realm, hit, alias, parent, child, menu, menuCommit, submenu, popup, bindPopup, installed, cleared, fired, immediate, reason, menuKeys, geometry, row, id, declare, owner, link, role, target, observe, callback, event, native, nativeEvent, drain, fault: flag => { faults |= flag }, origin: value => { activeOrigin = value }, node: value => id(value, 1) })\\n})(2048,2,\"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\") }); } } catch { /* Unavailable observer. */ }",
        "count": 1
      },
      {
        "from": "const stableResult = await this._checkElementIsStable(node);",
        "to": "const stableResult = await this._checkElementIsStable(node, df43Key);",
        "count": 1
      },
      {
        "from": "      if (stableResult === false)\\n        return { missingState: \"stable\" };",
        "to": "      if (stableResult === false) {\\n        try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.native?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key) df43Observer.native(56, df43Key, node, 6)\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}\\n        return { missingState: \"stable\" };\\n      }",
        "count": 1
      },
      {
        "from": "async _checkElementIsStable(node) {",
        "to": "async _checkElementIsStable(node, df43Key) {\\n    let df43Rects = 0;",
        "count": 1
      },
      {
        "from": "      const rect = { x: clientRect.top, y: clientRect.left, width: clientRect.width, height: clientRect.height };",
        "to": "      const rect = { x: clientRect.top, y: clientRect.left, width: clientRect.width, height: clientRect.height };\\n      try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.row?.value === \"function\" &&\\n    typeof df43Shape.node?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key) {\\n        if (++df43Rects <= 16)\\n          df43Observer.row(\\n            55,\\n            df43Key[0],\\n            df43Key[1],\\n            df43Key[2],\\n            df43Observer.node(element),\\n            rect.x,\\n            rect.y,\\n            rect.width,\\n            rect.height,\\n          )\\n        else df43Observer.fault(64)\\n      }\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}",
        "count": 1
      },
      {
        "from": "expectHitTarget(hitPoint, targetElement) {",
        "to": "expectHitTarget(hitPoint, targetElement, df43Key) {",
        "count": 1
      },
      {
        "from": "    const hitParents = [];",
        "to": "    try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.hit?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key)\\n        df43Observer.hit(df43Key, targetElement, hitElement, hitPoint)\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}\\n    const hitParents = [];",
        "count": 1
      },
      {
        "from": "setupHitTargetInterceptor(node, action, hitPoint, blockAllEvents) {",
        "to": "setupHitTargetInterceptor(node, action, hitPoint, blockAllEvents, df43Key) {\\n    let df43PreliminaryDone = false;\\n    try { if (df43Key) { let df43Owner = globalThis; while (df43Owner && !Object.getOwnPropertyDescriptor(df43Owner, \"__df43Utility\")) df43Owner = Object.getPrototypeOf(df43Owner); if (!df43Owner) Object.defineProperty(globalThis, \"__df43Utility\", { value: (function deferred43Recorder(capacity, realm, source) {\\n  const tape = new Float64Array(capacity * 24)\\n  const objects = new WeakMap(), nodes = new WeakMap(), events = new WeakMap()\\n  const declarations = new WeakMap(), owners = new WeakMap(), links = new WeakMap()\\n  const forwardOwners = new WeakMap(), callbackRefs = new WeakMap(), parentCallbacks = new WeakMap(), exportParents = new WeakMap(), menuParents = new WeakMap(), menuCallbacks = new WeakMap(), aliases = new WeakMap(), callbackOwners = new WeakMap(), contexts = new WeakMap(), parentOwners = new WeakMap(), popupOwners = new WeakMap(), delaySerials = new WeakMap(), reasons = new WeakMap()\\n  const origins = new WeakMap(), rootNodes = new WeakMap(), activeOwners = new WeakMap(), timerOrigins = new Uint32Array(513)\\n  const timers = new Uint32Array(513), timerOwners = new Uint32Array(513), timerKinds = new Uint32Array(513), timerRoles = new Uint32Array(513), timerGenerations = new Uint32Array(513), timerReturnTypes = new Uint32Array(513), timerReturns = new Float64Array(513), timerDelays = new Float64Array(513)\\n  let objectCount = 0, nodeCount = 0, eventCount = 0, timerCount = 0\\n  let count = 0, faults = 0, first = 0, drained = false, activeOrigin = 0\\n  const allowed = new Uint32Array(71)\\n  for (const site of [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,29,30,31,40,41,42,43,44,45,46,50,51,52,53,54,55,56]) allowed[site] = 1\\n  const origin = performance.timeOrigin\\n  const objectLike = value => value !== null && (typeof value === \"object\" || typeof value === \"function\")\\n  function id(value, kind = 0) {\\n    if (!objectLike(value)) return 0\\n    const map = kind === 1 ? nodes : kind === 2 ? events : objects\\n    let result = map.get(value)\\n    if (result) return result\\n    const current = kind === 1 ? nodeCount : kind === 2 ? eventCount : objectCount\\n    if (current >= (kind === 0 ? 512 : 256)) { faults |= 1; return 0 }\\n    result = current + 1\\n    map.set(value, result)\\n    if (kind === 1) nodeCount = result\\n    else if (kind === 2) eventCount = result\\n    else objectCount = result\\n    return result\\n  }\\n  function row(site, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0, k = 0, l = 0, m = 0, n = 0, o = 0, p = 0, q = 0, r = 0, s = 0, t = 0, u = 0) {\\n    try {\\n      if (!Number.isInteger(site) || site < 1 || site > 70 || !allowed[site] || arguments.length > 22) { faults |= 2; return 0 }\\n      for (let index = 1; index < arguments.length; index++) {\\n        if (typeof arguments[index] !== \"number\" || !Number.isFinite(arguments[index])) { faults |= 2; return 0 }\\n      }\\n      if (drained || count === capacity) { faults |= 1; return 0 }\\n      const offset = count++ * 24\\n      tape[offset] = site; tape[offset + 1] = count; tape[offset + 2] = performance.now()\\n      tape[offset + 3] = a; tape[offset + 4] = b; tape[offset + 5] = c\\n      tape[offset + 6] = d; tape[offset + 7] = e; tape[offset + 8] = f\\n      tape[offset + 9] = g; tape[offset + 10] = h; tape[offset + 11] = i\\n      tape[offset + 12] = j; tape[offset + 13] = k; tape[offset + 14] = l\\n      tape[offset + 15] = m; tape[offset + 16] = n; tape[offset + 17] = o\\n      tape[offset + 18] = p; tape[offset + 19] = q; tape[offset + 20] = r\\n      tape[offset + 21] = s; tape[offset + 22] = t; tape[offset + 23] = u\\n      return count\\n    } catch { faults |= 4; return 0 }\\n  }\\n  function declare(role, ref, upstream, downstream) {\\n    if (!objectLike(ref) || (role !== 1 && role !== 2)) { faults |= 8; return }\\n    const old = declarations.get(ref)\\n    if (old && old[0] !== role) { old[0] = 0; faults |= 8; return }\\n    const value = old || [role, 0, 0]\\n    value[1] = objectLike(upstream) ? upstream : 0; value[2] = objectLike(downstream) ? downstream : 0\\n    declarations.set(ref, value)\\n  }\\n  function owner(value, ref, downstream, popupChange, parentContext, context) {\\n    if (objectLike(ref)) forwardOwners.set(ref, value)\\n    if (objectLike(context)) contexts.set(context, value)\\n    if (objectLike(parentContext)) parentOwners.set(value, contexts.get(parentContext))\\n    if (objectLike(popupChange)) callbackOwners.set(popupChange, value)\\n    if (!objectLike(value)) { faults |= 8; return }\\n    let state = owners.get(value)\\n    if (!state) { state = [value, 0, 0, 0, 0, 0, 0, 0]; owners.set(value, state) }\\n    const declaration = objectLike(ref) && declarations.get(ref)\\n    const link = objectLike(popupChange) && links.get(popupChange)\\n    const nextRole = declaration ? declaration[0] : link ? link[0] : 0\\n    if (state[2] !== (objectLike(ref) ? ref : 0) || state[1] !== nextRole) { state[4] = 0; state[5]++ }\\n    state[1] = nextRole\\n    state[2] = objectLike(ref) ? ref : 0; state[3] = objectLike(downstream) ? downstream : 0; state[6] = objectLike(popupChange) ? popupChange : 0\\n    state[7] = link ? link[1] : 0\\n    const menuCallback = objectLike(popupChange) && exportParents.get(popupChange)\\n    const menuOwner = objectLike(menuCallback) && menuCallbacks.get(menuCallback)\\n    if (menuOwner && objectLike(parentContext) && contexts.get(parentContext)) menuParents.set(menuOwner, contexts.get(parentContext))\\n    if (declaration && declaration[2] !== (objectLike(downstream) ? downstream : 0)) { state[1] = 0; faults |= 8 }\\n  }\\n  function link(callback, role, parent) {\\n    if (!objectLike(callback) || !Number.isInteger(role) || role < 3 || role > 5) { faults |= 8; return }\\n    const previous = links.get(callback), parentRef = objectLike(parent) ? parent : 0\\n    if (previous && (previous[0] !== role || previous[1] !== parentRef)) { previous[0] = 0; faults |= 8; return }\\n    links.set(callback, [role, parentRef])\\n  }\\n  function resolve(value) {\\n    for (let depth = 0; depth < 4 && objectLike(value); depth++) {\\n      const next = aliases.get(value) || callbackOwners.get(value)\\n      if (!next) return value\\n      value = next\\n    }\\n    return value\\n  }\\n  function role(value) {\\n    value = resolve(value)\\n    const state = objectLike(value) && owners.get(value)\\n    if (state && state[1] === 3 && role(parentOwners.get(value)) !== 1) return 0\\n    if (state && state[1] === 4 && role(menuParents.get(value)) !== 1) return 0\\n    return state && state[4] && state[1] ? state[1] : 0\\n  }\\n  function target(value, node, marked) {\\n    const state = objectLike(value) && owners.get(value)\\n    if (!state) { faults |= 8; return }\\n    const nodeId = id(node, 1), previousNode = state[4]\\n    if (marked && nodeId && (state[1] === 1 || state[1] === 2)) {\\n      let pair = rootNodes.get(node)\\n      if (!pair) { pair = [0, 0]; rootNodes.set(node, pair) }\\n      const old = pair[state[1] - 1]\\n      if (old && old !== state[0] && activeOwners.get(old) === nodeId) { state[1] = 0; faults |= 8 }\\n      else pair[state[1] - 1] = state[0]\\n    }\\n    if (state[4] !== nodeId) state[5]++\\n    state[4] = marked || state[1] === 3 ? nodeId : 0\\n    if (marked && !state[1]) faults |= 8\\n    activeOwners.set(value, state[4])\\n    if (previousNode || state[4]) row(1, id(value), state[1], state[5], nodeId, id(state[2]), id(state[3]), id(state[6]), id(state[7]), id(parentOwners.get(value) || menuParents.get(value)))\\n  }\\n  function observe(site, value, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0) {\\n    value = resolve(value)\\n    const state = objectLike(value) && owners.get(value)\\n    if (state && role(value)) {\\n      const index = row(site, id(value), state[1], state[5], a, b, c, d, e, f, origins.get(value) || activeOrigin)\\n      if (site === 5 || site === 7 || site === 8 || site === 9 || site === 14 || site === 15) origins.set(value, index)\\n      return index\\n    }\\n  }\\n  function callback(site, callback, a = 0, b = 0) {\\n    const data = objectLike(callback) && links.get(callback)\\n    const ref = objectLike(callback) && callbackRefs.get(callback)\\n    const owner = objectLike(ref) ? forwardOwners.get(ref) : callbackOwners.get(callback)\\n    if (data && data[0] && role(owner)) row(site, id(callback), data[0], id(data[1]), a, b, id(owner))\\n  }\\n  function eventType(value) {\\n    const result = value === \"pointerdown\" ? 1 : value === \"mousedown\" ? 2 : value === \"mouseup\" ? 3 : value === \"click\" ? 4 : value === \"pointerenter\" ? 5 : value === \"mouseenter\" ? 6 : value === \"pointerleave\" ? 7 : value === \"mouseleave\" ? 8 : value === \"contextmenu\" ? 9 : value === \"focus\" ? 10 : value === \"blur\" ? 11 : value === \"touchstart\" ? 12 : value === \"touchend\" ? 13 : value === \"pointerup\" ? 14 : value === \"touchcancel\" ? 15 : value === \"mousemove\" ? 16 : 0\\n    if (!result) faults |= 2\\n    return result\\n  }\\n  function event(site, value, event) {\\n    if (!role(value) || !objectLike(event)) return\\n    observe(site, value, id(event.nativeEvent || event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, typeof event.clientX === \"number\" ? event.clientX : -1, typeof event.clientY === \"number\" ? event.clientY : -1)\\n  }\\n  function native(site, key, node, result) {\\n    if (!key || key.length !== 3 || !key.every(Number.isSafeInteger)) { faults |= 2; return }\\n    row(site, key[0], key[1], key[2], id(node, 1), result)\\n  }\\n  function nativeEvent(key, node, event, result, fresh, preliminary = false) {\\n    if (!key || !fresh) return\\n    const index = row(51, key[0], key[1], key[2], id(node, 1), result, id(event, 2), eventType(event.type), event.isTrusted === true ? 1 : 0, event.clientX, event.clientY, typeof event.button === \"number\" ? event.button : -1, event.timeStamp)\\n    if (result === 2 && preliminary && !first) first = index\\n  }\\n  function alias(value, parent) { if (objectLike(value) && objectLike(parent)) aliases.set(value, parent) }\\n  function parent(openCallback, menuCallback) {\\n    link(openCallback, 5, openCallback); link(menuCallback, 5, openCallback)\\n    if (objectLike(openCallback)) parentCallbacks.set(openCallback, menuCallback)\\n  }\\n  function child(upstream, downstream, ref) {\\n    const parent = objectLike(upstream) && links.get(upstream)\\n    if (parent && parent[0] === 5) {\\n      link(downstream, 5, ref); callbackRefs.set(upstream, ref); callbackRefs.set(downstream, ref)\\n      const close = parentCallbacks.get(upstream); if (objectLike(close)) callbackRefs.set(close, ref)\\n    }\\n  }\\n  function menu(value, callback) {\\n    if (!owners.has(value)) { owner(value, null, callback, null); owners.get(value)[1] = 4 }\\n    menuCallbacks.set(callback, value)\\n    link(callback, 4, value)\\n  }\\n  function menuCommit(value, callback) {\\n    const state = owners.get(value)\\n    if (!state || !objectLike(value.current)) { faults |= 8; return }\\n    target(value, value.current, true); callbackOwners.set(callback, value)\\n  }\\n  function submenu(callback, popupRef, parentCallback) {\\n    const parent = objectLike(parentCallback) && links.get(parentCallback)\\n    if (!parent || parent[0] !== 4) { faults |= 8; return }\\n    link(callback, 3, parentCallback); exportParents.set(callback, parentCallback)\\n  }\\n  function popup(value, callback) { alias(value, callback) }\\n  function bindPopup(value, node) { if (objectLike(node)) popupOwners.set(node, value) }\\n  function timerRecord(site, serial) {\\n    row(site, timerOwners[serial], timerRoles[serial], timerGenerations[serial], serial, timerKinds[serial], timerReturnTypes[serial], timerReturns[serial], timerDelays[serial], timerOrigins[serial])\\n  }\\n  function installed(ref, callback, kind, returned, delay) {\\n    const value = resolve(callback), state = objectLike(value) && owners.get(value)\\n    if (!state || !role(value)) return 0\\n    if (kind !== 1 && kind !== 2 || !Number.isFinite(delay)) { faults |= 2; return 0 }\\n    const type = typeof returned === \"number\" ? 1 : objectLike(returned) ? 2 : 0\\n    if (!type || type === 1 && (!Number.isSafeInteger(returned) || returned < 0)) { faults |= 2; return 0 }\\n    if (timerCount === 512) { faults |= 1; return 0 }\\n    const old = objectLike(ref) && delaySerials.get(ref)\\n    if (old && timers[old] === 1) faults |= 16\\n    const serial = ++timerCount\\n    timers[serial] = 1; timerOwners[serial] = id(value); timerRoles[serial] = state[1]; timerGenerations[serial] = state[5]\\n    timerKinds[serial] = kind; timerReturnTypes[serial] = type; timerReturns[serial] = type === 1 ? returned : id(returned)\\n    timerDelays[serial] = delay; timerOrigins[serial] = origins.get(value) || 0\\n    if (objectLike(ref)) delaySerials.set(ref, serial)\\n    timerRecord(10, serial)\\n    return serial\\n  }\\n  function cleared(ref, kind) {\\n    const serial = objectLike(ref) && delaySerials.get(ref)\\n    if (!serial) { if (role(ref)) faults |= 16; return }\\n    if (timerKinds[serial] !== kind) { faults |= 16; return }\\n    if (timers[serial] === 1) { timers[serial] = 2; timerRecord(11, serial) }\\n    else if (timers[serial] === 3) { timers[serial] = 4; timerRecord(31, serial) }\\n    else { faults |= 16; return }\\n    delaySerials.delete(ref)\\n  }\\n  function fired(ref, callback, serial, kind) {\\n    if (!serial) { if (role(callback)) faults |= 16; return }\\n    const value = resolve(callback)\\n    if (!role(value) || timerOwners[serial] !== id(value) || timerKinds[serial] !== kind || timers[serial] !== 1) { faults |= 16; return }\\n    timers[serial] = 3; origins.set(value, timerOrigins[serial]); timerRecord(12, serial)\\n  }\\n  function immediate(callback, phase) { observe(phase === 1 ? 29 : 30, callback, 0, phase) }\\n  function reason(value, reason) {\\n    if (reason !== 3 || reasons.get(value) !== 4) reasons.set(value, reason)\\n    observe(19, value, reason)\\n  }\\n  function menuKeys(value, keys) {\\n    observe(20, value, reasons.get(value) || 0, keys.length)\\n    reasons.delete(value)\\n  }\\n  function geometry(popup, target, targetRect, popupRect, next) {\\n    const value = objectLike(popup) && popupOwners.get(popup), state = value && owners.get(value)\\n    if (!state || !role(value)) return\\n    row(25, id(value), state[1], state[5], id(popup, 1), id(target, 1), targetRect.x, targetRect.y, targetRect.width, targetRect.height, popupRect.x, popupRect.y, popupRect.width, popupRect.height, next.offsetX, next.offsetY, next.offsetR, next.offsetB, next.arrowX, next.arrowY, next.scaleX, next.scaleY)\\n  }\\n  function hit(key, target, top, point) {\\n    const tag = top?.nodeName\\n    const descriptor = tag === \"TEXTAREA\" ? 2 : tag === \"BUTTON\" ? 3 : tag === \"LI\" ? 4 : tag === \"SPAN\" ? 5 : tag === \"DIV\" ? 6 : 7\\n    row(54, key[0], key[1], key[2], id(target, 1), id(top, 1), descriptor, -1, point.x, point.y)\\n  }\\n  function drain() {\\n    if (drained) throw new Error(\"already drained\")\\n    drained = true\\n    const rows = []\\n    for (let index = 0; index < count; index++) rows.push(Array.from(tape.subarray(index * 24, (index + 1) * 24)))\\n    return { schema: 43, realm, source, timeOrigin: origin, count, faults, first, counts: [objectCount, nodeCount, eventCount, timerCount], rows, order: 0 }\\n  }\\n  return Object.freeze({ source, realm, hit, alias, parent, child, menu, menuCommit, submenu, popup, bindPopup, installed, cleared, fired, immediate, reason, menuKeys, geometry, row, id, declare, owner, link, role, target, observe, callback, event, native, nativeEvent, drain, fault: flag => { faults |= flag }, origin: value => { activeOrigin = value }, node: value => id(value, 1) })\\n})(2048,2,\"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\") }); } } catch { /* Unavailable observer. */ }",
        "count": 1
      },
      {
        "from": "const preliminaryResult = this.expectHitTarget(hitPoint, element);",
        "to": "const preliminaryResult = this.expectHitTarget(hitPoint, element, df43Key);\\n      df43PreliminaryDone = preliminaryResult === \"done\";\\n      try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.native?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key)\\n        df43Observer.native(50, df43Key, element, df43PreliminaryDone ? 1 : 2)\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}",
        "count": 1
      },
      {
        "from": "      if (result === void 0 && point)\\n        result = this.expectHitTarget({ x: point.clientX, y: point.clientY }, element);",
        "to": "      if (result === void 0 && point) {\\n        result = this.expectHitTarget({ x: point.clientX, y: point.clientY }, element, df43Key);\\n        try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.nativeEvent?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key)\\n        df43Observer.nativeEvent(\\n          df43Key,\\n          element,\\n          event,\\n          result === \"done\" ? 1 : 2,\\n          true,\\n          df43PreliminaryDone,\\n        )\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}\\n      } else {\\n        try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.native?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key)\\n        df43Observer.native(\\n          52,\\n          df43Key,\\n          element,\\n          result === \"done\" ? 1 : result === void 0 ? 0 : 2,\\n        )\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}\\n      }",
        "count": 1
      },
      {
        "from": "    const stop = () => {",
        "to": "    const stop = () => {\\n      try {\\n  const df43Entry = Object.getOwnPropertyDescriptor(globalThis, \"__df43Utility\")\\n  const df43Observer = df43Entry?.value\\n  const df43Shape =\\n    df43Observer && Object.getOwnPropertyDescriptors(df43Observer)\\n  if (\\n    df43Entry &&\\n    df43Entry.writable === false &&\\n    df43Entry.configurable === false &&\\n    Object.isFrozen(df43Observer) &&\\n    df43Shape?.source?.value ===\\n      \"aa48400ea320ca9bf7ffd7e6f9f2ce7c34b8299dfa7f370a963391716cc1402d\" &&\\n    df43Shape.realm?.value === 2 &&\\n    Object.values(df43Shape).every((field) => Object.hasOwn(field, \"value\")) &&\\n    typeof df43Shape.fault?.value === \"function\" &&\\n    typeof df43Shape.native?.value === \"function\"\\n  ) {\\n    try {\\n      if (df43Key)\\n        df43Observer.native(\\n          53,\\n          df43Key,\\n          element,\\n          result === \"done\" ? 1 : result === void 0 ? 0 : 2,\\n        )\\n    } catch {\\n      try {\\n        df43Observer.fault(4)\\n      } catch {\\n        /* Observer failure cannot change product behavior. */\\n      }\\n    }\\n  }\\n} catch {\\n  /* Unavailable observer. */\\n}",
        "count": 1
      }
    ]
  }
]

const deferred43Hash = bytes => createHash("sha256").update(bytes).digest("hex")

export function resolveDeferred43Sources(frontendRoot, repoRoot) {
  const resolve = (from, request) => {
    const resolved = realpathSync(createRequire(from).resolve(request))
    if (!resolved.startsWith(`${realpathSync(repoRoot)}${path.sep}`)) throw new Error("DEFERRED43_SOURCE_OUTSIDE_CHECKOUT")
    return resolved
  }
  const antd = resolve(path.join(repoRoot, "apps/packages/ui/package.json"), "antd/package.json")
  const dropdown = resolve(antd, "@rc-component/dropdown/package.json")
  const tooltip = resolve(antd, "@rc-component/tooltip/package.json")
  const menu = resolve(antd, "@rc-component/menu/package.json")
  const trigger = resolve(dropdown, "@rc-component/trigger/package.json")
  if (resolve(tooltip, "@rc-component/trigger/package.json") !== trigger || resolve(menu, "@rc-component/trigger/package.json") !== trigger) throw new Error("DEFERRED43_TRIGGER_LOAD_COLLISION")
  const testPackage = resolve(path.join(frontendRoot, "package.json"), "@playwright/test/package.json")
  const playwright = resolve(testPackage, "playwright/package.json")
  const core = resolve(playwright, "playwright-core/package.json")
  return [
    ...["es/index.js", "es/hooks/useDelay.js", "es/hooks/useWinClick.js", "es/hooks/useAlign.js"].map(file => path.join(path.dirname(trigger), file)),
    ...["es/SubMenu/PopupTrigger.js", "es/SubMenu/index.js", "es/Menu.js"].map(file => path.join(path.dirname(menu), file)),
    path.join(path.dirname(antd), "es/dropdown/dropdown.js"),
    path.join(path.dirname(dropdown), "es/Dropdown.js"),
    path.join(path.dirname(tooltip), "es/Tooltip.js"),
    path.join(repoRoot, "apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx"),
    path.join(path.dirname(core), "lib/server/dom.js"),
    path.join(path.dirname(core), "lib/generated/injectedScriptSource.js"),
  ]
}

export function applyDeferred43Edits(original, patch) {
  let result = original
  for (const { from, to, count } of patch.edits) {
    if (result.split(from).length - 1 !== count) throw new Error("DEFERRED43_REPLACEMENT_COUNT")
    result = result.split(from).join(to)
  }
  return result
}

export function installDeferred43Sources({
  repoRoot,
  frontendRoot,
  artifactRoot,
  receiptRoot,
  targets = resolveDeferred43Sources(frontendRoot, repoRoot),
  afterSwap = () => {},
  transaction = {},
}) {
  const root = realpathSync(repoRoot),
    artifacts = realpathSync(artifactRoot),
    frontend = realpathSync(frontendRoot);
  const runId = path.basename(artifacts),
    temporary = realpathSync(tmpdir());
  if (
    !/^[A-Za-z0-9._-]{1,128}$/.test(runId) ||
    artifacts !== path.join(frontend, 'test-results/live-tier-uat', runId) ||
    !artifacts.startsWith(`${root}${path.sep}`) ||
    targets.length !== deferred43Patches.length
  )
    throw new Error('DEFERRED43_SCOPE');
  const receiptDirectory = realpathSync(receiptRoot);
  if (
    receiptDirectory !==
    path.join(temporary, `tldw-onboarding-uat-${runId}`, 'reports/live-tier-uat')
  )
    throw new Error('DEFERRED43_RECEIPT_SCOPE');
  const receiptAncestors = [];
  for (
    let ancestor = path.resolve(receiptRoot);
    ancestor !== path.resolve(tmpdir());
    ancestor = path.dirname(ancestor)
  ) {
    const stats = lstatSync(ancestor);
    if (
      !ancestor.startsWith(`${path.resolve(tmpdir())}${path.sep}`) ||
      !stats.isDirectory() ||
      stats.isSymbolicLink() ||
      stats.uid !== process.getuid?.()
    )
      throw new Error('DEFERRED43_RECEIPT_ANCESTOR');
    receiptAncestors.push({ path: ancestor, ino: stats.ino, dev: stats.dev });
  }
  const records = [],
    receiptPath = path.join(receiptDirectory, 'deferred43-source-receipt.json');
  const receipt = { schema: 43, source: deferred43Source, appSource: deferred43AppSource, installed: false, restored: false, records, runtime: { platform: process.platform, node: process.version, binary: deferred43Hash(readFileSync(process.execPath)) } }
  let receiptIdentity = null,
    receiptHash = null;
  const save = () => {
    for (const ancestor of receiptAncestors) {
      const stats = lstatSync(ancestor.path);
      if (
        !stats.isDirectory() ||
        stats.isSymbolicLink() ||
        stats.ino !== ancestor.ino ||
        stats.dev !== ancestor.dev ||
        stats.uid !== process.getuid?.()
      )
        throw new Error('DEFERRED43_RECEIPT_ANCESTOR_CHANGED');
    }
    if (realpathSync(receiptRoot) !== receiptDirectory)
      throw new Error('DEFERRED43_RECEIPT_ANCESTOR_CHANGED');
    const bytes = Buffer.from(JSON.stringify(receipt));
    const fd = openSync(
      receiptPath,
      constants.O_RDWR |
        constants.O_NOFOLLOW |
        (receiptIdentity ? 0 : constants.O_CREAT | constants.O_EXCL),
      0o600
    );
    try {
      const stats = fstatSync(fd);
      if (
        !stats.isFile() ||
        stats.nlink !== 1 ||
        (stats.mode & 0o777) !== 0o600 ||
        stats.uid !== process.getuid?.() ||
        (receiptIdentity &&
          (stats.ino !== receiptIdentity.ino ||
            stats.dev !== receiptIdentity.dev ||
            deferred43Hash(readFileSync(fd)) !== receiptHash))
      )
        throw new Error('DEFERRED43_RECEIPT_CHANGED');
      let written = 0;
      while (written < bytes.length) {
        const count = writeSync(fd, bytes, written, bytes.length - written, written);
        if (count === 0) throw new Error('DEFERRED43_RECEIPT_SHORT_WRITE');
        written += count;
      }
      ftruncateSync(fd, bytes.length);
      const readback = lstatSync(receiptPath);
      if (
        !readback.isFile() ||
        readback.isSymbolicLink() ||
        readback.ino !== stats.ino ||
        readback.dev !== stats.dev ||
        readback.nlink !== 1 ||
        (readback.mode & 0o777) !== 0o600 ||
        readback.uid !== process.getuid?.() ||
        realpathSync(receiptPath) !== receiptPath ||
        deferred43Hash(readFileSync(receiptPath)) !== deferred43Hash(bytes)
      )
        throw new Error('DEFERRED43_RECEIPT_CHANGED');
      receiptIdentity = { ino: stats.ino, dev: stats.dev };
      receiptHash = deferred43Hash(bytes);
      transaction.ownsReceipt = true;
    } finally {
      closeSync(fd);
    }
  };
  const restore = () => {
    const errors = []
    for (const record of [...records].reverse()) {
      if (!record.moved || record.restored) continue
      try {
        let current
        try { current = lstatSync(record.target) } catch (error) { if (error.code !== "ENOENT") throw error }
        if (current) {
          if (!current.isFile() || current.ino !== record.candidateInode || current.dev !== record.dev || deferred43Hash(readFileSync(record.target)) !== record.candidateHash) throw new Error("DEFERRED43_RESTORE_TARGET_CHANGED")
        }
        const backup = lstatSync(record.backup)
        if (!backup.isFile() || backup.ino !== record.ino || backup.dev !== record.dev || deferred43Hash(readFileSync(record.backup)) !== record.originalHash) throw new Error("DEFERRED43_BACKUP_CHANGED")
        renameSync(record.backup, record.target)
        const restored = lstatSync(record.target)
        if (restored.ino !== record.ino || restored.dev !== record.dev || (restored.mode & 0o777) !== record.mode || restored.nlink !== record.nlink || deferred43Hash(readFileSync(record.target)) !== record.originalHash) throw new Error("DEFERRED43_RESTORE_MISMATCH")
        record.restored = true
      } catch (error) { errors.push(error) }
    }
    receipt.restored = records.every(record => !record.moved || record.restored)
    save()
    if (errors.length) throw new AggregateError(errors, "DEFERRED43_RESTORATION_UNAVAILABLE")
  }
  Object.assign(transaction, { restore, receipt, receiptPath, ownsReceipt: false });
  try {
    save();
    // Validate every source and ancestor before the first rename. Hardlinks are preserved by rename.
    const staged = targets.map((target, index) => {
      target = path.resolve(target)
      if (!target.startsWith(`${root}${path.sep}`) || realpathSync(target) !== target) throw new Error("DEFERRED43_TARGET_ISOLATION")
      let ancestor = path.dirname(target)
      while (ancestor !== root) {
        const stats = lstatSync(ancestor)
        if (!stats.isDirectory() || stats.isSymbolicLink()) throw new Error("DEFERRED43_ANCESTOR_ISOLATION")
        ancestor = path.dirname(ancestor)
      }
      const stats = lstatSync(target), original = readFileSync(target, "utf8"), patch = deferred43Patches[index]
      if (!stats.isFile() || stats.isSymbolicLink() || stats.uid !== process.getuid?.() || deferred43Hash(original) !== patch.sha256) throw new Error("DEFERRED43_SOURCE_HASH")
      const candidate = applyDeferred43Edits(original, patch)
      if (deferred43Hash(candidate) !== patch.candidateHash) throw new Error("DEFERRED43_CANDIDATE_HASH")
      return { target, original, candidate, stats }
    })
    for (let index = 0; index < staged.length; index++) {
      const { target, candidate, stats } = staged[index]
      const suffix = `.deferred43-${process.pid}-${randomBytes(8).toString("hex")}`
      const stage = `${target}${suffix}.candidate`, backup = `${target}${suffix}.original`
      writeFileSync(stage, candidate, { flag: "wx", mode: stats.mode & 0o777 })
      chmodSync(stage, stats.mode & 0o777)
      const independent = lstatSync(stage)
      if (independent.nlink !== 1 || independent.ino === stats.ino || independent.dev !== stats.dev) throw new Error("DEFERRED43_CANDIDATE_INODE")
      const record = { target, backup, stage, dev: stats.dev, ino: stats.ino, mode: stats.mode & 0o777, nlink: stats.nlink, originalHash: deferred43Patches[index].sha256, candidateHash: deferred43Patches[index].candidateHash, candidateInode: independent.ino, moved: false, restored: false }
      records.push(record); save()
      if (existsSync(backup) || lstatSync(target).ino !== stats.ino || deferred43Hash(readFileSync(target)) !== record.originalHash) throw new Error("DEFERRED43_SOURCE_CHANGED")
      renameSync(target, backup); record.moved = true; save()
      renameSync(stage, target); save()
      afterSwap(index)
    }
    receipt.installed = true; save()
    return transaction;
  } catch (error) {
    try { restore() } catch (restoreError) { throw new AggregateError([error, restoreError], "DEFERRED43_INSTALL_RESTORE_FAILED") }
    throw error
  }
}

function inspectDeferred43Ledgers(artifactRoot, retainedBodies) {
  let files
  try { files = readdirSync(artifactRoot).filter(name => /^deferred43-server-\d+\.json$/.test(name)) }
  catch { return { files: 0, owning: 0, faults: 4, available: false } }
  let owning = 0, faults = 0
  for (const name of files) {
    let bytes, stats;
    try {
      const file = path.join(artifactRoot, name);
      stats = lstatSync(file);
      if (!stats.isFile() || stats.isSymbolicLink() || realpathSync(file) !== path.resolve(file)) {
        faults |= 4;
        retainedBodies?.push({ kind: 'server', fault: 4 });
        continue;
      }
      if (stats.size > 262144) {
        faults |= 1;
        retainedBodies?.push({ kind: 'server', fault: 1 });
        continue;
      }
      bytes = readFileSync(file)
    } catch { faults |= 4; continue }
    if (bytes.length > 262144) {
      faults |= 1;
      retainedBodies?.push({ kind: 'server', fault: 1 });
      continue;
    }
    try {
      const payload = JSON.parse(bytes)
      if (
        payload.schema !== 43 ||
        payload.realm !== 3 ||
        payload.source !== deferred43Source ||
        !Number.isInteger(payload.count) ||
        payload.count < 0 ||
        payload.count > 1024 ||
        !Number.isSafeInteger(payload.faults) ||
        payload.faults < 0 ||
        !Number.isFinite(payload.timeOrigin) ||
        payload.first !== 0 ||
        payload.order !== 0 ||
        !Array.isArray(payload.counts) ||
        payload.counts.length !== 4 ||
        payload.counts.some((value) => !Number.isSafeInteger(value) || value < 0 || value > 512) ||
        !Array.isArray(payload.rows) ||
        payload.rows.length !== payload.count ||
        payload.rows.some(
          (row, index) =>
            !Array.isArray(row) ||
            row.length !== 24 ||
            ![40, 41, 42, 43, 44, 45, 46].includes(row[0]) ||
            row[1] !== index + 1 ||
            row.some((value) => typeof value !== 'number' || !Number.isFinite(value))
        )
      ) {
        faults |= 2;
        retainedBodies?.push({ kind: 'server', fault: 2 });
        continue;
      }
      if (retainedBodies && payload.count) {
        const browser = payload.browser;
        const flags =
          /^--(?:headless(?:=(?:new|old))?|enable-automation|no-sandbox|disable-dev-shm-usage)$|^--window-size=\d{1,5},\d{1,5}$|^--force-device-scale-factor=\d{1,2}(?:\.\d{1,4})?$/;
        const hash = (value) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
        const safe =
          bytes.equals(Buffer.from(JSON.stringify(payload))) &&
          stats.uid === process.getuid?.() &&
          stats.nlink === 1 &&
          (stats.mode & 0o777) === 0o600 &&
          Object.keys(payload).sort().join(',') ===
            'binary,browser,count,counts,faults,first,order,pid,platform,realm,rows,schema,source,timeOrigin,version' &&
          Number.isSafeInteger(payload.pid) &&
          payload.pid > 0 &&
          payload.pid === Number(name.slice(18, -5)) &&
          ['linux', 'darwin', 'win32'].includes(payload.platform) &&
          /^v\d+\.\d+\.\d+$/.test(payload.version) &&
          hash(payload.binary) &&
          (browser === null ||
            (Object.keys(browser).sort().join(',') ===
              'binary,executable,flags,omittedArgs,pid,readback,version' &&
              Number.isSafeInteger(browser.pid) &&
              browser.pid > 0 &&
              typeof browser.executable === 'string' &&
              /^[A-Za-z0-9/_.+:-]{1,128}$/.test(browser.executable) &&
              /^\d+(?:\.\d+){1,3}$/.test(browser.version) &&
              Array.isArray(browser.flags) &&
              browser.flags.length <= 64 &&
              browser.flags.every((value) => typeof value === 'string' && flags.test(value)) &&
              Number.isSafeInteger(browser.omittedArgs) &&
              browser.omittedArgs >= 0 &&
              browser.omittedArgs <= 64 &&
              [0, 1].includes(browser.readback) &&
              (browser.binary === null || hash(browser.binary))));
        retainedBodies.push(safe ? { kind: 'server', bytes } : { kind: 'server', fault: 2 });
      }
      if (payload.count) owning++
      faults |= payload.faults
    } catch {
      faults |= 4;
      retainedBodies?.push({ kind: 'server', fault: 4 });
    }
  }
  return { files: files.length, owning, faults, available: owning === 1 && faults === 0 }
}

const moduleDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRootDefault = path.resolve(moduleDir, "../..")
const repoRootDefault = path.resolve(frontendRootDefault, "../..")
const validProjects = new Set(["tier-1", "tier-2", "tier-3", "ux-regression", "notes-studio"])

function runIdNow() {
  return new Date().toISOString().replace(/[:.]/g, "-")
}

function requireNumber(value, flag) {
  const parsed = Number(value)
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${flag} requires a positive integer`)
  return parsed
}

export function parseArgs(argv = process.argv.slice(2)) {
  const options = {
    projects: ["tier-1", "tier-2", "tier-3"],
    workers: 1,
    listOnly: false,
    grep: null,
    preserveRuntime: false,
    failOnSkip: true,
    runId: null,
    help: false,
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === "--help" || arg === "-h") options.help = true
    else if (arg === "--list-only") options.listOnly = true
    else if (arg === "--preserve-runtime") options.preserveRuntime = true
    else if (arg === "--allow-skips") options.failOnSkip = false
    else if (arg.startsWith("--projects=")) options.projects = arg.slice(11).split(",").filter(Boolean)
    else if (arg === "--projects") options.projects = (argv[++index] ?? "").split(",").filter(Boolean)
    else if (arg.startsWith("--workers=")) options.workers = requireNumber(arg.slice(10), "--workers")
    else if (arg === "--workers") options.workers = requireNumber(argv[++index], "--workers")
    else if (arg.startsWith("--grep=")) options.grep = arg.slice(7)
    else if (arg === "--grep") options.grep = argv[++index] ?? null
    else if (arg.startsWith("--run-id=")) options.runId = arg.slice(9)
    else if (arg === "--run-id") options.runId = argv[++index] ?? null
    else throw new Error(`Unknown option: ${arg}`)
  }

  if (!options.projects.length || options.projects.some((project) => !validProjects.has(project))) {
    throw new Error("--projects entries must be tier-1, tier-2, tier-3, ux-regression, or notes-studio")
  }
  if (options.runId && !/^[A-Za-z0-9._-]+$/.test(options.runId)) {
    throw new Error("--run-id may contain only letters, numbers, dots, underscores, and hyphens")
  }
  return options
}

export function formatUsage() {
  return [
    "Usage: bun run uat:live-tiers -- [options]",
    "",
    "  --projects=tier-1,tier-2,tier-3  Complete projects to list and run (also: ux-regression, notes-studio).",
    "  --workers=1                       Playwright worker count.",
    "  --list-only                       List and inventory without executing tests.",
    "  --grep=<pattern>                  Non-certifying bounded smoke selection.",
    "  --preserve-runtime                Keep the disposable backend profile.",
    "  --allow-skips                     Report skips without failing the run.",
    "  --run-id=<id>                     Stable artifact identifier.",
  ].join("\n")
}

export function isCertificationRun() {
  // This runner uses Next dev, a mutable checkout, SQLite single-user and
  // controlled inference. Even a complete run is diagnostic evidence only.
  return false
}

function safeEnv(baseEnv, keys) {
  return Object.fromEntries(keys.flatMap((key) =>
    typeof baseEnv[key] === "string" ? [[key, baseEnv[key]]] : []
  ))
}

function withPythonPath(env, value) {
  return { ...env, PYTHONPATH: [value, env.PYTHONPATH].filter(Boolean).join(path.delimiter) }
}

function playwrightProjectArgs(projects) {
  return projects.map((project) => `--project=${project}`)
}

const baseEnvKeys = ["PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "PYTHONPATH", "VIRTUAL_ENV"]

/** Where the ux-regression side-panel specs find the unpacked Chrome extension. */
export const EXTENSION_DIR_ENV = "TLDW_UXR_EXTENSION_DIR"

/**
 * The ux-regression project's side-panel specs load the built browser
 * extension. A run that includes the project builds it first (production
 * profile, as the extension's own CI does). Setting TLDW_UXR_EXTENSION_DIR to
 * an existing build skips that step for local iteration; the specs still fail,
 * never skip, if the directory holds no usable build.
 *
 * @returns {{ dir: string, command: { name: string, command: string, args: string[], cwd: string, env: Record<string, string> } | null } | null}
 */
export function resolveExtensionBuild({
  frontendRoot = frontendRootDefault,
  projects,
  baseEnv = process.env,
}) {
  if (!projects.includes("ux-regression")) return null
  const extensionRoot = path.resolve(frontendRoot, "../extension")
  const prebuilt = baseEnv[EXTENSION_DIR_ENV]
  if (prebuilt) return { dir: path.resolve(prebuilt), command: null }
  return {
    dir: path.join(extensionRoot, ".output/chrome-mv3"),
    command: {
      name: "extension-build",
      command: "bun",
      args: ["run", "build:chrome:prod"],
      cwd: extensionRoot,
      // Rollup's production bundle exceeds Node's default heap during rendering.
      env: { ...safeEnv(baseEnv, baseEnvKeys), NODE_OPTIONS: "--max-old-space-size=6144" },
    },
  }
}

export function buildCommands({
  repoRoot = repoRootDefault,
  frontendRoot = frontendRootDefault,
  ports,
  profile,
  projects,
  workers,
  runId,
  grep = null,
  artifactRoot = profile.reportsDir,
  baseEnv = process.env,
}) {
  const python = resolvePythonCommand({ repoRoot, baseEnv })
  const backendEnv = buildLiveTierBackendEnv({ profile, mockPort: ports.mock, baseEnv })
  const apiKey = backendEnv.SINGLE_USER_API_KEY
  const backendUrl = `http://127.0.0.1:${ports.backend}`
  const webUrl = `http://localhost:${ports.web}`
  const mockUrl = `http://127.0.0.1:${ports.mock}`
  if (!/^[A-Za-z0-9._-]+$/.test(runId)) {
    throw new Error("runId is not safe for the isolated Next build directory")
  }
  const nextDistDir = `.next-live-tier-${runId}`
  const nextDistPath = path.join(frontendRoot, nextDistDir)
  const baseKeys = baseEnvKeys
  const extension = resolveExtensionBuild({ frontendRoot, projects, baseEnv })
  const sharedFrontendEnv = {
    ...safeEnv(baseEnv, baseKeys),
    ...(extension ? { [EXTENSION_DIR_ENV]: extension.dir } : {}),
    NEXT_PUBLIC_API_URL: backendUrl,
    NEXT_PUBLIC_API_VERSION: "v1",
    NEXT_PUBLIC_X_API_KEY: apiKey,
    TLDW_SERVER_URL: backendUrl,
    TLDW_E2E_SERVER_URL: backendUrl,
    TLDW_E2E_API_KEY: apiKey,
    TLDW_API_KEY: apiKey,
    TLDW_WEB_URL: webUrl,
    TLDW_WEB_AUTOSTART: "false",
    TLDW_E2E_ALLOW_OFFLINE: "0",
    TLDW_MOCK_OPENAI_URL: `${mockUrl}/v1`,
    TLDW_LIVE_TIER_UAT: "1",
    TLDW_LIVE_TIER_UAT_RUN_ID: runId,
    TLDW_E2E_ACP_WORKSPACE_ROOT_BASE: profile.acpWorkspaceRootBase,
    TLDW_E2E_INGESTION_SOURCE_ROOT: profile.fixtureRoot,
    TLDW_NEXT_DIST_DIR: nextDistDir,
  }
  const projectArgs = playwrightProjectArgs(projects)
  const jsonPath = path.join(artifactRoot, "playwright-results.json")
  const listJsonPath = path.join(artifactRoot, "playwright-list.json")
  const runArgs = [
    "playwright", "test", ...projectArgs, `--workers=${workers}`, "--retries=0", "--reporter=line,json",
    `--output=${path.join(artifactRoot, 'playwright-output')}`,
  ]
  const grepArgs = grep ? ["--grep", grep] : []
  runArgs.push(...grepArgs)

  return {
    urls: { backend: backendUrl, web: webUrl, mock: mockUrl },
    backendEnv,
    mockOpenai: {
      name: "mock-openai",
      command: python,
      args: [
        "-m", "mock_openai.server", "--config",
        path.join(frontendRoot, "e2e/onboarding-uat/mock-openai/configs/local-success.json"),
        "--host", "127.0.0.1", "--port", String(ports.mock),
      ],
      cwd: path.join(frontendRoot, "e2e/onboarding-uat/mock-openai"),
      env: withPythonPath(safeEnv(baseEnv, baseKeys), path.join(repoRoot, "mock_openai_server")),
    },
    authInit: {
      name: "auth-init",
      command: python,
      args: ["-m", "tldw_Server_API.app.core.AuthNZ.initialize", "--non-interactive"],
      cwd: profile.runDir,
      env: withPythonPath(backendEnv, repoRoot),
    },
    backend: {
      name: "backend",
      command: python,
      args: ["-m", "uvicorn", "tldw_Server_API.app.main:app", "--host", "127.0.0.1", "--port", String(ports.backend)],
      cwd: profile.runDir,
      env: withPythonPath(backendEnv, repoRoot),
    },
    frontend: {
      name: "frontend",
      command: "bun",
      args: ["run", "dev", "--", "-p", String(ports.web)],
      cwd: frontendRoot,
      env: sharedFrontendEnv,
    },
    playwrightList: {
      name: "playwright-list",
      command: "bunx",
      args: ["playwright", "test", "--list", "--reporter=json", ...projectArgs, ...grepArgs],
      cwd: frontendRoot,
      env: { ...sharedFrontendEnv, PLAYWRIGHT_JSON_OUTPUT_NAME: listJsonPath },
    },
    playwrightRun: {
      name: "playwright-run",
      command: "bunx",
      args: runArgs,
      cwd: frontendRoot,
      env: { ...sharedFrontendEnv, PLAYWRIGHT_JSON_OUTPUT_NAME: jsonPath },
    },
    jsonPath,
    listJsonPath,
    nextDistPath,
  }
}

function appendLog(logPath, chunk) {
  mkdirSync(path.dirname(logPath), { recursive: true })
  appendFileSync(logPath, redactText(chunk), "utf8")
}

/**
 * Run a one-shot command and wait for authoritative process-tree teardown on abort.
 *
 * @param {{ name?: string, command: string, args?: string[], cwd?: string, env?: NodeJS.ProcessEnv }} commandRecord
 * @param {string} logPath
 * @param {{ signal?: AbortSignal, stop?: (record: unknown) => Promise<void> }} [options]
 */
export async function runCommand(commandRecord, logPath, {
  signal,
  stop = stopProcessTree,
} = {}) {
  if (signal?.aborted) {
    return {
      code: null,
      signal: null,
      error: signal.reason,
      output: "",
      aborted: true,
    }
  }
  return new Promise((resolve) => {
    const child = spawn(commandRecord.command, commandRecord.args, {
      cwd: commandRecord.cwd,
      env: commandRecord.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    })
    const record = { ...commandRecord, child, pid: child.pid, loggingErrors: [] }
    let output = ""
    let settled = false
    let aborting = false
    const finish = (result) => {
      if (settled) return
      settled = true
      signal?.removeEventListener("abort", abort)
      resolve(result)
    }
    const abort = () => {
      if (settled || aborting) return
      aborting = true
      const reason = signal?.reason instanceof Error
        ? signal.reason
        : new Error(`${commandRecord.name ?? "Command"} aborted`)
      void stop(record).then(
        () => finish({ code: null, signal: null, error: reason, output, aborted: true }),
        (stopError) => finish({
          code: null,
          signal: null,
          error: new AggregateError([reason, stopError], `Failed to abort ${commandRecord.name ?? "command"}`),
          output,
          aborted: true,
        })
      )
    }
    const capture = (chunk) => {
      const text = String(chunk)
      output += text
      appendLog(logPath, text)
      process.stdout.write(redactText(text))
    }
    child.stdout?.on("data", capture)
    child.stderr?.on("data", capture)
    child.once("error", (error) => {
      if (!aborting) finish({ code: null, signal: null, error, output })
    })
    child.once("close", (code, childSignal) => {
      if (!aborting) finish({ code, signal: childSignal, output })
    })
    signal?.addEventListener("abort", abort, { once: true })
    if (signal?.aborted) abort()
  })
}

/**
 * @param {{
 *   processLike?: { on: (event: string, listener: () => void) => unknown, off: (event: string, listener: () => void) => unknown },
 *   controller?: AbortController,
 * }} [options]
 */
export function installTerminationHandlers({
  processLike = process,
  controller = new AbortController(),
} = {}) {
  const abortFor = (signalName) => {
    if (controller.signal.aborted) return
    const error = new Error(`Live Tier UAT interrupted by ${signalName}`)
    error.name = "AbortError"
    error.signalName = signalName
    controller.abort(error)
  }
  const handlers = {
    SIGINT: () => abortFor("SIGINT"),
    SIGTERM: () => abortFor("SIGTERM"),
  }
  processLike.on("SIGINT", handlers.SIGINT)
  processLike.on("SIGTERM", handlers.SIGTERM)

  return {
    controller,
    signal: controller.signal,
    dispose() {
      processLike.off("SIGINT", handlers.SIGINT)
      processLike.off("SIGTERM", handlers.SIGTERM)
    },
  }
}

export async function stopSpawnedProcesses(records, stop = stopProcessTree) {
  const errors = []
  for (const record of [...records].reverse()) {
    try {
      await stop(record)
    } catch (error) {
      errors.push(
        new Error(
          `Failed to stop ${record.name ?? "spawned process"}: ${error?.message ?? String(error)}`,
          { cause: error }
        )
      )
    }
  }
  if (errors.length) {
    throw new AggregateError(
      errors,
      `Failed to stop spawned processes: ${errors.map((error) => error.message).join("; ")}`
    )
  }
}

export function assertFreshRunTargets(targets, exists = existsSync) {
  for (const target of targets) {
    if (exists(target)) {
      throw new Error(`Live Tier run target already exists: ${target}`)
    }
  }
}

export function readPlaywrightReport(jsonPath, {
  exists = existsSync,
  read = readFileSync,
} = {}) {
  if (!exists(jsonPath)) {
    throw new Error(`Playwright JSON report was not produced: ${jsonPath}`)
  }
  return JSON.parse(read(jsonPath, "utf8"))
}

/** Retain a flushed partial report after teardown, while preserving the original abort reason. */
export function readPlaywrightOutcome(jsonPath, { signal } = {}) {
  let report = null
  let error = null
  try {
    report = readPlaywrightReport(jsonPath)
  } catch (caught) {
    error = caught
  }
  return { report, error: signal?.aborted ? signal.reason : error }
}

export function assertServicesStopped(stopped) {
  if (!stopped) {
    throw new Error("One or more spawned live-tier services did not stop")
  }
}

const retryableCleanupCodes = new Set(["EBUSY", "ENOTEMPTY", "EPERM"])

function makeGeneratedDirectoryTreeWritable(target) {
  if (!existsSync(target)) return
  const pending = [target]
  while (pending.length) {
    const current = pending.pop()
    let stats
    try {
      stats = lstatSync(current)
    } catch (error) {
      if (error?.code === "ENOENT") continue
      throw error
    }
    if (stats.isSymbolicLink() || !stats.isDirectory()) continue
    chmodSync(current, 0o700)
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        pending.push(path.join(current, entry.name))
      }
    }
  }
}

/**
 * Remove one explicitly-scoped generated directory, retrying transient
 * filesystem races without widening the cleanup target.
 *
 * @param {string} target
 * @param {{
 *   expectedParent: string,
 *   expectedPrefix: string,
 *   remove?: typeof rmSync,
 *   wait?: (delayMs: number) => Promise<void>,
 *   maxAttempts?: number,
 *   retryDelayMs?: number,
 * }} options
 * @returns {Promise<void>}
 */
export async function removeGeneratedPath(target, {
  expectedParent,
  expectedPrefix,
  remove = rmSync,
  wait = (delayMs) => new Promise((resolve) => setTimeout(resolve, delayMs)),
  maxAttempts = 10,
  retryDelayMs = 100,
}) {
  const resolvedTarget = path.resolve(target)
  const resolvedParent = path.resolve(expectedParent)
  const basename = path.basename(resolvedTarget)
  if (
    path.dirname(resolvedTarget) !== resolvedParent ||
    !basename.startsWith(expectedPrefix) ||
    basename.length <= expectedPrefix.length
  ) {
    throw new Error(`Refusing unsafe generated cleanup target: ${resolvedTarget}`)
  }

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      makeGeneratedDirectoryTreeWritable(resolvedTarget)
      remove(resolvedTarget, { recursive: true, force: true })
      return
    } catch (error) {
      if (!retryableCleanupCodes.has(error?.code) || attempt === maxAttempts) {
        throw error
      }
      await wait(retryDelayMs)
    }
  }
}

export function assertNoMutableRepoDatabasePaths(logText, repoRoot) {
  const normalizedLog = String(logText).replaceAll("\\", "/")
  const normalizedRoot = path.resolve(repoRoot).replaceAll("\\", "/")
  const escapedRoots = [
    `${normalizedRoot}/Databases/`,
    `${normalizedRoot}/tldw_Server_API/Databases/`,
  ]
  const absoluteLeak = escapedRoots.find((root) => normalizedLog.includes(root))
  const relativeLeak = normalizedLog.match(/\b(?:db_path=|database:\s*)Databases\/[^\s]+\.db\b/)
  if (absoluteLeak || relativeLeak) {
    throw new Error(
      `Live Tier mutable database path escaped the disposable profile: ${absoluteLeak ?? relativeLeak[0]}`
    )
  }
}

export function assertOnlyLoopbackHttpRequests(logText) {
  // sensitive-endpoint.invalid is the unresolvable placeholder the backend logs
  // in place of a sensitive request's real URL (tldw_Server_API/app/core/http_client.py);
  // it is never a real outbound request.
  const allowedHosts = new Set(["127.0.0.1", "::1", "localhost", "sensitive-endpoint.invalid"])
  for (const line of String(logText).split(/\r?\n/)) {
    if (!/(?:url\.full|HTTP Request:)/.test(line)) continue
    for (const match of line.matchAll(/https?:\/\/[^\s"'<>]+/g)) {
      const candidate = match[0].replace(/[),.;]+$/, "")
      let hostname
      try {
        hostname = new URL(candidate).hostname.toLowerCase().replace(/^\[|\]$/g, "")
      } catch {
        continue
      }
      if (!allowedHosts.has(hostname)) {
        throw new Error(`Live Tier backend made a non-loopback HTTP request to ${hostname}`)
      }
    }
  }
}

async function probeStopped(url) {
  try {
    await fetch(url, { signal: AbortSignal.timeout(750) })
    return false
  } catch {
    return true
  }
}

function commitHash(repoRoot) {
  const result = spawnSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
  return result.status === 0 ? result.stdout.trim() : "unknown"
}

export async function runLiveTierUat({
  options = parseArgs(),
  repoRoot = repoRootDefault,
  frontendRoot = frontendRootDefault,
  baseEnv = process.env,
  signal,
} = {}) {
  if (options.help) {
    process.stdout.write(`${formatUsage()}\n`)
    return { status: "help" }
  }

  if (options.listOnly) return listLiveTierCases({ options, frontendRoot, baseEnv, signal })

  const runId = options.runId ?? runIdNow()
  const artifactRoot = path.join(frontendRoot, "test-results/live-tier-uat", runId)
  const profileRoot = path.join(tmpdir(), `tldw-onboarding-uat-${runId}`)
  const nextDistPath = path.join(frontendRoot, `.next-live-tier-${runId}`)
  const logs = {
    extensionBuild: path.join(artifactRoot, "extension-build.log"),
    mock: path.join(artifactRoot, "mock-openai.log"),
    backend: path.join(artifactRoot, "backend.log"),
    frontend: path.join(artifactRoot, "frontend.log"),
    list: path.join(artifactRoot, "playwright-list.log"),
    run: path.join(artifactRoot, "playwright-run.log"),
  }
  const reportPath = path.join(artifactRoot, "report.md")
  const summaryPath = path.join(artifactRoot, "summary.json")
  const inventoryPath = path.join(artifactRoot, "api-interception-inventory.json")
  const processes = []
  const health = { before: false, after: false, stopped: false }
  const startedCommit = commitHash(repoRoot)
  let registeredCases = []
  let playwrightReport = null
  let listed = {}
  let results = {}
  let inventory = []
  let skippedTests = []
  let status = "failed"
  let runnerError = null
  let ports = {}
  let profile = null
  let commands = null
  let ownsGeneratedTargets = false
  let ownsArtifactRoot = false
  let deferred43Transaction = null
  let deferred43Ledger = null
  const deferred43RetainedBodies = [];

  try {
    assertFreshRunTargets([artifactRoot, profileRoot, nextDistPath])
    ownsGeneratedTargets = true
    mkdirSync(artifactRoot, { recursive: true })
    ownsArtifactRoot = true
    // Build before reserving ports, so a slow build cannot outlive the
    // reservation and a broken one fails before any service starts.
    const extensionBuild = resolveExtensionBuild({ frontendRoot, projects: options.projects, baseEnv })?.command
    if (extensionBuild) {
      const build = await runCommand(extensionBuild, logs.extensionBuild, { signal })
      signal?.throwIfAborted()
      if (build.code !== 0) {
        throw new Error(`Extension build failed with exit code ${build.code}; see ${logs.extensionBuild}`)
      }
    }
    ports = await reservePorts(["backend", "web", "mock"])
    const pythonCommand = resolvePythonCommand({ repoRoot, baseEnv })
    profile = buildLiveTierProfile({
      repoRoot,
      frontendRoot,
      runId,
      mockPort: ports.mock,
      baseTmpDir: tmpdir(),
      pythonCommand,
    })
    if (options.projects.length === 1 && options.projects[0] === 'ux-regression' && !options.grep) {
      deferred43Transaction = {};
      installDeferred43Sources({
        repoRoot,
        frontendRoot,
        artifactRoot,
        receiptRoot: profile.reportsDir,
        transaction: deferred43Transaction,
      });
    }
    commands = buildCommands({
      repoRoot,
      frontendRoot,
      ports,
      profile,
      projects: options.projects,
      workers: options.workers,
      runId,
      grep: options.grep,
      artifactRoot,
      baseEnv,
    })
    if (deferred43Transaction?.receipt)
      commands.playwrightRun.env.TLDW_DEFERRED43_SOURCE = deferred43Source;
    signal?.throwIfAborted()
    processes.push(spawnLoggedProcess({ ...commands.mockOpenai, logPath: logs.mock }))
    await waitForHttpOk(`${commands.urls.mock}/health`, { timeoutMs: 30_000, signal })

    const auth = await runCommand(commands.authInit, logs.backend, { signal })
    signal?.throwIfAborted()
    if (auth.code !== 0) throw new Error(`AuthNZ initialization failed with exit code ${auth.code}`)

    processes.push(spawnLoggedProcess({ ...commands.backend, logPath: logs.backend }))
    await waitForHttpOk(`${commands.urls.backend}/api/v1/health`, {
      timeoutMs: 120_000,
      headers: { "X-API-KEY": commands.backendEnv.SINGLE_USER_API_KEY },
      signal,
    })

    processes.push(spawnLoggedProcess({ ...commands.frontend, logPath: logs.frontend }))
    await waitForHttpOk(commands.urls.web, { timeoutMs: 120_000, signal })
    const startupBackendLog = readFileSync(logs.backend, "utf8")
    assertNoMutableRepoDatabasePaths(startupBackendLog, repoRoot)
    assertOnlyLoopbackHttpRequests(startupBackendLog)
    health.before = true

    const listResult = await runCommand(commands.playwrightList, logs.list, { signal })
    signal?.throwIfAborted()
    if (listResult.code !== 0) throw new Error(`Playwright list failed with exit code ${listResult.code}`)
    registeredCases = collectPlaywrightCases(readPlaywrightReport(commands.listJsonPath))
    listed = countRegisteredCases(registeredCases)
    for (const project of options.projects) {
      if (!listed[project]) throw new Error(`Playwright listed no tests for ${project}`)
    }

    inventory = inventoryProjects(frontendRoot, options.projects)
    writeFileSync(inventoryPath, `${JSON.stringify(inventory, null, 2)}\n`, "utf8")

    const testResult = await runCommand(commands.playwrightRun, logs.run, { signal })
    const outcome = readPlaywrightOutcome(commands.jsonPath, { signal })
    playwrightReport = outcome.report
    if (playwrightReport) {
      results = summarizePlaywrightReport(playwrightReport)
      skippedTests = collectSkippedTests(playwrightReport)
    }
    if (outcome.error) throw outcome.error
    assertProjectAccounting({
      projects: options.projects,
      listed,
      results,
      allowSkips: !options.failOnSkip,
      registeredCases,
      report: playwrightReport,
    })
    if (testResult.code !== 0) {
      status = "failed"
    } else if (options.failOnSkip && skippedTests.length) {
      status = "failed"
      runnerError = `Strict live-tier UAT rejected ${skippedTests.length} skipped test(s)`
    } else {
      status = "passed"
    }

    await waitForHttpOk(`${commands.urls.backend}/api/v1/health`, {
      timeoutMs: 5_000,
      headers: { "X-API-KEY": commands.backendEnv.SINGLE_USER_API_KEY },
      signal,
    })
    await waitForHttpOk(commands.urls.web, { timeoutMs: 5_000, signal })
    const completedBackendLog = readFileSync(logs.backend, "utf8")
    assertNoMutableRepoDatabasePaths(completedBackendLog, repoRoot)
    assertOnlyLoopbackHttpRequests(completedBackendLog)
    health.after = true
  } catch (error) {
    runnerError = error?.stack ?? error?.message ?? String(error)
    status = "failed"
  } finally {
    try {
      await stopSpawnedProcesses(processes)
    } catch (error) {
      runnerError = [runnerError, error?.stack ?? error?.message ?? String(error)].filter(Boolean).join("\n")
      status = "failed"
    }

    try {
      health.stopped = commands
        ? await Promise.all([
            probeStopped(commands.urls.mock),
            probeStopped(commands.urls.backend),
            probeStopped(commands.urls.web),
          ]).then((values) => values.every(Boolean))
        : processes.length === 0
      assertServicesStopped(health.stopped)
    } catch (error) {
      runnerError = [runnerError, error?.stack ?? error?.message ?? String(error)].filter(Boolean).join("\n")
      status = "failed"
    }

    if (deferred43Transaction?.receipt) {
      try {
        if (health.stopped) {
          try {
            deferred43Ledger = inspectDeferred43Ledgers(artifactRoot, deferred43RetainedBodies);
          } catch {
            deferred43Ledger = { files: 0, owning: 0, faults: 4, available: false };
          }
          deferred43Transaction.restore()
        } else {
          runnerError = [runnerError, "DEFERRED43_SERVICES_STILL_RUNNING"].filter(Boolean).join("\n")
          status = "failed"
        }
      } catch (error) {
        runnerError = [runnerError, error?.stack ?? error?.message ?? String(error)].filter(Boolean).join("\n")
        status = "failed"
      }
      // Cold retention uses the existing stdout and artifact paths after stopped restoration.
      const retention = { faults: 0, missing: [], malformed: [], oversize: [], records: [] };
      (() => {
        try {
          if (!health.stopped || !deferred43Transaction.ownsReceipt)
            throw new Error('DEFERRED43_RETAIN_RECEIPT_UNOWNED');
          const bytes = Buffer.from(JSON.stringify(deferred43Transaction.receipt));
          if (bytes.length > 1048576) {
            retention.faults |= 1;
            retention.oversize.push('receipt');
          } else {
            const stats = lstatSync(deferred43Transaction.receiptPath);
            if (
              !stats.isFile() ||
              stats.isSymbolicLink() ||
              stats.nlink !== 1 ||
              stats.uid !== process.getuid?.() ||
              realpathSync(deferred43Transaction.receiptPath) !==
                deferred43Transaction.receiptPath ||
              deferred43Hash(readFileSync(deferred43Transaction.receiptPath)) !==
                deferred43Hash(bytes)
            )
              throw new Error('DEFERRED43_RETAIN_RECEIPT_CHANGED');
            const frontend = realpathSync(frontendRoot),
              destination = path.resolve(artifactRoot);
            if (destination !== path.join(frontend, 'test-results/live-tier-uat', runId))
              throw new Error('DEFERRED43_RETAIN_ARTIFACT_SCOPE');
            const missingDirectories = [];
            for (
              let ancestor = destination;
              ancestor !== frontend;
              ancestor = path.dirname(ancestor)
            ) {
              if (!ancestor.startsWith(`${frontend}${path.sep}`))
                throw new Error('DEFERRED43_RETAIN_ARTIFACT_SCOPE');
              let stats;
              try {
                stats = lstatSync(ancestor);
              } catch (error) {
                if (error.code !== 'ENOENT') throw error;
                missingDirectories.unshift(ancestor);
                continue;
              }
              if (
                !stats.isDirectory() ||
                stats.isSymbolicLink() ||
                stats.uid !== process.getuid?.() ||
                realpathSync(ancestor) !== ancestor
              )
                throw new Error('DEFERRED43_RETAIN_ARTIFACT_ANCESTOR');
            }
            for (const directory of missingDirectories) {
              const parent = path.dirname(directory),
                stats = lstatSync(parent);
              if (
                !stats.isDirectory() ||
                stats.isSymbolicLink() ||
                stats.uid !== process.getuid?.() ||
                realpathSync(parent) !== parent
              )
                throw new Error('DEFERRED43_RETAIN_ARTIFACT_ANCESTOR');
              mkdirSync(directory, { mode: 0o700 });
              if (
                realpathSync(directory) !== directory ||
                lstatSync(directory).uid !== process.getuid?.()
              )
                throw new Error('DEFERRED43_RETAIN_ARTIFACT_ANCESTOR');
            }
            if (realpathSync(artifactRoot) !== destination)
              throw new Error('DEFERRED43_RETAIN_ARTIFACT_SCOPE');
            writeFileSync(path.join(artifactRoot, 'deferred43-source-receipt.json'), bytes, {
              flag: 'wx',
              mode: 0o600,
            });
            deferred43RetainedBodies.push({ kind: 'receipt', bytes });
          }
        } catch {
          retention.faults |= 4;
          retention.malformed.push('receipt');
        }
        try {
          if (!health.stopped) throw new Error('DEFERRED43_RETAIN_SERVICES_RUNNING');
          const report = playwrightReport;
          const matches = [],
            pending = [report];
          let visits = 0;
          while (pending.length) {
            const suite = pending.pop();
            if (
              !suite ||
              ++visits > 128 ||
              (suite.suites && (!Array.isArray(suite.suites) || suite.suites.length > 64)) ||
              (suite.specs && (!Array.isArray(suite.specs) || suite.specs.length > 64))
            )
              throw new Error('DEFERRED43_RETAIN_REPORT_SHAPE');
            pending.push(...(suite.suites ?? []));
            for (const spec of suite.specs ?? []) {
              if (
                spec.file !== 'e2e/ux-regression/notes-p0.spec.ts' ||
                spec.title !== "NE-04: 'Print / Save as PDF' prints the note without a pop-up error"
              )
                continue;
              if (!Array.isArray(spec.tests) || spec.tests.length > 8)
                throw new Error('DEFERRED43_RETAIN_TEST_SHAPE');
              for (const test of spec.tests)
                if (test.projectName === 'ux-regression') matches.push(test);
            }
          }
          if (
            matches.length !== 1 ||
            !Array.isArray(matches[0].results) ||
            matches[0].results.length !== 1 ||
            !Array.isArray(matches[0].results[0].attachments) ||
            matches[0].results[0].attachments.length > 32
          )
            throw new Error('DEFERRED43_RETAIN_TEST_SHAPE');
          const attachments = matches[0].results[0].attachments;
          const numeric = (value) => typeof value === 'number' && Number.isFinite(value);
          const keys = (value, expected) =>
            value &&
            typeof value === 'object' &&
            !Array.isArray(value) &&
            Object.keys(value).sort().join(',') === expected;
          const clock = (value) =>
            keys(value, 'after,before,local,origin') && Object.values(value).every(numeric);
          const id = (value) => typeof value === 'string' && /^[A-Za-z0-9_.:-]{0,128}$/.test(value);
          const sites = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 27, 28, 29, 30, 31, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56,
          ];
          for (const [kind, bound, realm] of [
            ['app', 1048576, 1],
            ['utility', 524288, 2],
            ['contexts', 65536, 0],
          ]) {
            const found = attachments.filter((entry) => entry.name === `deferred43-${kind}.json`);
            if (!found.length) continue;
            if (
              found.length !== 1 ||
              !keys(found[0], 'body,contentType,name') ||
              found[0].contentType !== 'application/json' ||
              typeof found[0].body !== 'string'
            ) {
              retention.faults |= 2;
              retention.malformed.push(kind);
              continue;
            }
            if (found[0].body.length > 4 * Math.ceil(bound / 3)) {
              retention.faults |= 1;
              retention.oversize.push(kind);
              continue;
            }
            const bytes = Buffer.from(found[0].body, 'base64');
            if (bytes.length > bound) {
              retention.faults |= 1;
              retention.oversize.push(kind);
              continue;
            }
            let payload;
            try {
              payload = JSON.parse(bytes.toString('utf8'));
            } catch {
              retention.faults |= 2;
              retention.malformed.push(kind);
              continue;
            }
            let valid =
              bytes.toString('base64') === found[0].body &&
              bytes.equals(Buffer.from(JSON.stringify(payload)));
            if (realm)
              valid &&=
                keys(
                  payload,
                  'count,counts,faults,first,order,realm,rows,schema,source,timeOrigin'
                ) &&
                payload.schema === 43 &&
                payload.source === deferred43Source &&
                payload.realm === realm &&
                numeric(payload.timeOrigin) &&
                Number.isSafeInteger(payload.count) &&
                payload.count > 0 &&
                payload.count <= (realm === 1 ? 4096 : 2048) &&
                Number.isSafeInteger(payload.faults) &&
                payload.faults >= 0 &&
                Number.isSafeInteger(payload.first) &&
                payload.first >= 0 &&
                payload.first <= payload.count &&
                payload.order === 0 &&
                Array.isArray(payload.counts) &&
                payload.counts.length === 4 &&
                payload.counts.every(
                  (value) => Number.isSafeInteger(value) && value >= 0 && value <= 512
                ) &&
                Array.isArray(payload.rows) &&
                payload.rows.length === payload.count &&
                payload.rows.every(
                  (row, index) =>
                    Array.isArray(row) &&
                    row.length === 24 &&
                    sites.includes(row[0]) &&
                    row[1] === index + 1 &&
                    row.every(numeric)
                );
            else
              valid &&=
                payload &&
                Object.keys(payload)
                  .filter((key) => key !== 'calibration')
                  .sort()
                  .join(',') ===
                  'clocks,contextLost,contexts,documents,faults,frameId,ordering,schema' &&
                payload.schema === 43 &&
                id(payload.frameId) &&
                Number.isSafeInteger(payload.documents) &&
                payload.documents >= 0 &&
                typeof payload.contextLost === 'boolean' &&
                Number.isSafeInteger(payload.faults) &&
                payload.faults >= 0 &&
                payload.ordering === 'ORDER_UNBOUND' &&
                (!Object.hasOwn(payload, 'calibration') || clock(payload.calibration)) &&
                Array.isArray(payload.clocks) &&
                payload.clocks.length <= 2 &&
                payload.clocks.every(
                  (value) =>
                    keys(value, 'after,before,local,origin,realm') &&
                    [1, 2].includes(value.realm) &&
                    Object.values(value).every(numeric)
                ) &&
                Array.isArray(payload.contexts) &&
                payload.contexts.length <= 2 &&
                payload.contexts.every(
                  (value) =>
                    keys(value, 'default,destroyed,frameId,id,isolated,uniqueId') &&
                    Number.isSafeInteger(value.id) &&
                    value.id > 0 &&
                    id(value.frameId) &&
                    id(value.uniqueId) &&
                    [value.default, value.isolated, value.destroyed].every(
                      (flag) => typeof flag === 'boolean'
                    )
                );
            if (!valid) {
              retention.faults |= 2;
              retention.malformed.push(kind);
              continue;
            }
            deferred43RetainedBodies.push({ kind, bytes });
          }
        } catch {
          retention.faults |= 256;
          retention.malformed.push('attachments');
        }
        try {
          const required = ['receipt', 'server', 'app', 'utility', 'contexts'];
          let emittedBytes = 0;
          for (const kind of required) {
            const entries = deferred43RetainedBodies.filter((entry) => entry.kind === kind);
            if (entries.length !== 1) {
              retention.faults |= 256;
              (entries.length ? retention.malformed : retention.missing).push(kind);
              continue;
            }
            const entry = entries[0];
            if (entry.fault) {
              retention.faults |= entry.fault;
              (entry.fault & 1 ? retention.oversize : retention.malformed).push(kind);
              continue;
            }
            if (!health.stopped || emittedBytes + entry.bytes.length > 3145728) {
              retention.faults |= 1;
              retention.oversize.push(kind);
              continue;
            }
            const encoded = entry.bytes.toString('base64'),
              digest = deferred43Hash(entry.bytes),
              parts = Math.ceil(encoded.length / 16384);
            for (let part = 0; part < parts; part++)
              process.stdout.write(
                `DEFERRED43_RETAIN ${JSON.stringify({ schema: 43, source: deferred43Source, kind, bytes: entry.bytes.length, sha256: digest, part: part + 1, parts, body: encoded.slice(part * 16384, (part + 1) * 16384) })}\n`
              );
            emittedBytes += entry.bytes.length;
            retention.records.push({ kind, bytes: entry.bytes.length, sha256: digest, parts });
          }
        } catch {
          retention.faults |= 4;
        }
        retention.complete = retention.faults === 0 && retention.records.length === 5;
        try {
          process.stdout.write(
            `DEFERRED43_RETENTION ${JSON.stringify({ schema: 43, source: deferred43Source, ...retention })}\n`
          );
        } catch {
          /* Retention cannot mask the original outcome. */
        }
      })();
      try {
      const line = `DEFERRED43_SUITE ${JSON.stringify({ schema: 43, restored: deferred43Transaction.receipt.restored, source: deferred43Source, ledger: deferred43Ledger })}\n`
      if (Buffer.byteLength(line) <= 8192) process.stdout.write(line)
      } catch { deferred43Ledger = { files: 0, owning: 0, faults: 4, available: false } }
    }

    if (ownsGeneratedTargets) {
      for (const cleanup of [
        {
          target: nextDistPath,
          expectedParent: frontendRoot,
          expectedPrefix: ".next-live-tier-",
        },
        ...(!options.preserveRuntime ? [{
          target: profileRoot,
          expectedParent: tmpdir(),
          expectedPrefix: "tldw-onboarding-uat-",
        }] : []),
      ]) {
        try {
          await removeGeneratedPath(cleanup.target, cleanup)
        } catch (error) {
          runnerError = [
            runnerError,
            `Generated cleanup failed for ${cleanup.target}: ${error?.message ?? String(error)}`,
          ].filter(Boolean).join("\n")
          status = "failed"
        }
      }
    }

    if (ownsArtifactRoot) {
      const artifacts = {
        root: artifactRoot,
        playwrightJson: commands?.jsonPath ?? path.join(artifactRoot, "playwright-results.json"),
        inventory: inventoryPath,
        logs,
      }
      const report = renderMarkdownReport({
        runId,
        commit: startedCommit,
        listed,
        results,
        inventory,
        health,
        artifacts,
        certification: isCertificationRun(),
        report: playwrightReport,
        skippedTests,
        error: runnerError,
      })
      writeFileSync(reportPath, report, "utf8")
      writeFileSync(summaryPath, `${JSON.stringify({ runId, status, ports, listed, results, skippedTests, health, artifacts, reportPath, error: runnerError }, null, 2)}\n`, "utf8")
    }
  }

  return { status, runId, ports, listed, results, skippedTests, inventory, health, artifactRoot, reportPath, summaryPath, error: runnerError }
}

function countRegisteredCases(cases) {
  const listed = {}
  for (const entry of cases) listed[entry.project] = (listed[entry.project] ?? 0) + 1
  return listed
}

async function listLiveTierCases({ options, frontendRoot, baseEnv, signal }) {
  const runId = options.runId ?? runIdNow()
  const artifactRoot = path.join(frontendRoot, "test-results/live-tier-uat", runId)
  const jsonPath = path.join(artifactRoot, "playwright-list.json")
  const logPath = path.join(artifactRoot, "playwright-list.log")
  let listed = {}
  let status = "failed"
  let error = null
  try {
    assertFreshRunTargets([artifactRoot])
    mkdirSync(artifactRoot, { recursive: true })
    const command = buildCollectionCommand({ env: baseEnv })
    command.cwd = frontendRoot
    command.args[0] = path.join(frontendRoot, "node_modules/@playwright/test/cli.js")
    command.args.push(...playwrightProjectArgs(options.projects))
    if (options.grep) command.args.push("--grep", options.grep)
    command.env.PLAYWRIGHT_JSON_OUTPUT_NAME = jsonPath
    const result = await runCommand(command, logPath, { signal })
    signal?.throwIfAborted()
    if (result.code !== 0) throw new Error(`Playwright list failed with exit code ${result.code}`)
    const report = readPlaywrightReport(jsonPath)
    if (report.errors?.length) throw new Error("Playwright collection reported errors")
    listed = countRegisteredCases(collectPlaywrightCases(report))
    for (const project of options.projects) {
      if (!listed[project]) throw new Error(`Playwright listed no tests for ${project}`)
    }
    status = "listed"
  } catch (caught) {
    error = caught?.message ?? String(caught)
  }
  return { status, runId, listed, ports: {}, artifactRoot, error }
}

const isEntrypoint = process.argv[1] === fileURLToPath(import.meta.url)
if (isEntrypoint) {
  const termination = installTerminationHandlers()
  const result = await runLiveTierUat({ signal: termination.signal })
  termination.dispose()
  if (result.error) process.stderr.write(`${redactText(result.error)}\n`)
  const signalName = termination.signal.reason?.signalName
  process.exitCode = signalName === "SIGINT"
    ? 130
    : signalName === "SIGTERM"
      ? 143
      : result.status === "passed" || result.status === "listed" || result.status === "help"
        ? 0
        : 1
}
