// serve/test_monitor.py ChatRequest: runs serve/web/app.js under node with a fake DOM, applies the sampling settings
// given as JSON (argv: app.js path, settings), sends "hi" through the chat's own send(), and prints the request body
// it POSTs to v1/chat/completions, the shared defaults it would save for other apps, and budgetProblem's verdicts.
// With "lost" as a third argument the stream sends one piece of the answer and then fails as Chrome's does ("network
// error"): it prints the request's id header, what the page POSTs to /cancel and the message it keeps. With "down" no
// server answers (fetch rejects, as when the page outlives its server): it prints the message the page keeps.
const vm = require("vm"), fs = require("fs");
function any() {                                       // a stand-in for any DOM object: every property/call works
  const f = function () {};
  return new Proxy(f, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => "";
      if (k === Symbol.iterator) return function* () {};
      if (k === "then") return undefined;
      if (k === "length") return 0;
      if (k in t) return t[k];
      return any();
    },
    set(t, k, v) { t[k] = v; return true; },
    apply() { return any(); }, construct() { return any(); },
  });
}
const els = {};                                        // elements keep what is set on them (the input's value)
const el = (id) => (els[id] = els[id] || any());
let posted = null, postedHeaders = null, cancelled = null;
const lost = process.argv[4] === "lost";
const down = process.argv[4] === "down";                // no server: fetch rejects as browsers do ("Failed to fetch")
const piece = 'data: {"choices": [{"index": 0, "delta": {"content": "partial"}}]}\n\n';
const brokenStream = () => {
  let n = 0;
  return {getReader: () => ({read: () => (n++ === 0 ? Promise.resolve({value: new TextEncoder().encode(piece), done: false})
                                                    : Promise.reject(new TypeError("network error")))})};
};
const ctx = {
  document: Object.assign(any(), {getElementById: el, querySelectorAll: () => [], addEventListener() {}}),
  window: Object.assign(any(), {addEventListener() {}}), location: {hash: "", search: "", pathname: "/", host: "127.0.0.1:8090"},
  localStorage: {getItem: () => null, setItem() {}}, matchMedia: () => ({matches: false, addEventListener() {}}),
  fetch: (url, opts) => {
    if (String(url).includes("chat/completions")) {
      posted = JSON.parse(opts.body);
      postedHeaders = opts.headers;
      if (lost) return Promise.resolve({ok: true, status: 200, body: brokenStream()});
      if (down) return Promise.reject(new TypeError("Failed to fetch"));
    }
    if (String(url) === "cancel") cancelled = JSON.parse(opts.body);
    return new Promise(() => {});
  },
  innerHeight: 800, setTimeout() {}, setInterval() {}, clearTimeout() {}, clearInterval() {}, requestAnimationFrame() {}, AbortController,
  crypto, TextDecoder, TextEncoder, TypeError, performance,
  navigator: any(), history: any(), sessionStorage: {getItem: () => null, setItem() {}}, Image: function () {},
  URLSearchParams, console, Number, Math, String, Date, JSON, Promise, Map, Set, Array, Object, RegExp,
};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
ctx.__s = JSON.parse(process.argv[3]);
vm.runInContext("settings = {...DEFAULTS, ...__s}; busy = null;", ctx);
el("input").value = "hi";
(async () => {
const sent = vm.runInContext("send()", ctx);   // without "lost" the answer never comes: do not wait for it
if (lost || down) await sent;
const out = {
  body: posted,
  shared: vm.runInContext("sharedDefaults(settings)", ctx),
  problems: vm.runInContext(`[["", ""], ["", "2048"], ["4096", "2048"], ["4096", "4032"], ["4096", "4033"],
                              ["100", "50"], ["", "1.5"], ["", "-3"], ["", "0"]].map(([m, b]) => budgetProblem(m, b))`, ctx),
};
if (down) out.msg = vm.runInContext("({text: messages.at(-1).text, error: messages.at(-1).error})", ctx);
if (lost) Object.assign(out, {rid: postedHeaders["X-Strata-Request-Id"], cancelled,
                              msg: vm.runInContext("({text: messages.at(-1).text, error: messages.at(-1).error})", ctx)});
console.log(JSON.stringify(out));
})();
