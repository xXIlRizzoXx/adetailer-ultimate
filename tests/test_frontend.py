"""Stylesheet and front-end script regressions; no browser or WebUI needed.

The JavaScript checks run the real scripts under Node.js against a tiny fake
DOM shaped like Gradio's output. They are skipped when Node.js is missing.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WHITE = re.compile(r"color:\s*(#fff\b|#ffffff\b|white\b|rgba?\(\s*255,\s*255,\s*255)", re.I)


def css_rules() -> list[tuple[list[str], str]]:
    text = (ROOT / "style.css").read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return [
        ([s.strip() for s in selectors.split(",")], body)
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text)
    ]


def test_white_text_is_limited_to_the_dark_theme():
    # On the light theme, white labels and placeholders were white on white:
    # the prompt boxes (no visible labels) looked empty and alike.
    for selectors, body in css_rules():
        if WHITE.search(body) and "background" not in body:
            assert all(s.startswith(".dark ") for s in selectors), selectors
    labels = [s for s, _ in css_rules() if any("ad-section-label" in x for x in s)]
    assert any(all(x.startswith(".dark ") for x in s) for s in labels)


def test_amber_status_text_is_limited_to_the_dark_theme():
    # On the light theme, amber text on the near-white Detection preview /
    # Run ADetailer status pill was about 1.6:1, barely readable.
    amber = re.compile(r"(?<![-\w])color:\s*#fbbf24\b", re.I)
    found = False
    for selectors, body in css_rules():
        if amber.search(body):
            found = True
            assert all(s.startswith(".dark ") for s in selectors), selectors
    assert found  # the dark theme keeps its amber


def _contrast_on_white(hex_colour: str) -> float:
    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    lum = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    return 1.05 / (lum + 0.05)


def test_guide_link_accent_and_badge_dimming_are_limited_to_the_dark_theme():
    # On the light theme the light-blue "Guide" link, dimmed by the version
    # badge's 0.55 opacity (a child cannot undo it), was about 1.6:1 on white.
    accent = re.compile(r"(?<![-\w])color:\s*#6ea8fe\b", re.I)
    dim = re.compile(r"(?<![-\w])opacity:\s*0?\.\d")
    found = 0
    for selectors, body in css_rules():
        badge_p = any(s.endswith(".ad-version-overlay p") for s in selectors)
        if accent.search(body) or (badge_p and dim.search(body)):
            found += 1
            assert all(s.startswith(".dark ") for s in selectors), selectors
    assert found == 2  # the dark theme keeps its accent and its dimming
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    link = rules['div[id*="adetailer_ad_version"].ad-version-overlay .ad-guide-open']
    colour = re.search(r"(?<![-\w])color:\s*(#[0-9a-f]{6})\b", link, re.I).group(1)
    assert _contrast_on_white(colour) >= 4.5


def test_version_badge_inner_copy_stays_in_normal_flow():
    # Gradio 3 repeats the badge's id and class on its inner .prose div; that
    # copy must not be positioned (offset) a second time.
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    body = rules['div.prose[id*="adetailer_ad_version"].ad-version-overlay']
    assert re.search(r"position:\s*static\s*!important", body)


def test_faded_status_block_is_hidden_by_a_class():
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    body = rules["div.block.ad-preset-status.ad-status-faded"]
    assert re.search(r"display:\s*none\s*!important", body)


def _node() -> str | None:
    found = shutil.which("node")
    if found is None and os.environ.get("ProgramFiles"):
        # Default Windows install folder, for runs with a reduced PATH.
        found = shutil.which("node", path=os.path.join(os.environ["ProgramFiles"], "nodejs"))
    return found


NODE = _node()
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

# Just enough DOM for the scripts: elements with ids, classes, children and
# text, simple selectors (tag, .class, [attr*="v"], [attr="v"], lists).
DOM = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
class Text {
    constructor(t) { this.nodeType = 3; this.data = t; this.parentElement = null; }
    get textContent() { return this.data; }
    set textContent(v) { this.data = v; }
    cloneNode() { return new Text(this.data); }
}
class El {
    constructor(tag, attrs, kids) {
        attrs = attrs || {};
        this.nodeType = 1; this.tagName = tag.toUpperCase();
        this.id = attrs.id || ""; this.className = attrs.class || "";
        this.type = attrs.type || ""; this.value = attrs.value || "";
        this.checked = !!attrs.checked; this.title = "";
        this.style = { display: attrs.display || "block" };
        this.children = []; this.parentElement = null; this.events = [];
        const el = this;
        this.classList = {
            contains: (c) => el.className.split(/\s+/).includes(c),
            add: (c) => { if (!el.classList.contains(c)) el.className = (el.className + " " + c).trim(); },
            remove: (c) => { el.className = el.className.split(/\s+/).filter((x) => x && x !== c).join(" "); },
        };
        (kids || []).forEach((k) => this.append(typeof k === "string" ? new Text(k) : k));
    }
    append(k) { k.parentElement = this; this.children.push(k); return k; }
    get textContent() { return this.children.map((c) => c.textContent).join(""); }
    set innerHTML(v) { this.children = []; if (v) this.append(new Text(v)); }
    cloneNode() {
        const c = new El(this.tagName, { id: this.id, class: this.className });
        this.children.forEach((k) => c.append(k.cloneNode()));
        return c;
    }
    remove() {
        const p = this.parentElement;
        if (p) { p.children = p.children.filter((x) => x !== this); this.parentElement = null; }
    }
    matches(sel) {
        return sel.split(",").some((s) => {
            const m = s.trim().match(/^([a-z]*)((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i);
            if (!m) throw new Error("unsupported selector " + s);
            if (m[1] && m[1].toUpperCase() !== this.tagName) return false;
            if (!m[2].split(".").filter(Boolean).every((c) => this.classList.contains(c))) return false;
            const re = /\[([\w-]+)(\*?=)"([^"]*)"\]/g;
            let a;
            while ((a = re.exec(m[3]))) {
                const v = String(this[a[1]] || "");
                if (a[2] === "=" ? v !== a[3] : v.indexOf(a[3]) < 0) return false;
            }
            return true;
        });
    }
    querySelectorAll(sel) {
        const out = [];
        const walk = (n) => n.children.forEach((k) => {
            if (k.nodeType === 1) { if (k.matches(sel)) out.push(k); walk(k); }
        });
        walk(this);
        return out;
    }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
    closest(sel) {
        for (let n = this; n; n = n.parentElement) if (n.matches(sel)) return n;
        return null;
    }
    dispatchEvent(e) { this.events.push(e.type); }
}
const observers = [];
const timers = [];
const listeners = {};
function run(file, body, extraWindow) {
    const all = (sel) => body.querySelectorAll(sel);
    const document = {
        readyState: "complete", body,
        getElementById: (id) => body.querySelectorAll("div").concat(all("button"))
            .find((e) => e.id === id) || null,
        querySelectorAll: all,
        addEventListener: (t, f) => { (listeners[t] = listeners[t] || []).push(f); },
        createTreeWalker: (root) => {
            const texts = [];
            const walk = (n) => (n.children || []).forEach((k) => {
                if (k.nodeType === 3) texts.push(k); else walk(k);
            });
            walk(root);
            return { nextNode: () => texts.shift() || null };
        },
    };
    const window = Object.assign({
        getComputedStyle: (e) => ({ display: e.style.display }),
        requestAnimationFrame: (f) => f(),
    }, extraWindow || {});
    class MutationObserver {
        constructor(cb) { this.cb = cb; observers.push(this); }
        observe(target, options) { this.target = target; this.options = options; }
    }
    const ctx = {
        window, document, MutationObserver, console, NodeFilter: { SHOW_TEXT: 4 },
        Event: class { constructor(t) { this.type = t; } },
        setTimeout: (f) => { timers.push(f); return timers.length; },
        clearTimeout: (id) => { timers[id - 1] = null; },
    };
    vm.createContext(ctx);
    vm.runInContext(fs.readFileSync(file, "utf8"), ctx);
}
function fireTimers() {
    const pending = timers.splice(0);
    pending.forEach((f) => f && f());
}
"""


def run_js(tmp_path: Path, script: str, scenario: str):
    driver = tmp_path / "driver.js"
    path = json.dumps(str(ROOT / "javascript" / script))
    driver.write_text(DOM + f"const SCRIPT = {path};\n" + scenario, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(driver)], capture_output=True, encoding="utf-8", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


CLASS_SYNC = r"""
const P = "script_txt2img_adetailer_";
const token = (s) => new El("div", { class: "token" }, [
    new El("span", {}, [s]), new El("div", { class: "token-remove" }, ["x"])]);
const include = new El("textarea", { value: CASE.backstop });
const body = new El("body", {}, [
    new El("div", { id: P + "ad_model" }, [new El("input", { value: "faces.pt" })]),
    new El("div", { id: P + "ad_model_classes_dropdown" }, [
        new El("div", { class: "wrap" }, [
            new El("div", { class: "wrap-inner" }, CASE.tokens.map(token))])]),
    new El("div", { id: P + "ad_model_classes", display: "none" }, [include]),
    new El("div", { id: P + "ad_model_classes_excluded", display: "none" }, [
        new El("textarea", {})]),
    new El("div", { id: P + "ad_model_classes_exclude" }, [
        new El("input", { type: "checkbox" })]),
]);
run(SCRIPT, body, { localization: CASE.localization });
// Generate click: the capture-phase listener syncs every tab.
(listeners.click || []).forEach((f) => f({ target: new El("button") }));
console.log(JSON.stringify(include.value));
"""


@needs_node
@pytest.mark.parametrize(
    ("tokens", "localization", "backstop", "expected"),
    [
        # The WebUI translated the token text: keep the real value.
        (["visage"], {"face": "visage"}, "face", "face"),
        (["visage", "hand"], {"face": "visage"}, "face,hand", "face,hand"),
        # Untranslated tokens are mirrored at once, as before.
        (["face", "hand"], {}, "", "face,hand"),
        (["face"], {"Generate": "Genera", "rtl": True}, "", "face"),
        (["face"], {"face": "face"}, "", "face"),
    ],
)
def test_class_sync_never_writes_a_translated_class_name(
    tmp_path, tokens, localization, backstop, expected
):
    case = json.dumps(
        {"tokens": tokens, "localization": localization, "backstop": backstop}
    )
    scenario = f"const CASE = {case};\n" + CLASS_SYNC
    assert run_js(tmp_path, "class-sync.js", scenario) == expected


TOOLTIPS = r"""
const P = "script_txt2img_adetailer_";
const els = {
    reset: new El("button", { id: P + "ad_preset_reset" }),
    reset2: new El("button", { id: P + "ad_preset_reset_2nd" }),
    resetAll: new El("div", { id: P + "ad_preset_reset_all" }, [
        new El("label", {}, [new El("input", { type: "checkbox" }), new El("span", {}, ["Reset every tab"])])]),
    resetAll2: new El("div", { id: P + "ad_preset_reset_all_2nd" }),
    copy11: new El("button", { id: P + "ad_copy_settings_11th" }),
};
run(SCRIPT, new El("body", {}, Object.values(els)));
const out = {};
for (const k in els) out[k] = els[k].title;
console.log(JSON.stringify(out));
"""


@needs_node
def test_reset_tooltip_is_not_put_on_the_reset_every_tab_checkbox(tmp_path):
    titles = run_js(tmp_path, "button-tooltips.js", TOOLTIPS)
    assert titles["reset"].startswith("Reset this tab")
    assert titles["reset2"] == titles["reset"]
    assert titles["copy11"].startswith("Copy all of this tab")
    assert titles["resetAll"] == ""
    assert titles["resetAll2"] == ""


STATUS_FADE = r"""
// Gradio's Markdown: outer block, a wrapper that gets `pending` while a
// request writing this status runs, and an inner copy of the class.
const md = new El("span", { class: "md" }, [new El("p", {}, ["Saved preset 'faces'."])]);
const inner = new El("div", { class: "prose ad-preset-status" }, [md]);
const wrapper = new El("div", { class: "svelte-1" }, [inner]);
const status = new El("div", { class: "block ad-preset-status" }, [wrapper]);
run(SCRIPT, new El("body", {}, [status]));
const watchers = observers.filter((o) => o.target === status || o.target === inner);
const out = { watched: watchers.map((o) => o.target === status ? "outer" : "inner") };
const obs = watchers[0];
out.options = obs.options;
fireTimers();  // four seconds later
out.fadedText = md.textContent;
out.faded = status.classList.contains("ad-status-faded");
// The same message again: Gradio does not redraw it, only the wrapper's
// `pending` class comes and goes.
wrapper.className = "svelte-1 pending";
obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1" }]);
wrapper.className = "svelte-1";
obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1 pending" }]);
out.shownAgain = !status.classList.contains("ad-status-faded");
fireTimers();
out.fadedAgain = status.classList.contains("ad-status-faded");
console.log(JSON.stringify(out));
"""


@needs_node
def test_repeated_identical_status_is_shown_again(tmp_path):
    # The fade used to empty the text behind Gradio's back, so an identical
    # message (e.g. a retried Save that failed again) never came back.
    out = run_js(tmp_path, "preset-status-fade.js", STATUS_FADE)
    assert out["fadedText"] == "Saved preset 'faces'."
    assert out["faded"] is True
    assert out["shownAgain"] is True
    assert out["fadedAgain"] is True
    assert out["options"]["attributes"] is True
    assert out["options"]["attributeOldValue"] is True
    # Only the outer block is hidden; Gradio copies the class inside it.
    assert out["watched"] == ["outer"]


def _tooltip(fragment: str) -> str:
    source = (ROOT / "javascript" / "button-tooltips.js").read_text(encoding="utf-8")
    return re.search(rf'{fragment}:\s*"([^"]*)"', source).group(1)


TRANSLATED_TOOLTIPS = r"""
const P = "script_txt2img_adetailer_";
const inner = new El("button", {});
const els = {
    load: new El("button", { id: P + "ad_preset_load" }),
    load2: new El("button", { id: P + "ad_preset_load_2nd" }),
    remove: new El("button", { id: P + "ad_preset_delete" }),
    exportBtn: new El("div", { id: P + "ad_preset_export_btn" }, [inner]),
};
run(SCRIPT, new El("body", {}, Object.values(els)), { localization: CASE });
const out = {};
for (const k in els) out[k] = els[k].title;
out.exportBtn = inner.title;
console.log(JSON.stringify(out));
"""


@needs_node
@pytest.mark.parametrize("translated", [True, False])
def test_tooltips_are_shown_in_the_webui_language(tmp_path, translated):
    # The WebUI's localizer translates a title only when its node is added,
    # before this script sets it, so every tooltip stayed English.
    load = _tooltip("adetailer_ad_preset_load")
    export = _tooltip("adetailer_ad_preset_export_btn")
    remove = _tooltip("adetailer_ad_preset_delete")
    localization = (
        {load: "LOAD (translated)", export: "EXPORT (translated)", "rtl": True}
        if translated
        else {}
    )
    scenario = f"const CASE = {json.dumps(localization)};\n" + TRANSLATED_TOOLTIPS
    titles = run_js(tmp_path, "button-tooltips.js", scenario)
    assert titles["load"] == ("LOAD (translated)" if translated else load)
    assert titles["load2"] == titles["load"]
    assert titles["exportBtn"] == ("EXPORT (translated)" if translated else export)
    assert titles["remove"] == remove  # no translation: English


EXPORT_TOOLTIPS = r"""
const P = "script_txt2img_adetailer_";
const inner = new El("button", {});
const els = {
    // Gradio 3: the plain fallback button carries the id and the class.
    gradio3: new El("button", {
        id: P + "ad_preset_export_btn", class: "lg secondary gradio-button ad-export-unavailable",
    }),
    gradio4: new El("div", { id: P + "ad_preset_export_btn_2nd" }, [inner]),
};
run(SCRIPT, new El("body", {}, Object.values(els)), { localization: CASE });
console.log(JSON.stringify({ gradio3: els.gradio3.title, gradio4: inner.title }));
"""


@needs_node
@pytest.mark.parametrize("translated", [True, False])
def test_export_tooltip_tells_gradio_3_that_it_cannot_download(tmp_path, translated):
    # On AUTOMATIC1111 (Gradio 3) Export cannot download a file, but its
    # tooltip promised a JSON file.
    export = _tooltip("adetailer_ad_preset_export_btn")
    source = (ROOT / "javascript" / "button-tooltips.js").read_text(encoding="utf-8")
    match = re.search(r'EXPORT_UNAVAILABLE =\s*"([^"]*)"', source)
    fallback = match.group(1) if match else "(missing)"
    localization = {fallback: "NO DOWNLOAD (translated)"} if translated else {}
    scenario = f"const CASE = {json.dumps(localization)};\n" + EXPORT_TOOLTIPS
    titles = run_js(tmp_path, "button-tooltips.js", scenario)
    assert titles["gradio4"] == export
    if translated:
        assert titles["gradio3"] == "NO DOWNLOAD (translated)"
    else:
        assert titles["gradio3"] != export
        assert titles["gradio3"] == fallback
        assert "user_presets.json" in fallback
        assert "Gradio 4" in fallback


CHANGED_LABEL = r"""
const tabs = [new Text("📥 Paste settings"), new Text("📥 Paste settings")];
const other = new Text("plain");
const body = new El("body", {}, [
    ...tabs.map((t) => new El("button", {}, [t])), new El("button", {}, [other])]);
run(SCRIPT, body, { localization: CASE });
const obs = observers[observers.length - 1];
// Gradio (Svelte) rewrites a button's text node in place when its value
// changes: one characterData record, no added node.
const change = (t, text) => {
    t.data = text;
    obs.cb([{ type: "characterData", target: t, addedNodes: [] }]);
    return t.data;
};
const out = { options: obs.options, swept: tabs.map((t) => t.data) };
out.copied = change(tabs[1], "📥 Paste from 1st tab");  // runtime text, no key
out.reset = change(tabs[1], "📥 Paste settings");
obs.cb([{ type: "characterData", target: tabs[1], addedNodes: [] }]);
out.again = tabs[1].data;
out.cycle = change(other, "🔁 A");  // never written: its translation is a key
console.log(JSON.stringify(out));
"""


@needs_node
def test_paste_label_stays_translated_after_copy_and_reset(tmp_path):
    # Copy and Reset set the Paste button's label back to English in place;
    # only added nodes were translated, so it stayed English.
    localization = {
        "📥 Paste settings": "📥 PASTE (translated)",
        "🔁 A": "🔁 B",
        "🔁 B": "🔁 A",
    }
    scenario = f"const CASE = {json.dumps(localization)};\n" + CHANGED_LABEL
    out = run_js(tmp_path, "localize_emoji_buttons.js", scenario)
    assert out["options"]["characterData"] is True
    assert out["swept"] == ["📥 PASTE (translated)"] * 2
    assert out["copied"] == "📥 Paste from 1st tab"
    assert out["reset"] == "📥 PASTE (translated)"
    assert out["again"] == "📥 PASTE (translated)"
    assert out["cycle"] == "🔁 A"


def _export_js() -> str:
    tree = ast.parse((ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8"))
    return next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_EXPORT_JS"
    )


EXPORT_STEP = r"""
const clicked = [];
globalThis.document = {
    createElement: () => ({ click() { clicked.push([this.href, this.download]); },
                            remove() {} }),
    body: { appendChild() {} },
};
// Gradio 4.40's wrapper for a front-end-only step with one output.
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const step = new AsyncFunction("__fn_args", `
  let result = await (${EXPORT_JS})(...__fn_args);
  if (typeof result === "undefined") return [];
  return (true && !Array.isArray(result)) ? [result] : result;`);
(async () => {
    const out = {};
    out.fresh = await step([{ url: "/file=gradio/1a2b/adetailer-ultimate-presets.json",
                              orig_name: null }]);
    out.cleared = await step([null]);
    out.clicked = clicked;
    console.log(JSON.stringify(out));
})();
"""


@needs_node
def test_export_step_downloads_the_new_file_once_and_clears_it(tmp_path):
    # Gradio 4's DownloadButton only downloads the value it already holds when
    # clicked: the first Export got nothing and later ones the previous file.
    driver = tmp_path / "export.js"
    driver.write_text(
        f"const EXPORT_JS = {json.dumps(_export_js())};\n" + EXPORT_STEP,
        encoding="utf-8",
    )
    result = subprocess.run(
        [NODE, str(driver)], capture_output=True, encoding="utf-8", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    # One download of the file this click wrote; the value is then cleared,
    # so the button never serves it again on the next click.
    assert out["clicked"] == [
        ["/file=gradio/1a2b/adetailer-ultimate-presets.json",
         "adetailer-ultimate-presets.json"]
    ]
    assert out["fresh"] == [None]
    assert out["cleared"] == [None]


def _workflow_paths(event: str) -> list[str]:
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    block = re.search(rf"^  {event}:\n(.*?)(?=^  \S)", text, re.M | re.S).group(1)
    return re.findall(r'^\s+- "([^"]+)"', block, re.M)


def _glob(pattern: str) -> re.Pattern[str]:
    # GitHub's path filters: "**" crosses folders, "*" does not.
    parts = [re.escape(p).replace(r"\*", "[^/]*") for p in pattern.split("**")]
    return re.compile(".*".join(parts) + r"\Z")


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_ci_runs_for_every_non_python_file_the_tests_read(event):
    # The checks of the stylesheet, the scripts, the README and the changelog
    # never ran on a change to those files alone.
    patterns = [_glob(p) for p in _workflow_paths(event)]
    files = ["style.css", "README.md", "CHANGELOG.md"] + [
        f"javascript/{js.name}" for js in sorted((ROOT / "javascript").glob("*.js"))
    ]
    for name in files:
        assert any(p.match(name) for p in patterns), name
