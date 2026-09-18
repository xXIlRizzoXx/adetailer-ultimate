"""Stylesheet and front-end script regressions; no browser or WebUI needed.

The JavaScript checks run the real scripts under Node.js against a tiny fake
DOM shaped like Gradio's output. They are skipped when Node.js is missing.
"""

from __future__ import annotations

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
        window, document, MutationObserver, console,
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
