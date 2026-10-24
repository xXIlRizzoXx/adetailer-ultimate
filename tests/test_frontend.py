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


def test_version_badge_leaves_room_for_the_title_and_the_tab_pill():
    # The badge is right-anchored with a fixed maximum width: on a narrow
    # panel (a window about half a 1920px screen) it slid over the "ADetailer"
    # title and the "2x Tabs" pill, on Gradio 3 and 4 alike.
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    outer = rules['div[id*="adetailer_ad_version"].ad-version-overlay']
    cap = re.search(
        r"max-width:\s*min\(\s*460px,\s*calc\(\s*100%\s*-\s*(\d+)px\s*\)\s*\)\s*!important",
        outer,
    )
    assert cap
    # The chevron offset (60px) plus the title, the Enable checkbox and the pill.
    assert int(cap.group(1)) >= 200
    # Browsers without min() keep the plain cap, declared first.
    fallback = re.search(r"max-width:\s*460px\s*!important", outer)
    assert fallback and fallback.start() < cap.start()
    # Gradio 3's inner copy matches the rule above too; it must fill the outer
    # badge instead of taking the room off a second time.
    inner = rules['div.prose[id*="adetailer_ad_version"].ad-version-overlay']
    assert re.search(r"max-width:\s*100%\s*!important", inner)
    # The text is cut at its end, so the Guide link at the start stays.
    text = rules['div[id*="adetailer_ad_version"].ad-version-overlay p']
    assert re.search(r"overflow:\s*hidden", text)
    assert re.search(r"text-overflow:\s*ellipsis", text)
    assert re.search(r"white-space:\s*nowrap", text)


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
    appendChild(k) { return this.append(k); }
    insertBefore(k, ref) {
        const at = this.children.indexOf(ref);
        if (at < 0) return this.append(k);
        k.parentElement = this; this.children.splice(at, 0, k); return k;
    }
    get nextElementSibling() {
        const p = this.parentElement;
        if (!p) return null;
        return p.children.slice(p.children.indexOf(this) + 1).find((k) => k.nodeType === 1) || null;
    }
    get textContent() { return this.children.map((c) => c.textContent).join(""); }
    set textContent(v) { this.children = []; this.append(new Text(String(v))); }
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
    addEventListener(t, f) { (this.handlers = this.handlers || {})[t] = (this.handlers[t] || []).concat([f]); }
    getBoundingClientRect() { return this.rect || { left: 0, top: 0, width: 0, height: 0 }; }
    // A real click: document capture listeners first, then Gradio toggles
    // the section whose header was clicked.
    click() {
        this.clicks = (this.clicks || 0) + 1;
        (listeners.click || []).forEach((f) => f({ target: this }));
        const header = this.closest(".label-wrap");
        if (header) {
            if (header.classList.contains("open")) header.classList.remove("open");
            else header.classList.add("open");
        }
    }
}
const observers = [];
const timers = [];
const intervals = [];
const listeners = {};
function run(file, body, extraWindow) {
    const all = (sel) => body.querySelectorAll(sel);
    const document = {
        readyState: "complete", body,
        getElementById: (id) => body.querySelectorAll("div").concat(all("button"))
            .find((e) => e.id === id) || null,
        querySelectorAll: all,
        createElement: (tag) => new El(tag),
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
    run.document = document;  // a scenario may set, e.g., activeElement
    const window = Object.assign({
        getComputedStyle: (e) => ({ display: e.style.display }),
        requestAnimationFrame: (f) => f(),
    }, typeof extraWindow === "function" ? {} : extraWindow || {});
    // A function may define getters (for example one that throws).
    if (typeof extraWindow === "function") extraWindow(window);
    class MutationObserver {
        constructor(cb) { this.cb = cb; observers.push(this); }
        observe(target, options) { this.target = target; this.options = options; }
    }
    const ctx = {
        window, document, MutationObserver, console, NodeFilter: { SHOW_TEXT: 4 },
        Event: class { constructor(t) { this.type = t; } },
        setTimeout: (f) => { timers.push(f); return timers.length; },
        clearTimeout: (id) => { timers[id - 1] = null; },
        setInterval: (f) => { intervals.push(f); return intervals.length; },
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


STATUS_FADE_FAST = r"""
const P = "script_txt2img_adetailer_";
function statusBlock(id) {
    const md = new El("span", { class: "md" }, [new El("p", {}, ["Enter a preset name first."])]);
    const wrapper = new El("div", { class: "svelte-1" }, [
        new El("div", { class: "prose ad-preset-status" }, [md])]);
    return new El("div", { id: P + id, class: "block ad-preset-status" }, [wrapper]);
}
const save = new El("span", {}, ["Save"]);
const exportBtn = new El("button", {}, [new El("span", {}, ["Export"])]);
const resetAll = new El("input", { type: "checkbox" });
const els = {
    status: statusBlock("ad_preset_status"),
    status2: statusBlock("ad_preset_status_2nd"),
    io: statusBlock("ad_preset_io_status"),
};
run(SCRIPT, new El("body", {}, [
    ...Object.values(els),
    new El("button", { id: P + "ad_preset_save" }, [save]),
    new El("button", { id: P + "ad_preset_save_2nd" }),
    // Gradio 4's DownloadButton: the id is on a wrapper.
    new El("div", { id: P + "ad_preset_export_btn" }, [exportBtn]),
    new El("div", { id: P + "ad_preset_reset_all" }, [new El("label", {}, [resetAll])]),
]));
const obs = observers.find((o) => o.target === els.status);
const faded = () => Object.fromEntries(
    Object.entries(els).map(([k, e]) => [k, e.classList.contains("ad-status-faded")]));
const out = {};
fireTimers();
out.start = faded();
// Gradio 4 answers within one frame: pending and complete land in the same
// flush and the unchanged message makes no DOM change at all.
save.click();
fireTimers();
out.afterSave = faded();
fireTimers();
out.fadesAgain = faded();
// A slower answer shows its `pending` class: the normal path handles it,
// so nothing is shown before the answer arrives.
save.click();
const wrapper = els.status.children[0];
wrapper.className = "svelte-1 pending";
obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1" }]);
fireTimers();
out.whilePending = faded();
wrapper.className = "svelte-1";
obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1 pending" }]);
out.answered = faded();
fireTimers();
exportBtn.children[0].click();
fireTimers();
out.afterExport = faded();
fireTimers();
resetAll.click();
fireTimers();
out.afterResetAll = faded();
console.log(JSON.stringify(out));
"""


@needs_node
def test_identical_status_answered_within_one_frame_is_shown_again(tmp_path):
    # On Gradio 4 a queue=False answer that arrives within one animation frame
    # never shows the `pending` class, and the same message causes no DOM
    # change: a retried Save that failed again showed nothing.
    out = run_js(tmp_path, "preset-status-fade.js", STATUS_FADE_FAST)
    everything = {"status": True, "status2": True, "io": True}
    assert out["start"] == everything
    assert out["afterSave"] == {**everything, "status": False}  # this tab only
    assert out["fadesAgain"] == everything
    assert out["whilePending"] == everything
    assert out["answered"] == {**everything, "status": False}
    assert out["afterExport"] == {**everything, "io": False}
    assert out["afterResetAll"] == everything


STATUS_FADE_PASTE = r"""
const P = "script_txt2img_adetailer_";
function statusBlock(id) {
    const md = new El("span", { class: "md" }, [new El("p", {}, ["Pasted into every tab."])]);
    const wrapper = new El("div", { class: "svelte-1" }, [
        new El("div", { class: "prose ad-preset-status" }, [md])]);
    return new El("div", { id: P + id, class: "block ad-preset-status" }, [wrapper]);
}
const paste = new El("span", {}, ["Paste"]);
const paste2 = new El("span", {}, ["Paste"]);
const swap = new El("input", { type: "checkbox" });
const every2 = new El("input", { type: "checkbox" });
const els = {
    status: statusBlock("ad_preset_status"),
    status2: statusBlock("ad_preset_status_2nd"),
};
run(SCRIPT, new El("body", {}, [
    ...Object.values(els),
    new El("button", { id: P + "ad_paste_settings" }, [paste]),
    new El("div", { id: P + "ad_paste_swap" }, [new El("label", {}, [swap])]),
    new El("div", { id: P + "ad_paste_all" }, [new El("label", {}, [new El("input", { type: "checkbox" })])]),
    new El("button", { id: P + "ad_paste_settings_2nd" }, [paste2]),
    new El("div", { id: P + "ad_paste_swap_2nd" }, [new El("label", {}, [new El("input", { type: "checkbox" })])]),
    new El("div", { id: P + "ad_paste_all_2nd" }, [new El("label", {}, [every2])]),
]));
const faded = () => Object.fromEntries(
    Object.entries(els).map(([k, e]) => [k, e.classList.contains("ad-status-faded")]));
const out = {};
fireTimers();
out.start = faded();
// A plain Paste writes no status: an old message must stay hidden.
paste.click();
fireTimers();
out.plain = faded();
// With a scope box ticked the same message comes back within one frame.
swap.checked = true;
paste.click();
fireTimers();
out.swap = faded();
fireTimers();
every2.checked = true;
paste2.click();
fireTimers();
out.every2 = faded();
console.log(JSON.stringify(out));
"""


@needs_node
def test_paste_with_a_scope_box_shows_its_identical_status_again(tmp_path):
    # Swap and "Paste into every tab" write the tab's status; on Gradio 4 the
    # same message answered within one frame changes nothing in the DOM.
    out = run_js(tmp_path, "preset-status-fade.js", STATUS_FADE_PASTE)
    everything = {"status": True, "status2": True}
    assert out["start"] == everything
    assert out["plain"] == everything
    assert out["swap"] == {**everything, "status": False}
    assert out["every2"] == {**everything, "status2": False}


STATUS_FADE_PLAIN_PASTE = r"""
const P = "script_txt2img_adetailer_";
function statusBlock(id) {
    const md = new El("span", { class: "md" }, [new El("p", {}, ["Loaded 'faces'."])]);
    const wrapper = new El("div", { class: "svelte-1" }, [
        new El("div", { class: "prose ad-preset-status" }, [md])]);
    return new El("div", { id: P + id, class: "block ad-preset-status" }, [wrapper]);
}
const paste = new El("span", {}, ["Paste"]);
const els = { status: statusBlock("ad_preset_status"), status2: statusBlock("ad_preset_status_2nd") };
run(SCRIPT, new El("body", {}, [
    ...Object.values(els),
    new El("button", { id: P + "ad_paste_settings" }, [paste]),
    new El("div", { id: P + "ad_paste_swap" }, [new El("label", {}, [new El("input", { type: "checkbox" })])]),
    new El("div", { id: P + "ad_paste_all" }, [new El("label", {}, [new El("input", { type: "checkbox" })])]),
]));
// Gradio 3: the request's `pending` class comes and goes on every output.
function requestEnds() {
    for (const block of Object.values(els)) {
        const wrapper = block.children[0];
        const obs = observers.find((o) => o.target === block);
        wrapper.className = "svelte-1 pending";
        obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1" }]);
        wrapper.className = "svelte-1";
        obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1 pending" }]);
    }
}
const faded = () => Object.fromEntries(
    Object.entries(els).map(([k, e]) => [k, e.classList.contains("ad-status-faded")]));
fireTimers();
const out = { start: faded() };
paste.click();
requestEnds();
out.afterPaste = faded();
// Another request writing the same message (a retried Load) still shows it.
requestEnds();
out.afterLoad = faded();
// Gradio 4: a fast Paste answer shows no end of its request at all. The
// block must not stay on: a later request with the same message shows it.
fireTimers();
paste.click();
fireTimers();  // the Paste block runs out
requestEnds();
out.afterQuietPaste = faded();
console.log(JSON.stringify(out));
"""


@needs_node
def test_a_plain_paste_does_not_bring_old_status_messages_back(tmp_path):
    # Paste lists every tab's status among its outputs: on Gradio 3 the end
    # of its request was read as an answer and the faded "Loaded 'faces'."
    # came back for 4 seconds in every tab.
    out = run_js(tmp_path, "preset-status-fade.js", STATUS_FADE_PLAIN_PASTE)
    everything = {"status": True, "status2": True}
    assert out["start"] == everything
    assert out["afterPaste"] == everything
    assert out["afterLoad"] == {"status": False, "status2": False}
    assert out["afterQuietPaste"] == {"status": False, "status2": False}


STATUS_FADE_IMPORT = r"""
const P = "script_txt2img_adetailer_";
function statusBlock(id) {
    const md = new El("span", { class: "md" }, [new El("p", {}, ["⏭ 1 not imported."])]);
    const wrapper = new El("div", { class: "svelte-1" }, [
        new El("div", { class: "prose ad-preset-status" }, [md])]);
    return new El("div", { id: P + id, class: "block ad-preset-status" }, [wrapper]);
}
const els = {
    status: statusBlock("ad_preset_status"),
    io: statusBlock("ad_preset_io_status"),
    io2: statusBlock("ad_preset_io_status_2nd"),
};
const config = { components: [
    { id: 20, props: { elem_id: P + "ad_preset_import_btn" } },
    { id: 21, props: { elem_id: P + "ad_preset_import_btn_2nd" } },
    { id: 22, props: { elem_id: P + "ad_preset_import_overwrite" } },
    { id: 23, props: {} },
] };
run(SCRIPT, new El("body", {}, Object.values(els)), CASE.config ? { gradio_config: config } : {});
const obs = observers.find((o) => o.target === els.io);
const faded = () => Object.fromEntries(
    Object.entries(els).map(([k, e]) => [k, e.classList.contains("ad-status-faded")]));
// Gradio's DOM event on its app root: it bubbles to the document.
const gradio = (id, event) => (listeners.gradio || []).forEach(
    (f) => f({ type: "gradio", detail: { id, event, data: null } }));
const out = { listens: (listeners.gradio || []).length > 0 };
fireTimers();
out.start = faded();
// The same import answered within one frame: no DOM change at all.
gradio(20, "upload");
fireTimers();
out.afterImport = faded();
fireTimers();
out.fadesAgain = faded();
gradio(21, "upload");
fireTimers();
out.afterImport2 = faded();
fireTimers();
// Other components and other events arm nothing.
gradio(22, "upload");
gradio(23, "upload");
gradio(99, "upload");
gradio(20, "change");
gradio(20, "click");
(listeners.gradio || []).forEach((f) => f({ type: "gradio" }));
fireTimers();
out.others = faded();
// A slower answer shows its `pending` class: the normal path handles it.
gradio(20, "upload");
const wrapper = els.io.children[0];
wrapper.className = "svelte-1 pending";
obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1" }]);
fireTimers();
out.whilePending = faded();
wrapper.className = "svelte-1";
obs.cb([{ type: "attributes", target: wrapper, oldValue: "svelte-1 pending" }]);
out.answered = faded();
console.log(JSON.stringify(out));
"""


@needs_node
@pytest.mark.parametrize("config", [True, False])
def test_identical_import_status_answered_within_one_frame_is_shown_again(
    tmp_path, config
):
    # Import answers a file upload, not a click, so the click arm above never
    # covered it: on Gradio 4 importing the same file again (every name
    # skipped, or unreadable) could show nothing at all.
    scenario = f"const CASE = {json.dumps({'config': config})};\n" + STATUS_FADE_IMPORT
    out = run_js(tmp_path, "preset-status-fade.js", scenario)
    everything = {"status": True, "io": True, "io2": True}
    assert out["listens"] is True
    assert out["start"] == everything
    assert out["fadesAgain"] == everything
    assert out["others"] == everything
    assert out["whilePending"] == everything
    assert out["answered"] == {**everything, "io": False}
    if config:
        assert out["afterImport"] == {**everything, "io": False}  # this tab only
        assert out["afterImport2"] == {**everything, "io2": False}
    else:
        # Without Gradio's config nothing can be mapped: nothing breaks.
        assert out["afterImport"] == everything
        assert out["afterImport2"] == everything


TAB_PILL = r"""
const P = "script_txt2img_adetailer_";
// The pill already in the header, so the script reuses it.
const pill = new El("span", { class: "ad-tab-count-pill" });
let pillText = "";
Object.defineProperty(pill, "textContent", { get: () => pillText, set: (v) => { pillText = v; } });
pill.classList.toggle = (c, on) => (on ? pill.classList.add(c) : pill.classList.remove(c));
const title = new El("span", {}, ["ADetailer", pill]);
const header = new El("div", { class: "label-wrap" }, [title]);
const box1 = new El("input", { type: "checkbox", checked: true });
const box2 = new El("input", { type: "checkbox" });
const acc = new El("div", { id: P + "ad_main_accordion" }, [
    header,
    new El("div", { id: P + "ad_tab_enable" }, [box1]),
    new El("div", { id: P + "ad_tab_enable_2nd" }, [box2]),
]);
// Selectors the fake DOM does not parse.
const find = El.prototype.querySelector;
acc.querySelector = function (sel) { return sel === ":scope > .label-wrap" ? header : find.call(this, sel); };
header.querySelector = function (sel) { return sel === ":scope > span" ? title : find.call(this, sel); };
acc.querySelectorAll = (sel) => {
    if (sel !== '[id*="_ad_tab_enable"] input[type="checkbox"]') throw new Error(sel);
    return [box1, box2];
};
run(SCRIPT, new El("body", {}, [acc]));
const gradio = (event) => (listeners.gradio || []).forEach(
    (f) => f({ type: "gradio", detail: { event, id: 7 } }));
const out = { start: pill.textContent };
// Paste settings switches tab 2 on: the server sets `checked`, which makes
// no DOM change and no user event (the fake observers never fire).
box2.checked = true;
gradio("select");
out.otherEvent = pill.textContent;
gradio("change");
out.pasted = pill.textContent;
box2.checked = false;  // Reset switches it off again
gradio("change");
out.reset = pill.textContent;
console.log(JSON.stringify(out));
"""


@needs_node
def test_tab_pill_follows_a_server_update_of_enable_this_tab(tmp_path):
    # On Gradio 4 a fast Paste settings / Load / Reset answer changes only the
    # checkbox's `checked` property: the pill kept its old count.
    out = run_js(tmp_path, "tab-count-pill.js", TAB_PILL)
    assert out == {
        "start": "1x Tab", "otherEvent": "1x Tab", "pasted": "2x Tabs", "reset": "1x Tab"
    }


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


CHANGED_VALUE = r"""
const box = (value) => { const i = new El("input", { value }); i.role = "listbox"; return i; };
const ckpt = box("Use same checkpoint");
const typed = box("Use same VAE");
const vae = box("Use same VAE");
const other = box("Use same checkpoint");  // not an ADetailer dropdown
const body = new El("body", {}, [
    new El("div", { id: "script_txt2img_adetailer_ad_main_accordion" }, [
        new El("div", { id: "script_txt2img_adetailer_ad_checkpoint" }, [ckpt]),
        new El("div", { id: "script_txt2img_adetailer_ad_vae" }, [typed]),
    ]),
    new El("div", { id: "script_img2img_adetailer_ad_main_accordion" }, [vae]),
    new El("div", { id: "setting_sd_model_checkpoint" }, [other]),
]);
run(SCRIPT, body, { localization: CASE });
const all = () => [ckpt.value, typed.value, vae.value, other.value];
// Gradio's DOM event on its app root, for any component.
const gradio = (event) => (listeners.gradio || []).forEach(
    (f) => f({ type: "gradio", detail: { event, id: 7, data: null } }));
const out = { swept: all() };
// A value change: Gradio 4 writes the English choice name into the input's
// `value` property, which is not a DOM mutation (the observer sees nothing).
ckpt.value = "model.safetensors";
gradio("change");
out.picked = ckpt.value;
ckpt.value = "Use same checkpoint";
typed.value = vae.value = "Use same VAE";
other.value = "Use same checkpoint";
run.document.activeElement = typed;  // the user is typing into this one
gradio("focus");
gradio("select");
(listeners.gradio || []).forEach((f) => f({ type: "gradio" }));
out.otherEvents = all();
gradio("change");
out.changed = all();
run.document.activeElement = null;
gradio("blur");
out.blurred = all();
out.listens = (listeners.gradio || []).length;
console.log(JSON.stringify(out));
"""


@needs_node
@pytest.mark.parametrize("translated", [True, False])
def test_dropdown_value_stays_translated_after_a_change(tmp_path, translated):
    # A value change (picking "Use same checkpoint" again, Reset, Load, Paste
    # settings) rewrote the dropdown's input value in English on Gradio 4, and
    # only added nodes were translated: it stayed English until a reload.
    localization = (
        {"Use same checkpoint": "USE SAME CHECKPOINT (translated)",
         "Use same VAE": "USE SAME VAE (translated)"}
        if translated
        else {}
    )
    scenario = f"const CASE = {json.dumps(localization)};\n" + CHANGED_VALUE
    out = run_js(tmp_path, "localize_emoji_buttons.js", scenario)
    ck, vae = "Use same checkpoint", "Use same VAE"
    if not translated:
        # English: nothing is registered and nothing is rewritten.
        assert out["listens"] == 0
        assert out["changed"] == [ck, vae, vae, ck]
        return
    tck, tvae = localization[ck], localization[vae]
    assert out["swept"] == [tck, tvae, tvae, tck]
    assert out["picked"] == "model.safetensors"  # not a key: left alone
    assert out["otherEvents"] == [ck, vae, vae, ck]
    # The ADetailer dropdowns only, and not the one being typed into.
    assert out["changed"] == [tck, vae, tvae, ck]
    assert out["blurred"] == [tck, tvae, tvae, ck]


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


def _export_blob_js() -> str:
    tree = ast.parse((ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8"))
    return next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "_EXPORT_BLOB_JS"
    )


EXPORT_BLOB_STEP = r"""
const clicked = [];
const blobs = {};
const revoked = [];
globalThis.document = {
    createElement: () => ({ click() { clicked.push([this.href, this.download]); },
                            remove() {} }),
    body: { appendChild() {} },
};
URL.createObjectURL = (blob) => { const u = "blob:" + Object.keys(blobs).length;
                                  blobs[u] = blob; return u; };
URL.revokeObjectURL = (u) => revoked.push(u);
globalThis.setTimeout = (f) => f();
// Gradio 3.41's wrapper for a front-end-only step with one output: it gets
// the input's value, then the output's.
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const step = new AsyncFunction("__fn_args", `let result = await (${EXPORT_JS})(...__fn_args);
                    return (true && !Array.isArray(result)) ? [result] : result;`);
(async () => {
    const out = {};
    out.saved = await step([STATUS, STATUS]);
    out.plain = await step(["Done.", "Done."]);
    out.broken = await step(["Done.\n\n<!-- adetailer-export:A -->", ""]);
    out.clicked = clicked;
    out.revoked = revoked;
    out.files = {};
    for (const [u, b] of Object.entries(blobs)) {
        out.files[u] = [b.type, Buffer.from(await b.arrayBuffer()).toString("utf8")];
    }
    console.log(JSON.stringify(out));
})();
"""


@needs_node
def test_gradio_3_export_step_saves_the_library_and_clears_the_status(tmp_path):
    # On AUTOMATIC1111 (Gradio 3) Export downloaded nothing. The click sends
    # the library base64 in a hidden comment of its status; the front-end step
    # saves it as a file in the browser and leaves only the message.
    import base64

    payload = '{"saved": {"prompt": "caf\u00e9 \u2014 face -->"}}'
    status = (
        "\u2705 Exported **1** preset(s).\n\n<!-- adetailer-export:"
        + base64.b64encode(payload.encode("utf-8")).decode("ascii")
        + " -->"
    )
    driver = tmp_path / "export-blob.js"
    driver.write_text(
        f"const EXPORT_JS = {json.dumps(_export_blob_js())};\n"
        f"const STATUS = {json.dumps(status)};\n" + EXPORT_BLOB_STEP,
        encoding="utf-8",
    )
    result = subprocess.run(
        [NODE, str(driver)], capture_output=True, encoding="utf-8", timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["clicked"] == [["blob:0", "adetailer-ultimate-presets.json"]]
    assert out["files"] == {"blob:0": ["application/json", payload]}
    assert out["revoked"] == ["blob:0"]
    assert out["saved"] == ["\u2705 Exported **1** preset(s)."]
    # A status without the comment is left as it is.
    assert out["plain"] == ["Done."]
    # A payload that cannot be decoded says so instead of claiming success.
    assert "user_presets.json" in out["broken"][0]
    assert "Exported" not in out["broken"][0]


MEMORY_KEY = "adetailer-ultimate:open-sections:v1"


def _section_id(mode: str, stem: str, suf: str = "") -> str:
    return f"script_{mode}_adetailer_ad_{stem}_accordion" + (f"_{suf}" if suf else "")


OPEN_SECTIONS = r"""
const KEY = "adetailer-ultimate:open-sections:v1";
const sid = (mode, stem, suf) =>
    "script_" + mode + "_adetailer_ad_" + stem + "_accordion" + (suf ? "_" + suf : "");
// Gradio's accordion: the header (with `open`) is a direct child of the block
// that carries the elem_id; the content follows it.
const sections = {};
// Gradio 4 also keeps the state in the component's props, but records a click
// only once its mount handler listens for prop changes; the next update spreads
// the props again.
let mounted = false;
function section(name, id, open, kids) {
    const header = new El("button", { class: "label-wrap" + (open ? " open" : "") },
        [new El("span", {}, [name])]);
    header.props = { open: !!open };
    if (CASE.gradio4) {
        const click = header.click;
        header.click = function () {
            click.call(this);
            if (mounted) this.props.open = this.classList.contains("open");
        };
    }
    const block = new El("div", { id, class: "block" }, [header, new El("div", {}, kids || [])]);
    sections[name] = { header, block };
    return block;
}
const inner = new El("button", {}, ["Preview"]);
const main = section("main", sid("txt2img", "main"), false, [
    section("detection", sid("txt2img", "detection"), false, [
        section("preview", sid("txt2img", "preview"), false, [inner])]),
    section("inpainting2", sid("txt2img", "inpainting", "2nd"), false),
    section("preview2", sid("txt2img", "preview", "2nd"), false),
    section("other", "controlnet_accordion", false),
]);
const body = new El("body", {}, [main, section("mask_i2i", sid("img2img", "mask_preprocessing"), true)]);

const data = {};
if (CASE.stored !== null) data[KEY] = CASE.stored;
const writes = [];
const storage = {
    getItem: (k) => { if (CASE.getThrows) throw new Error("blocked"); return k in data ? data[k] : null; },
    setItem: (k, v) => { writes.push(v); if (CASE.setThrows) throw new Error("QuotaExceededError"); data[k] = String(v); },
};
let uiLoaded = null;
run(SCRIPT, body, (w) => {
    if (CASE.getterThrows) Object.defineProperty(w, "localStorage", { get() { throw new Error("SecurityError"); } });
    else if (!CASE.noStorage) w.localStorage = storage;
    if (CASE.onUiLoaded) w.onUiLoaded = (f) => { uiLoaded = f; };
});
const clicks = () => Object.fromEntries(Object.entries(sections).map(([k, s]) => [k, s.header.clicks || 0]));
const opened = () => Object.fromEntries(Object.entries(sections).map(([k, s]) => [k, s.header.classList.contains("open")]));
const stored = () => (KEY in data ? JSON.parse(data[KEY]) : null);
const out = {};
if (CASE.clickEarly) { sections.detection.header.click(); fireTimers(); }
if (CASE.onUiLoaded) fireTimers();  // a script that does not wait restores here
out.beforeUi = clicks();
if (uiLoaded) uiLoaded();
out.sync = clicks();  // clicked inside the onUiLoaded callback itself
mounted = true;
fireTimers();  // the first pass, one tick later
out.boot = { clicks: clicks(), open: opened(), writes: writes.length };
if (CASE.gradio4) {
    Object.values(sections).forEach((s) => {
        if (s.header.props.open) s.header.classList.add("open");
        else s.header.classList.remove("open");
    });
    out.respread = opened();
}
// A section that mounts after the page has loaded.
main.append(section("inpainting3", sid("txt2img", "inpainting", "3rd"), false));
fireTimers();  // the second pass, 1.5 s later
fireTimers();
out.rescan = { clicks: clicks(), open: opened(), writes: writes.length };
sections.inpainting2.header.children[0].click(); fireTimers();  // on the title
out.firstClick = stored();
sections.inpainting2.header.click(); fireTimers();
out.secondClick = stored();
const before = writes.length;
sections.main.header.click(); fireTimers();
inner.click(); fireTimers();
sections.other.header.click(); fireTimers();
out.ignoredWrites = writes.length - before;
// Another browser tab writes in between.
if (KEY in data && !CASE.setThrows) {
    const s = stored(); s[sid("img2img", "inpainting")] = true; data[KEY] = JSON.stringify(s);
}
sections.preview2.header.click(); fireTimers();
out.final = stored();
out.finalOpen = opened();
console.log(JSON.stringify(out));
"""


def _open_sections(tmp_path: Path, **case):
    config = {
        "stored": None, "getThrows": False, "setThrows": False, "noStorage": False,
        "getterThrows": False, "onUiLoaded": False, "clickEarly": False,
        "gradio4": False,
    }
    config.update(case)
    scenario = f"const CASE = {json.dumps(config)};\n" + OPEN_SECTIONS
    return run_js(tmp_path, "accordion-memory.js", scenario)


@needs_node
def test_open_sections_are_restored_once_and_never_the_main_panel(tmp_path):
    # Issue #6: the sections a user left open came back closed after a reload.
    stored = {
        _section_id("txt2img", "detection"): True,
        _section_id("txt2img", "preview", "2nd"): False,  # already closed
        _section_id("img2img", "mask_preprocessing"): False,  # open in the page
        _section_id("txt2img", "inpainting", "3rd"): True,  # mounts late
        _section_id("txt2img", "main"): True,  # would switch ADetailer on
        _section_id("txt2img", "bogus"): True,
        _section_id("txt2img", "inpainting", "2nd"): "yes",
    }
    out = _open_sections(tmp_path, stored=json.dumps(stored))
    boot = out["boot"]
    assert boot["clicks"] == {
        "main": 0, "detection": 1, "preview": 0, "inpainting2": 0, "preview2": 0,
        "other": 0, "mask_i2i": 1,
    }
    assert boot["open"]["detection"] is True
    assert boot["open"]["mask_i2i"] is False
    assert boot["writes"] == 0  # restoring stores nothing
    # The second pass opens the late section and toggles nothing twice.
    assert out["rescan"]["clicks"]["inpainting3"] == 1
    assert out["rescan"]["open"]["inpainting3"] is True
    assert out["rescan"]["clicks"]["detection"] == 1
    assert out["rescan"]["clicks"]["mask_i2i"] == 1
    assert out["rescan"]["writes"] == 0  # not even once its timers ran
    # A user click is stored; the invalid entries drop out on that write.
    assert out["firstClick"] == {
        _section_id("txt2img", "detection"): True,
        _section_id("txt2img", "preview", "2nd"): False,
        _section_id("img2img", "mask_preprocessing"): False,
        _section_id("txt2img", "inpainting", "3rd"): True,
        _section_id("txt2img", "inpainting", "2nd"): True,
    }
    assert out["secondClick"][_section_id("txt2img", "inpainting", "2nd")] is False
    # The main panel, a button inside a section and another extension's
    # accordion are never stored.
    assert out["ignoredWrites"] == 0
    # A change made by another browser tab in between is kept.
    assert out["final"][_section_id("img2img", "inpainting")] is True
    assert out["final"][_section_id("txt2img", "preview", "2nd")] is True


@needs_node
def test_open_sections_wait_for_the_webui_and_keep_an_earlier_user_click(tmp_path):
    stored = {_section_id("txt2img", "detection"): True}
    out = _open_sections(tmp_path, stored=json.dumps(stored), onUiLoaded=True)
    assert out["beforeUi"]["detection"] == 0
    assert out["boot"]["clicks"]["detection"] == 1
    out = _open_sections(
        tmp_path, stored=json.dumps(stored), onUiLoaded=True, clickEarly=True
    )
    # The user opened it before the page finished loading: not toggled again.
    assert out["boot"]["clicks"]["detection"] == 1
    assert out["boot"]["open"]["detection"] is True


@needs_node
def test_open_sections_stay_open_on_gradio_4(tmp_path):
    # On Forge / Forge Neo a section opened inside the onUiLoaded callback was
    # not recorded by Gradio 4 and closed again on the next page update.
    stored = {
        _section_id("txt2img", "detection"): True,
        _section_id("img2img", "mask_preprocessing"): False,
    }
    out = _open_sections(tmp_path, stored=json.dumps(stored), onUiLoaded=True, gradio4=True)
    assert set(out["sync"].values()) == {0}
    assert out["boot"]["clicks"]["detection"] == 1
    assert out["respread"]["detection"] is True
    assert out["respread"]["mask_i2i"] is False
    assert out["respread"]["main"] is False


@needs_node
@pytest.mark.parametrize("raw", ["{bad", "[1, 2]", "null", "42", '"open"'])
def test_open_sections_ignore_unexpected_stored_values(tmp_path, raw):
    out = _open_sections(tmp_path, stored=raw)
    assert set(out["boot"]["clicks"].values()) == {0}
    assert out["firstClick"] == {_section_id("txt2img", "inpainting", "2nd"): True}


@needs_node
@pytest.mark.parametrize("blocked", ["getThrows", "setThrows", "noStorage", "getterThrows"])
def test_open_sections_work_without_usable_storage(tmp_path, blocked):
    stored = json.dumps({_section_id("txt2img", "detection"): True})
    out = _open_sections(tmp_path, stored=stored, **{blocked: True})
    # Sections still open and close; nothing breaks.
    assert out["finalOpen"]["preview2"] is True
    if blocked == "setThrows":
        assert out["boot"]["clicks"]["detection"] == 1  # reading still works
        assert out["final"] == json.loads(stored)
    else:
        assert set(out["boot"]["clicks"].values()) == {0}


def test_open_sections_cover_every_tab_section_but_the_main_panel():
    # The script's list must follow the accordions built in aaaaaa/ui.py.
    ui = (ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8")
    stems = set(re.findall(r'eid\("ad_(\w+)_accordion"\)', ui))
    assert "main" in stems
    js = (ROOT / "javascript" / "accordion-memory.js").read_text(encoding="utf-8")
    pattern = re.search(r"const ID_RE = /(.+)/;", js).group(1)
    listed = re.search(r"_ad_\(\?:([\w|]+)\)_accordion", pattern).group(1).split("|")
    assert set(listed) == stems - {"main"}
    source = ast.parse(ui)
    module = ast.Module(
        body=[
            node for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"ordinal", "suffix", "elem_id"}
        ],
        type_ignores=[],
    )
    namespace: dict = {}
    exec(compile(module, "ui.py", "exec"), namespace)
    id_re = re.compile(pattern)
    for n in range(15):
        for img2img in (False, True):
            for stem in stems:
                matched = id_re.match(namespace["elem_id"](f"ad_{stem}_accordion", n, img2img))
                assert bool(matched) == (stem != "main"), (stem, n, img2img)


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


def test_ci_runs_on_pushes_to_the_beta_branch():
    # The betas are built from the beta branch, and the macOS Python 3.10-3.14
    # matrix runs only on GitHub: a push there started no run.
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    block = re.search(r"^  push:\n(.*?)(?=^  \S)", text, re.M | re.S).group(1)
    branches = re.search(r"^\s+branches:\s*\[([^\]]*)\]", block, re.M).group(1)
    assert {"main", "beta"} <= {b.strip().strip("\"'") for b in branches.split(",")}


PICKER = r"""
const P = "script_img2img_adetailer_";
const img = new El("img", {});
img.naturalWidth = CASE.natural[0];
img.naturalHeight = CASE.natural[1];
img.objectFit = "contain";
img.rect = { left: 10, top: 20, width: 400, height: 400 };
const output = new El("div", { id: P + "ad_preview_output" + CASE.sfx }, [img]);
output.rect = { left: 10, top: 20, width: 400, height: 400 };
const boxes = new El("textarea", { value: CASE.boxes });
const field = new El("textarea", { value: CASE.field });
const body = new El("body", {}, [
    output,
    new El("div", { id: P + "ad_preview_boxes" + CASE.sfx }, [boxes]),
    new El("div", { id: P + "ad_inpaint_indices" + CASE.sfx }, [field]),
]);
run(SCRIPT, body, (w) => {
    w.getComputedStyle = (e) => ({
        display: e.style.display,
        objectFit: e.objectFit || "fill",
        position: e.style.position || "static",
    });
});
const tick = () => intervals.forEach((f) => f());
const state = () => {
    const layer = output.querySelector(".ad-pick-layer");
    return layer ? layer.children.map((b) => [
        b.title, b.classList.contains("ad-picked"),
        b.style.left, b.style.top, b.style.width, b.style.height,
    ]) : null;
};
tick();
const out = { first: state(), writes: [] };
for (const k of CASE.clicks) {
    const box = output.querySelector(".ad-pick-layer").children[k - 1];
    box.handlers.click.forEach((f) => f({ preventDefault() {}, stopPropagation() {} }));
    out.writes.push([field.value, field.events.length]);
}
tick();
out.after = state();
console.log(JSON.stringify(out));
"""

_TWO_BOXES = json.dumps({"w": 200, "h": 100, "boxes": [[0, 0, 100, 50], [100, 50, 200, 100]]})
# A 200x100 picture drawn in a 400x400 <img> with object-fit: contain is
# 400x200, 100 px down: the boxes are scaled by 2 and moved down by 100.
_BOX_1 = ["#1", "0px", "100px", "200px", "100px"]
_BOX_2 = ["#2", "200px", "200px", "200px", "100px"]


def _picker(tmp_path, *, natural=(200, 100), boxes=_TWO_BOXES, field="", clicks=(), sfx=""):
    case = json.dumps({
        "natural": list(natural), "boxes": boxes, "field": field,
        "clicks": list(clicks), "sfx": sfx,
    })
    return run_js(tmp_path, "picker.js", f"const CASE = {case};\n" + PICKER)


def _boxes(state, picked):
    return [[b[0], b[0] in picked, *b[1:]] for b in (_BOX_1, _BOX_2)] == state


@needs_node
@pytest.mark.parametrize("sfx", ["", "_2nd"])
def test_picker_lays_a_box_over_each_detection(tmp_path, sfx):
    out = _picker(tmp_path, sfx=sfx)
    assert _boxes(out["first"], set())


@needs_node
def test_picker_click_adds_and_removes_a_number(tmp_path):
    out = _picker(tmp_path, clicks=[2, 2, 1])
    # Each write is announced to Gradio with an input event.
    assert out["writes"] == [["2", 1], ["", 2], ["1", 3]]
    assert _boxes(out["after"], {"#1"})


@needs_node
@pytest.mark.parametrize(
    ("field", "first", "click", "written"),
    [
        # The field is read as the detailer pass reads it.
        ("1-2", {"#1", "#2"}, 1, "2"),
        ("#2", {"#2"}, 1, "1,2"),
        ("2 1", {"#1", "#2"}, 2, "1"),
        ("\uff11\u3001\uff12", {"#1", "#2"}, 2, "1"),
        ("1 - 2", {"#1", "#2"}, 2, "1"),
        # A number with no box on this preview is kept; text that is not a
        # number is skipped, as the detailer pass skips it.
        ("9, face", set(), 1, "1,9"),
    ],
)
def test_picker_follows_the_numbers_in_the_field(tmp_path, field, first, click, written):
    out = _picker(tmp_path, field=field, clicks=[click])
    assert _boxes(out["first"], first)
    assert out["writes"][0][0] == written


@needs_node
def test_picker_draws_a_box_inside_another_on_top(tmp_path):
    # The face box came after the eyes, mouth and nose boxes it contains and
    # was drawn over them, so every click on those went to the face.
    boxes = json.dumps({"w": 200, "h": 100, "boxes": [[50, 25, 100, 50], [0, 0, 200, 100]]})
    out = _picker(tmp_path, boxes=boxes)
    assert [b[0] for b in out["first"]] == ["#2", "#1"]
    # The click handler of each box still toggles its own number.
    out = _picker(tmp_path, boxes=boxes, clicks=[2])
    assert out["writes"] == [["1", 1]]


@needs_node
@pytest.mark.parametrize(
    ("natural", "boxes"),
    [
        ((200, 200), _TWO_BOXES),  # the picture shown is not the one boxed
        ((200, 100), ""),  # Combine all tabs, no detections or a failure
        ((200, 100), "not json"),
    ],
)
def test_picker_shows_nothing_when_unsure(tmp_path, natural, boxes):
    assert _picker(tmp_path, natural=natural, boxes=boxes)["first"] is None


def test_the_preview_boxes_field_takes_no_room():
    # The hidden field of picker.js sits in the row of the preview button. The
    # "form" container Gradio wraps it in took half that row, squeezing the
    # status line: it is hidden too, but only when it holds nothing else.
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    for selector in (
        ".ad-preview-boxes",
        "div.form:has(> .ad-preview-boxes):not(:has(> :not(.ad-preview-boxes)))",
    ):
        assert re.search(r"display:\s*none\s*!important", rules[selector]), selector


SHOW_LABELS = r"""
const P = "script_txt2img_adetailer_";
const img = new El("img", {});
img.naturalWidth = CASE.natural[0];
img.naturalHeight = CASE.natural[1];
img.objectFit = "contain";
img.rect = { left: 10, top: 20, width: 400, height: 400 };
const output = new El("div", { id: P + "ad_preview_output" + CASE.sfx }, [img]);
output.rect = { left: 10, top: 20, width: 400, height: 400 };
const btn = new El("button", { id: P + "ad_preview_btn" + CASE.sfx });
const row = new El("div", { class: "row" }, [
    btn, new El("div", { id: P + "ad_preview_all_tabs" + CASE.sfx })]);
const body = new El("body", {}, [
    output, row,
    new El("div", { id: P + "ad_preview_boxes" + CASE.sfx }, [
        new El("textarea", { value: CASE.boxes })]),
    new El("div", { id: P + "ad_inpaint_indices" + CASE.sfx }, [new El("textarea", {})]),
]);
const store = CASE.stored === null ? {} : { "adetailer-show-labels": CASE.stored };
run(SCRIPT, body, (w) => {
    w.getComputedStyle = (e) => ({
        display: e.style.display,
        objectFit: e.objectFit || "fill",
        position: e.style.position || "static",
    });
    if (CASE.blocked) {
        Object.defineProperty(w, "localStorage", { get() { throw new Error("blocked"); } });
    } else {
        w.localStorage = {
            getItem: (k) => (k in store ? store[k] : null),
            setItem: (k, v) => { store[k] = String(v); },
        };
    }
});
const tick = () => intervals.forEach((f) => f());
const view = () => {
    const sw = row.querySelectorAll("label.ad-show-labels");
    const plain = output.querySelectorAll("img.ad-plain-picture");
    const layer = output.querySelector(".ad-pick-layer");
    return {
        row: row.children.map((c) => c.tagName),
        switches: sw.length,
        checked: sw.length ? sw[0].querySelector("input").checked : null,
        text: sw.length ? sw[0].textContent : null,
        plain: plain.length ? [plain.length, plain[0].src, plain[0].style.left,
            plain[0].style.top, plain[0].style.width, plain[0].style.height] : null,
        boxes: layer ? layer.children.map((b) => [b.style.left, b.style.top]) : null,
        stored: store["adetailer-show-labels"] === undefined ? null : store["adetailer-show-labels"],
    };
};
tick();
tick();
const out = { first: view(), steps: [] };
for (let i = 0; i < CASE.toggles; i += 1) {
    const box = row.querySelector("label.ad-show-labels").querySelector("input");
    box.checked = !box.checked;
    box.handlers.change.forEach((f) => f({ target: box }));
    tick();
    out.steps.push(view());
}
console.log(JSON.stringify(out));
"""

_PLAIN = "data:image/jpeg;base64,AAAA"
_WITH_PLAIN = json.dumps({
    "w": 200, "h": 100, "boxes": [[0, 0, 100, 50], [100, 50, 200, 100]], "plain": _PLAIN,
})
# Over the picture, drawn 400x200 and 100 px down in its 400x400 <img>.
_PLAIN_AT = [1, _PLAIN, "0px", "100px", "400px", "200px"]


def _show_labels(tmp_path, *, boxes=_WITH_PLAIN, stored=None, toggles=0,
                 natural=(200, 100), blocked=False, sfx=""):
    case = json.dumps({
        "natural": list(natural), "boxes": boxes, "stored": stored,
        "toggles": toggles, "blocked": blocked, "sfx": sfx,
    })
    return run_js(tmp_path, "picker.js", f"const CASE = {case};\n" + SHOW_LABELS)


@needs_node
@pytest.mark.parametrize("sfx", ["", "_2nd"])
def test_show_labels_switch_sits_after_the_preview_button(tmp_path, sfx):
    out = _show_labels(tmp_path, sfx=sfx)["first"]
    # One switch per button, right after it, ticked by default: the preview
    # looks as before.
    assert out["row"] == ["BUTTON", "LABEL", "DIV"]
    assert out["switches"] == 1
    assert out["checked"] is True
    assert out["text"] == "Show labels"
    assert out["plain"] is None


@needs_node
def test_show_labels_off_lays_the_plain_picture_under_the_boxes(tmp_path):
    out = _show_labels(tmp_path, toggles=2)
    off, on = out["steps"]
    assert off["checked"] is False and off["stored"] == "0"
    assert off["plain"] == _PLAIN_AT
    # The boxes stay clickable, in the same places.
    assert off["boxes"] == out["first"]["boxes"] == [["0px", "100px"], ["200px", "200px"]]
    assert on["checked"] is True and on["stored"] == "1"
    assert on["plain"] is None


@needs_node
def test_show_labels_off_is_remembered(tmp_path):
    out = _show_labels(tmp_path, stored="0")["first"]
    assert out["checked"] is False
    assert out["plain"] == _PLAIN_AT


@needs_node
def test_show_labels_works_on_combined_previews(tmp_path):
    # Combine all tabs sends the plain picture but no boxes.
    boxes = json.dumps({"w": 200, "h": 100, "plain": _PLAIN})
    out = _show_labels(tmp_path, boxes=boxes, stored="0")["first"]
    assert out["plain"] == _PLAIN_AT
    assert out["boxes"] is None


@needs_node
@pytest.mark.parametrize(
    ("natural", "boxes"),
    [
        ((200, 200), _WITH_PLAIN),  # the picture shown is not the one copied
        ((200, 100), _TWO_BOXES),  # no plain picture: the switch does nothing
        ((200, 100), ""),
        ((200, 100), json.dumps({"w": 200, "h": 100, "plain": "javascript:x"})),
    ],
)
def test_show_labels_off_does_nothing_when_unsure(tmp_path, natural, boxes):
    out = _show_labels(tmp_path, natural=natural, boxes=boxes, stored="0")["first"]
    assert out["plain"] is None


@needs_node
def test_show_labels_works_without_storage(tmp_path):
    out = _show_labels(tmp_path, blocked=True, toggles=1)
    assert out["first"]["checked"] is True
    assert out["steps"][0]["plain"] == _PLAIN_AT


def test_show_labels_styles_exist():
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    plain = rules["img.ad-plain-picture"]
    assert re.search(r"pointer-events:\s*none", plain)
    # Below the clickable boxes of picker.js.
    z = int(re.search(r"z-index:\s*(\d+)", plain).group(1))
    assert z < int(re.search(r"z-index:\s*(\d+)", rules[".ad-pick-layer"]).group(1))
    assert "label.ad-show-labels" in rules


def test_show_labels_checkbox_shows_its_tick():
    # AUTOMATIC1111's theme sets appearance: none on every checkbox and draws
    # the tick only on Gradio's own: the plain checkbox picker.js adds looked
    # the same ticked and unticked.
    rules = {s: body for selectors, body in css_rules() for s in selectors}
    box = rules['label.ad-show-labels > input[type="checkbox"]']
    assert re.search(r"(?<!-)appearance:\s*auto\s*!important", box)
    assert re.search(r"-webkit-appearance:\s*checkbox\s*!important", box)


def _option_tooltips() -> dict[str, str]:
    source = (ROOT / "javascript" / "option-tooltips.js").read_text(encoding="utf-8")
    block = source.split("const TOOLTIPS = {", 1)[1].split("\n    };", 1)[0]
    return dict(re.findall(r'(\w+):\s*"([^"]*)"', block))


OPTION_TOOLTIPS = r"""
const P = "script_txt2img_adetailer_";
const els = {
    width: new El("div", { id: P + "ad_inpaint_width" }),
    width2: new El("div", { id: P + "ad_inpaint_width_2nd" }),
    useWh: new El("div", { id: P + "ad_use_inpaint_width_height" }),
    img2img11: new El("div", { id: "script_img2img_adetailer_ad_confidence_11th" }),
    classes: new El("div", { id: P + "ad_model_classes" }),
    dropdown: new El("div", { id: P + "ad_model_classes_dropdown" }),
    exclude: new El("div", { id: P + "ad_model_classes_exclude" }),
    resetAll: new El("div", { id: P + "ad_preset_reset_all_3rd" }),
    resetBtn: new El("button", { id: P + "ad_preset_reset" }),
    other: new El("div", { id: "txt2img_denoising_strength" }),
    own: new El("div", { id: P + "ad_mask_blur" }),
};
els.own.title = "kept";
const body = new El("body", {}, Object.values(els));
run(SCRIPT, body, { localization: CASE });
// Gradio re-renders a block: the new node has no title until the observer runs.
const late = body.append(new El("div", { id: P + "ad_mask_k_2nd" }));
const before = late.title;
observers[observers.length - 1].cb();
const out = { before, late: late.title };
for (const k in els) out[k] = els[k].title;
console.log(JSON.stringify(out));
"""


@needs_node
@pytest.mark.parametrize("translated", [True, False])
def test_each_option_gets_its_own_tooltip_in_every_tab(tmp_path, translated):
    # Most options of a tab had only a label: their meaning was in the README.
    tips = _option_tooltips()
    localization = {tips["ad_confidence"]: "CONFIDENCE (translated)"} if translated else {}
    scenario = f"const CASE = {json.dumps(localization)};\n" + OPTION_TOOLTIPS
    titles = run_js(tmp_path, "option-tooltips.js", scenario)
    assert titles["width"] == tips["ad_inpaint_width"]
    assert titles["width2"] == titles["width"]
    assert titles["useWh"] == tips["ad_use_inpaint_width_height"]
    assert titles["img2img11"] == (
        "CONFIDENCE (translated)" if translated else tips["ad_confidence"]
    )
    assert titles["classes"] == tips["ad_model_classes"]
    assert titles["dropdown"] == ""  # it has an info line of its own
    assert titles["exclude"] == tips["ad_model_classes_exclude"]
    assert titles["resetAll"] == tips["ad_preset_reset_all"]
    assert titles["resetBtn"] == ""  # a button: button-tooltips.js names it
    assert titles["other"] == ""
    assert titles["own"] == "kept"
    assert titles["before"] == ""
    assert titles["late"] == tips["ad_mask_k"]


DROPDOWN_TOOLTIPS = r"""
const P = "script_txt2img_adetailer_";
const input = new El("input", { value: "DPM++ 2M" });
const euler = new El("input", { value: "Euler a" });
const hinted = new El("input", {});
hinted.title = "Do not do anything special";
const sampler = new El("div", { id: P + "ad_sampler_2nd" }, [
    new El("div", { class: "wrap" }, [input, new El("ul", { class: "options" })])]);
const sampler3 = new El("div", { id: P + "ad_sampler_3rd" }, [euler]);
const preprocessor = new El("div", { id: P + "ad_controlnet_module" }, [hinted]);
const other = new El("input", {});
const body = new El("body", {}, [
    sampler, sampler3, preprocessor, new El("div", { id: "txt2img_sampling" }, [other])]);
// AUTOMATIC1111's hints.js keeps its own texts in the global `titles`.
run(SCRIPT, body, { localization: CASE, titles: { "Euler a": "Euler Ancestral (hint)" } });
const first = input.title;
// AUTOMATIC1111's hints.js: the dropdown's list changed, so it empties the
// input's title and fills it only for a few values it knows.
input.title = "";
observers[observers.length - 1].cb();
console.log(JSON.stringify({
    first, again: input.title, block: sampler.title, hinted: hinted.title, other: other.title,
    euler: euler.title,
}));
"""


@needs_node
def test_a_dropdown_input_keeps_the_tooltip_when_automatic1111_empties_it(tmp_path):
    # AUTOMATIC1111's hints.js sets title="" on a dropdown's input each time
    # its list opens, and an empty title hides the block's: on the dropdown
    # field the tooltip was gone.
    tips = _option_tooltips()
    scenario = "const CASE = {};\n" + DROPDOWN_TOOLTIPS
    titles = run_js(tmp_path, "option-tooltips.js", scenario)
    assert titles["block"] == tips["ad_sampler"]
    assert titles["first"] == tips["ad_sampler"]
    assert titles["again"] == tips["ad_sampler"]
    assert titles["hinted"] == "Do not do anything special"  # a title it set stays
    # A value hints.js explains is left for it to fill a second later.
    assert titles["euler"] == ""
    assert titles["other"] == ""


def _panel_widgets() -> list[tuple[str, str, bool]]:
    """(elem_id key, component, has info=) for every widget of the panel."""
    tree = ast.parse((ROOT / "aaaaaa" / "ui.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not ast.unparse(node.func).startswith("gr."):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        eid = kw.get("elem_id")
        if not (isinstance(eid, ast.Call) and ast.unparse(eid.func) == "eid"):
            continue
        key = eid.args[0].value
        hidden = isinstance(kw.get("visible"), ast.Constant) and kw["visible"].value is False
        if not hidden:
            out.append((key, ast.unparse(node.func).split(".")[-1], "info" in kw))
    return out


def test_every_panel_option_without_an_info_line_has_a_tooltip():
    tips = _option_tooltips()
    options = ("Checkbox", "Slider", "Dropdown", "Textbox", "Radio", "Number")
    # The hidden field picker.js fills from the preview is not an option.
    no_tooltip = {"ad_preview_boxes"}
    widgets = _panel_widgets()
    keys = {key for key, _, _ in widgets}
    checked = [key for key, kind, _ in widgets if kind in options]
    assert len(checked) > 60
    missing = [
        key
        for key, kind, has_info in widgets
        if kind in options and not has_info and key not in no_tooltip and key not in tips
    ]
    assert missing == []
    # Every tooltip names a real option, which shows no info line of its own.
    assert set(tips) <= keys
    assert [k for k, _, has_info in widgets if has_info and k in tips] == []


def test_option_tooltips_are_one_plain_text_block():
    # The WebUI's localizer and Language Diffusion translate a whole string:
    # markdown, line breaks or quotes inside it would show as they are.
    for key, text in _option_tooltips().items():
        assert text and text == text.strip(), key
        assert not re.search(r"\*|`|\[[^\]]*\]\(|\n|<", text), key
        assert text.isascii(), key
