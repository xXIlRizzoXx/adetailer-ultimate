/* ADetailer Ultimate — remember which ADetailer sections are open (issue #6).
 *
 * Neither the WebUI nor Gradio keeps the open/closed state of an accordion,
 * and Gradio sends no open/close event to the server. This script remembers,
 * per browser, which of the per-tab sections the user left open (Preset
 * library export / import, Inpaint prompts, Detection, Detection preview, Run
 * ADetailer on an image, Mask Preprocessing, Inpainting), in every detector
 * tab of txt2img and img2img, and opens them again when the page loads.
 *
 * The main "ADetailer" accordion is NEVER touched: the host's
 * inputAccordion.js links its open state to the "Enable ADetailer" switch, so
 * reopening it would switch ADetailer on. Its id is not in ID_RE on purpose.
 *
 * Pure client-side DOM (no Gradio components, no event listeners on the Python
 * side), so it cannot affect any Gradio wiring / fn_index — index-safe. Works
 * the same on A1111 (Gradio 3, `div.label-wrap`) and Forge / Forge Neo
 * (Gradio 4, `button.label-wrap`): both put the section header, with an
 * `open` class, as a direct child of the block that carries the elem_id.
 * Only sections the user actually toggled are stored, in localStorage; if
 * storage is blocked or holds something unexpected, the page behaves as if
 * this script were not there.
 */

(function () {
    "use strict";

    const KEY = "adetailer-ultimate:open-sections:v1";
    // The per-tab sections: aaaaaa/ui.py eid("ad_<stem>_accordion") plus the
    // "_2nd", "_3rd"... suffix of tabs 2 and later. ad_main_accordion is
    // deliberately left out (see above).
    const ID_RE = /^script_(?:txt2img|img2img)_adetailer_ad_(?:preset_io|prompts|detection|preview|apply|mask_preprocessing|inpainting)_accordion(?:_\d+(?:st|nd|rd|th))?$/;

    // Set while restore() clicks headers, so those clicks are not stored.
    let restoring = false;
    // Sections already restored or toggled by the user: never toggled twice.
    const applied = {};

    // The localStorage getter itself can throw (blocked site data).
    function storage() {
        try {
            return window.localStorage || null;
        } catch (e) {
            return null;
        }
    }

    // Only known section ids with a boolean value are kept; anything else
    // (corrupt JSON, an array, a value written by something else) is ignored.
    function readStore() {
        const out = {};
        try {
            const ls = storage();
            const raw = ls ? ls.getItem(KEY) : null;
            const obj = raw ? JSON.parse(raw) : null;
            if (obj && typeof obj === "object" && !Array.isArray(obj)) {
                Object.keys(obj).forEach(function (id) {
                    if (ID_RE.test(id) && typeof obj[id] === "boolean") out[id] = obj[id];
                });
            }
        } catch (e) {
            /* unreadable or blocked storage: behave as if nothing was stored */
        }
        return out;
    }

    // Re-read before writing, so another browser tab's changes are kept.
    function writeOne(id, open) {
        try {
            const ls = storage();
            if (!ls) return;
            const store = readStore();
            store[id] = open;
            ls.setItem(KEY, JSON.stringify(store));
        } catch (e) {
            /* quota exceeded or blocked storage: nothing is remembered */
        }
    }

    // The section's own header: a direct child, not the header of a section
    // nested inside it.
    function headerOf(block) {
        const kids = (block && block.children) || [];
        for (let i = 0; i < kids.length; i++) {
            const k = kids[i];
            if (k.classList && k.classList.contains("label-wrap")) return k;
        }
        return null;
    }

    function isOpen(header) {
        return header.classList.contains("open");
    }

    // Capture phase: runs before Gradio toggles the section, so the new state
    // is read on the next tick.
    function onClick(e) {
        if (restoring) return;
        const t = e.target;
        const header = t && t.closest ? t.closest(".label-wrap") : null;
        if (!header) return;
        const block = header.parentElement;
        if (!block || !ID_RE.test(block.id) || headerOf(block) !== header) return;
        applied[block.id] = true;
        const was = isOpen(header);
        setTimeout(function () {
            const now = isOpen(header);
            if (now !== was) writeOne(block.id, now);
        }, 0);
    }

    // Click a header only when its current state differs from the stored one.
    function restore() {
        const store = readStore();
        restoring = true;
        try {
            Object.keys(store).forEach(function (id) {
                if (applied[id]) return;
                const header = headerOf(document.getElementById(id));
                if (!header) return;
                applied[id] = true;
                if (isOpen(header) !== store[id]) header.click();
            });
        } catch (e) {
            /* never break the page */
        } finally {
            restoring = false;
        }
    }

    // The first pass waits one tick: on Gradio 4 the host runs onUiLoaded
    // before Gradio starts recording an accordion's own open/close, so a click
    // made there is lost and the next page update closes the section again.
    function start() {
        setTimeout(function () {
            restore();
            setTimeout(restore, 1500); // one more pass for sections mounted late
        }, 0);
    }

    document.addEventListener("click", onClick, true);
    if (typeof window.onUiLoaded === "function") {
        window.onUiLoaded(start);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
