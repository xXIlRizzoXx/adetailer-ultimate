/* ADetailer Ultimate — instant CLASSES sync (race fix).
 *
 * THE BUG THIS FIXES (reported 2026-07-17)
 * ----------------------------------------
 * The visible "detector CLASSES" multi-select is UI-only. The value the engine
 * actually reads lives in the HIDDEN ad_model_classes / ad_model_classes_excluded
 * textboxes, which Python fills via a `.change` handler — and that handler goes
 * through Gradio's QUEUE (ui.py `_sync_dropdown`, deliberately not queue=False,
 * so two fast selections can't land out of order). Consequence: while the queue
 * is busy — e.g. a 10-image batch — the sync can sit unprocessed for MINUTES. If
 * a generation starts before it lands, the browser submits the STALE hidden
 * value. And an empty class filter means "inpaint EVERY class the model finds",
 * so the user silently gets classes they explicitly did not select.
 *
 * THE FIX
 * -------
 * Mirror the multi-select into its hidden textboxes CLIENT-SIDE, the instant it
 * changes — no server round-trip, so there is nothing to wait for and nothing to
 * lose a race against. Writing the field and dispatching `input` updates Gradio's
 * own client state synchronously, so a Generate click that happens one
 * millisecond later already carries the correct value.
 *
 * The Python `.change` sync is intentionally LEFT IN PLACE as a backstop: it
 * computes the exact same CSV, so it can only ever confirm what we wrote. If
 * this script is disabled, or the DOM shape is one we don't recognise, behaviour
 * degrades to exactly what it is today — never worse.
 *
 * SAFETY
 * ------
 * A wrong write would be worse than the bug, so every path bails out rather than
 * guesses: unknown dropdown DOM (no `.wrap-inner`), an unreadable token, or a
 * missing "Exclude selected" checkbox (which decides WHICH hidden field receives
 * the CSV) all mean "don't write" — the server sync then handles it as before.
 *
 * Pure client-side DOM: no Gradio components and no Python-side event listeners
 * are added, so fn_index / dependency ordering is untouched — index-safe. Works
 * on A1111 (Gradio 3) and Forge / Forge Neo / reForge (Gradio 4); WebUIs
 * auto-load javascript/*.js.
 */

(function () {
    "use strict";

    const DD_KEY = "_ad_model_classes_dropdown";
    const DD_SEL = 'div[id*="' + DD_KEY + '"]';

    // The <textarea>/<input> inside a Gradio Textbox wrapper, addressed by the
    // wrapper's elem_id. Hidden textboxes (visible=False) still render, so this
    // resolves for ad_model_classes / ad_model_classes_excluded too.
    function fieldOf(wrapperId) {
        const box = document.getElementById(wrapperId);
        return box ? box.querySelector("textarea, input") : null;
    }

    // Read the selected class names out of a multi-select. Returns null — meaning
    // "I don't understand this DOM, do not write anything" — rather than risking a
    // wrong value. `.wrap-inner` is the multi-select marker on both Gradio 3 and 4;
    // its absence is our feature-detection gate.
    function readTokens(dropdown) {
        const inner = dropdown.querySelector(".wrap-inner");
        if (!inner) return null;
        const out = [];
        const tokens = inner.querySelectorAll(".token");
        for (let i = 0; i < tokens.length; i += 1) {
            // Strip the "×" remove control before reading, so its glyph can never
            // leak into a class name.
            const clone = tokens[i].cloneNode(true);
            const remove = clone.querySelectorAll(".token-remove");
            for (let j = 0; j < remove.length; j += 1) remove[j].remove();
            const label = (clone.textContent || "").trim();
            if (!label) return null; // a token we can't read → bail entirely
            out.push(label);
        }
        return out;
    }

    // Write through Gradio's own input path so its client state updates in the
    // same synchronous tick. Skipping no-op writes keeps the MutationObserver from
    // re-firing on our own changes.
    function writeField(field, value) {
        if (!field || field.value === value) return;
        field.value = value;
        field.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // Positive YOLO-World detection from the sibling detector dropdown's value.
    // World models carry "-world" in their filename; the dropdown renders that
    // name into an <input> we can read. This is STARTUP-SAFE, unlike the
    // visibility guard below which only recognises world mode after
    // on_ad_model_update has run (a detector change) — that callback does NOT
    // fire on initial render, so a RESTORED world detector would otherwise reach
    // syncOne with its (visible-in-world) free-text ad_model_classes box still
    // hidden and holding the user's saved vocabulary, and an empty dropdown would
    // wipe it. Reading the model name closes that gap. If the input can't be
    // read we return false and fall through to the visibility guard (unchanged).
    function modelIsWorld(dropdownId) {
        const modelBox = document.getElementById(
            dropdownId.replace(DD_KEY, "_ad_model")
        );
        if (!modelBox) return false;
        const inp = modelBox.querySelector("input");
        return !!inp && (inp.value || "").indexOf("-world") >= 0;
    }

    function syncOne(dropdown) {
        const id = dropdown.id;
        if (!id || id.indexOf(DD_KEY) < 0) return;

        // YOLO-World: ad_model_classes is a VISIBLE free-text box the user types
        // into, NOT a mirror of this dropdown — never write to it. Detected from
        // the model name so it holds at startup too (see modelIsWorld).
        if (modelIsWorld(id)) return;

        const incId = id.replace(DD_KEY, "_ad_model_classes");

        // WORLD-MODE GUARD — do NOT sync for YOLO-World (open-vocabulary) models.
        // There the include field ad_model_classes is a VISIBLE free-text box the
        // user types class names into directly, and the dropdown beside it is left
        // visible-but-empty — structurally identical to a fixed-class model with
        // nothing selected. Syncing would overwrite the user's typed classes with
        // the empty dropdown. The discriminator is exactly that visibility:
        // on_ad_model_update keeps ad_model_classes hidden (visible=False) for
        // every fixed-class / MediaPipe / None model and shows it ONLY in world
        // mode. We test the field's OWN computed display rather than offsetParent
        // so a collapsed accordion (an ancestor at display:none) can't be mistaken
        // for world mode — a descendant's own computed display is unaffected by an
        // ancestor being hidden. In world mode we bail and let the Python sync
        // handle it, exactly as it did before this script existed.
        const incBox = document.getElementById(incId);
        if (incBox && window.getComputedStyle(incBox).display !== "none") return;

        const tokens = readTokens(dropdown);
        if (tokens === null) return;

        // The "Exclude selected (NOT)" checkbox decides which hidden field gets the
        // CSV. Without it we cannot tell include from exclude — and guessing would
        // wipe a restored exclude filter — so bail and let the server sync do it.
        const flagBox = document.getElementById(
            id.replace(DD_KEY, "_ad_model_classes_exclude")
        );
        const flag = flagBox ? flagBox.querySelector('input[type="checkbox"]') : null;
        if (!flag) return;

        const csv = tokens.join(",");
        const exclude = flag.checked;
        writeField(fieldOf(incId), exclude ? "" : csv);
        writeField(
            fieldOf(id.replace(DD_KEY, "_ad_model_classes_excluded")),
            exclude ? csv : ""
        );
    }

    function syncAll() {
        const dropdowns = document.querySelectorAll(DD_SEL);
        for (let i = 0; i < dropdowns.length; i += 1) syncOne(dropdowns[i]);
    }

    function scheduleSync() {
        if (scheduleSync._q) return;
        scheduleSync._q = true;
        const run = function () {
            scheduleSync._q = false;
            syncAll();
        };
        if (window.requestAnimationFrame) window.requestAnimationFrame(run);
        else setTimeout(run, 16);
    }

    function boot() {
        syncAll();

        // PRIMARY trigger. Picking or removing a class rewrites the token DOM, and
        // Svelte does that AFTER the click handler returns — so an observer, not a
        // click listener, is what sees the real selection. Also covers programmatic
        // changes (preset load, paste, Reset, detector switch), where it simply
        // rewrites the same value Python already sent.
        const obs = new MutationObserver(scheduleSync);
        obs.observe(document.body, { childList: true, subtree: true });

        // Toggling "Exclude selected (NOT)" moves the CSV between the two hidden
        // fields. `checked` is already updated when this fires, so sync at once.
        document.addEventListener(
            "change",
            function (e) {
                const t = e.target;
                if (t && t.closest && t.closest('div[id*="_ad_model_classes_exclude"]')) {
                    syncAll();
                }
            },
            true
        );

        // BELT AND BRACES. Capture phase runs before the WebUI's own handler, so
        // even if an observer frame hasn't landed yet, anything that can start a
        // generation carries fresh values. Cheap: a handful of DOM reads, and
        // writeField no-ops when nothing changed.
        document.addEventListener(
            "click",
            function (e) {
                if (e.target && e.target.closest && e.target.closest("button")) syncAll();
            },
            true
        );
        document.addEventListener(
            "keydown",
            function (e) {
                // Ctrl+Enter / Alt+Enter — the WebUI's Generate shortcuts.
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey || e.altKey)) syncAll();
            },
            true
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
