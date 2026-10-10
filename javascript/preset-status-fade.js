/* ADetailer Ultimate — auto-clear of the preset / IO status messages.
 *
 * The preset library widgets (Load / Save / Delete / Rename / Reset +
 * Export / Import) write feedback messages like "✅ Loaded 'X'." or
 * "➕ 2 added · ⏭ 1 skipped" into a small markdown widget below the
 * action row. Without auto-clearing those messages linger forever — the
 * user has to keep ignoring them. This script watches the relevant
 * markdown containers and hides them ~4 seconds after a new non-empty
 * message appears, so the UI returns to a clean state.
 *
 * Mechanics:
 * - Look for any element with class `.ad-preset-status` (preset
 *   button row) or `.ad-preview-status` is excluded (those are
 *   warnings that should stay until the user fixes the precondition).
 *   Gradio also copies the class onto the inner markdown div; only the
 *   outer block is watched.
 * - On any DOM mutation inside that element, sample the inner <p>
 *   text. If it changed AND is non-empty, schedule a hide in 4 s.
 * - Hide by adding the `ad-status-faded` class (a CSS rule hides the
 *   block). The text itself is left alone: Gradio does not redraw a
 *   value that did not change, so text wiped behind its back never came
 *   back when the same message was sent again (a retried Save that
 *   failed again showed nothing).
 * - The same message sent again is recognised by the `pending` class
 *   Gradio puts on the markdown's wrapper while a request that writes
 *   this status runs: when it goes away, the message is shown again and
 *   the cycle restarts. Reset writes every tab's status, so another
 *   tab's last message can come back for 4 s; those tabs are not on
 *   screen at the time.
 * - Gradio 4 applies a fast answer's `pending` and `complete` states in
 *   the same animation frame, so an unchanged message then changes
 *   nothing in the DOM at all. A click on a tab's Load / Save / Rename /
 *   Delete / Reset / Export button arms a check of that tab's status:
 *   if the status saw no change at all ARM_MS later, the answer was the
 *   same message, and it is shown again. A plain DOM listener, not a
 *   Gradio event (index-safe).
 * - Import answers a file upload, after the file dialog has closed, so a
 *   click cannot arm it: Gradio's bubbling "gradio" DOM event for the
 *   Import button's "upload" arms that tab's import/export status instead
 *   (the button is found by its component id in window.gradio_config).
 */

(function () {
    "use strict";

    const FADE_DELAY_MS = 4000;
    const ARM_MS = 300;
    const WATCH_CLASS = "ad-preset-status";
    const FADED_CLASS = "ad-status-faded";
    // "<prefix>_ad_preset_<action>[_2nd …]" writes "<prefix>_ad_preset_status"
    // (Export: "_ad_preset_io_status") with the same tab suffix. The "Reset
    // every tab" checkbox (…_ad_preset_reset_all) does not match.
    const ACTION_RE =
        /^(.*)_ad_preset_(load|save|rename|delete|reset|export_btn)((?:_\d+(?:st|nd|rd|th))?)$/;
    // "<prefix>_ad_preset_import_btn[_2nd …]" writes "<prefix>_ad_preset_io_status".
    const IMPORT_RE = /^(.*)_ad_preset_import_btn((?:_\d+(?:st|nd|rd|th))?)$/;

    // A request that writes this status has just finished: the `pending`
    // class left an element inside it (Gradio 3 and 4 markdown wrapper).
    function answered(records, statusEl) {
        return (records || []).some(
            (r) =>
                r.type === "attributes" &&
                r.target !== statusEl &&
                /(^|\s)pending(\s|$)/.test(r.oldValue || "") &&
                !r.target.classList.contains("pending")
        );
    }

    function setupWatcher(statusEl) {
        if (statusEl.__adStatusFadeAttached) return;
        const outer = statusEl.parentElement;
        if (outer && outer.closest("." + WATCH_CLASS)) return; // inner copy
        statusEl.__adStatusFadeAttached = true;

        let timer = null;
        let lastSeen = "";
        let armed = null;

        const tick = (records) => {
            // Any change inside the block (our own fade class aside): the
            // answer is being shown by the normal path.
            if (armed && (records || []).some((r) => r.target !== statusEl)) {
                clearTimeout(armed);
                armed = null;
            }
            const md = statusEl.querySelector(".md");
            if (!md) return;
            const text = (md.textContent || "").trim();
            if (text && (text !== lastSeen || answered(records, statusEl))) {
                lastSeen = text;
                statusEl.classList.remove(FADED_CLASS);
                if (timer) clearTimeout(timer);
                timer = setTimeout(() => {
                    // Hide if the message is still the same one we
                    // scheduled for — don't hide a fresher message
                    // that arrived in the meantime.
                    timer = null;
                    const fresh = (md.textContent || "").trim();
                    if (fresh === lastSeen) {
                        statusEl.classList.add(FADED_CLASS);
                    }
                }, FADE_DELAY_MS);
            } else if (!text && lastSeen) {
                // Outside reset (e.g. Gradio overwrote with empty) —
                // forget our timer state.
                if (timer) {
                    clearTimeout(timer);
                    timer = null;
                }
                lastSeen = "";
                statusEl.classList.remove(FADED_CLASS);
            }
        };

        // A click on this status's own action button: nothing changed at
        // all ARM_MS later means the same message came back unseen.
        statusEl.__adStatusArm = () => {
            if (armed) clearTimeout(armed);
            armed = setTimeout(() => {
                armed = null;
                lastSeen = "";
                tick();
            }, ARM_MS);
        };

        // Initial check + observer on the .md subtree so we catch
        // Gradio overwrites + Svelte rerenders, and on class changes so
        // we see a request finish.
        tick();
        const obs = new MutationObserver(tick);
        obs.observe(statusEl, {
            childList: true,
            subtree: true,
            characterData: true,
            attributes: true,
            attributeFilter: ["class"],
            attributeOldValue: true,
        });
    }

    function scanAll() {
        document
            .querySelectorAll("." + WATCH_CLASS)
            .forEach((el) => setupWatcher(el));
    }

    // Walk up from the click: on Gradio 4 the Export button's id sits on a
    // wrapper around the real button.
    function onAction(e) {
        for (let n = e.target; n && n.nodeType === 1; n = n.parentElement) {
            const m = ACTION_RE.exec(n.id || "");
            if (!m) continue;
            const kind = m[2] === "export_btn" ? "_ad_preset_io_status" : "_ad_preset_status";
            const statusEl = document.getElementById(m[1] + kind + m[3]);
            if (statusEl && statusEl.__adStatusArm) statusEl.__adStatusArm();
            return;
        }
    }

    // Gradio dispatches it on its app root with the component's id, before
    // the request starts.
    function onUpload(e) {
        const d = e && e.detail;
        if (!d || d.event !== "upload") return;
        const cfg = window.gradio_config;
        const comps = cfg && Array.isArray(cfg.components) ? cfg.components : [];
        const comp = comps.find((c) => c && c.id === d.id);
        const m = IMPORT_RE.exec((comp && comp.props && comp.props.elem_id) || "");
        if (!m) return;
        const statusEl = document.getElementById(m[1] + "_ad_preset_io_status" + m[2]);
        if (statusEl && statusEl.__adStatusArm) statusEl.__adStatusArm();
    }

    function boot() {
        document.addEventListener("click", onAction, true);
        document.addEventListener("gradio", onUpload, true);
        scanAll();
        // Re-scan on body mutations so we attach to status elements
        // that mount lazily (e.g. when a tab is first opened).
        const docObs = new MutationObserver(() => scanAll());
        docObs.observe(document.body, {
            childList: true,
            subtree: true,
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
