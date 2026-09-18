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
 */

(function () {
    "use strict";

    const FADE_DELAY_MS = 4000;
    const WATCH_CLASS = "ad-preset-status";
    const FADED_CLASS = "ad-status-faded";

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

        const tick = (records) => {
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

    function boot() {
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
