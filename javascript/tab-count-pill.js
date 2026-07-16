/* ADetailer Ultimate — a small green "active tabs" pill in the accordion
 * header, à la ControlNet Integrated's "1x Unit" counter.
 *
 * It shows, even when the accordion is collapsed, how many ADetailer detector
 * tabs are ACTIVE — i.e. how many "Enable this tab" checkboxes are ticked
 * (this is exactly the per-tab activation control). Hidden when the count is 0.
 *
 * Pure client-side DOM (no Gradio components, no event listeners on the Python
 * side), so it cannot affect any Gradio wiring / fn_index — index-safe. Works
 * the same on A1111 (Gradio 3) and Forge / Forge Neo (Gradio 4); WebUIs
 * auto-load javascript/*.js. Reactivity: a light childList observer (for
 * re-renders) plus delegated change/input listeners scoped to the accordion
 * (for the "Enable this tab" checkbox toggles).
 */

(function () {
    "use strict";

    const ACCORDION_SEL = 'div[id*="adetailer_ad_main_accordion"]';

    // Count active detector tabs inside one ADetailer accordion: how many
    // "Enable this tab" checkboxes are ticked. Scoped to this accordion, so the
    // txt2img and img2img instances count independently. Hidden inner-tab
    // panels keep their checkbox in the DOM, so every tab is counted.
    function countActiveTabs(accordion) {
        let n = 0;
        const boxes = accordion.querySelectorAll(
            '[id*="_ad_tab_enable"] input[type="checkbox"]'
        );
        boxes.forEach(function (box) {
            if (box.checked) n += 1;
        });
        return n;
    }

    // Find (or create) this accordion's pill in its header, so it stays visible
    // whether the accordion is expanded or collapsed. Returns null for elements
    // that carry the accordion id fragment but are NOT the real accordion root
    // (e.g. the hidden InputAccordion checkbox block) — they have no direct
    // `.label-wrap` header — so no spurious/duplicate pills are created.
    function ensurePill(accordion) {
        const header = accordion.querySelector(":scope > .label-wrap");
        if (!header) return null;
        let pill = header.querySelector(".ad-tab-count-pill");
        if (pill) return pill;
        pill = document.createElement("span");
        pill.className = "ad-tab-count-pill";
        pill.setAttribute("aria-hidden", "true");
        // Match ControlNet Integrated: put the badge INSIDE the title span,
        // right after the "ADetailer" text (hugging the title on the left), not
        // as a far-right sibling before the chevron. The first direct-child
        // <span> of the label-wrap is the title (the chevron is `.icon`).
        const titleSpan = header.querySelector(":scope > span");
        (titleSpan || header).appendChild(pill);
        return pill;
    }

    function update() {
        const accordions = document.querySelectorAll(ACCORDION_SEL);
        accordions.forEach(function (acc) {
            const pill = ensurePill(acc);
            if (!pill) return;
            const n = countActiveTabs(acc);
            // Guard every write: the childList observer sees a textContent
            // rewrite (it replaces the text node), so writing only when the
            // value actually changed lets the observer settle instead of
            // re-firing every frame. Visibility is toggled via a CLASS (an
            // attribute change the childList observer ignores), never inline
            // display, so hiding also can't retrigger the observer — and it
            // wins over the base rule via higher specificity, not !important
            // races.
            const text = n + "x Tab" + (n === 1 ? "" : "s");
            if (pill.textContent !== text) {
                pill.textContent = text;
            }
            const title = n + " active ADetailer tab" + (n === 1 ? "" : "s");
            if (pill.title !== title) {
                pill.title = title;
            }
            const hidden = n === 0;
            if (pill.classList.contains("ad-pill-hidden") !== hidden) {
                pill.classList.toggle("ad-pill-hidden", hidden);
            }
        });
    }

    function scheduleUpdate() {
        // Coalesce bursts of DOM/events into a single update on the next frame.
        if (scheduleUpdate._q) return;
        scheduleUpdate._q = true;
        const run = function () {
            scheduleUpdate._q = false;
            update();
        };
        if (window.requestAnimationFrame) {
            window.requestAnimationFrame(run);
        } else {
            setTimeout(run, 16);
        }
    }

    function boot() {
        update();
        // Structural re-renders (preset load, tab switch, paste) — cheap, and
        // the pill is reused so this is idempotent.
        const obs = new MutationObserver(scheduleUpdate);
        obs.observe(document.body, { childList: true, subtree: true });
        // Live reactions to toggling "Enable this tab" or picking a detector.
        const onEvt = function (e) {
            if (
                e.target &&
                e.target.closest &&
                e.target.closest(ACCORDION_SEL)
            ) {
                scheduleUpdate();
            }
        };
        document.addEventListener("change", onEvt, true);
        document.addEventListener("input", onEvt, true);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
