/* ADetailer Ultimate — a small "📖 Guide" quick-link in the accordion header.
 *
 * The header already shows a version badge (the `.ad-version-overlay` markdown,
 * pulled onto the accordion header bar by CSS). We inject a clickable
 * "📖 Guide" link just before the version text inside that same overlay, so it
 * reuses the overlay's existing positioning instead of fighting it with a
 * second absolute element. Clicking it opens the top-level "ADetailer Guide"
 * tab.
 *
 * Pure DOM manipulation (no Gradio-version-specific params), so it works the
 * same on A1111 (Gradio 3) and Forge / Forge Neo (Gradio 4). Forge Neo / A1111
 * auto-load any .js file in the extension's `javascript/` folder.
 *
 * The overlay itself is `pointer-events: none` (so header clicks still toggle
 * the accordion); the injected link re-enables pointer events on itself via CSS
 * and stops propagation so clicking it never also toggles the accordion.
 */

(function () {
    "use strict";

    const GUIDE_TAB_LABEL = "ADetailer Guide";

    function clickGuideNav() {
        // Primary path (translation-proof): A1111 / Forge give every top-level
        // tab panel a stable id `tab_<ifid>` — here `tab_adetailer_guide`. The
        // nav buttons in `.tab-nav` are in the SAME order as the panels, so we
        // click the button at the guide panel's ordinal position. This survives
        // localization, which rewrites the button's visible label.
        const panel = document.getElementById("tab_adetailer_guide");
        if (panel && panel.parentElement) {
            const panels = Array.prototype.filter.call(
                panel.parentElement.children,
                function (el) {
                    return el.id && el.id.indexOf("tab_") === 0;
                }
            );
            const idx = panels.indexOf(panel);
            const navWrap =
                document.querySelector("#tabs .tab-nav") ||
                document.querySelector(".tab-nav");
            if (idx >= 0 && navWrap) {
                const buttons = navWrap.querySelectorAll("button");
                if (buttons[idx]) {
                    buttons[idx].click();
                    return true;
                }
            }
        }
        // Fallback: match the nav button by its (English) label text.
        const navs = document.querySelectorAll(
            "#tabs .tab-nav button, .tab-nav > button"
        );
        for (const b of navs) {
            if (b.textContent && b.textContent.trim() === GUIDE_TAB_LABEL) {
                b.click();
                return true;
            }
        }
        return false;
    }

    function openGuideTab(e) {
        if (e) {
            e.preventDefault();
            e.stopPropagation();
        }
        clickGuideNav();
    }

    function injectGuideLinks() {
        // The guide tab is opt-in (Settings -> ADetailer, off by default). Only
        // show the header quick-link when the tab actually exists, so it's never
        // a dead link.
        if (!document.getElementById("tab_adetailer_guide")) {
            return;
        }
        // One overlay per ADetailer accordion instance (txt2img + img2img).
        const paras = document.querySelectorAll(
            'div[id*="adetailer_ad_version"].ad-version-overlay p'
        );
        for (const p of paras) {
            if (p.dataset.adGuideInjected) {
                continue;
            }
            p.dataset.adGuideInjected = "1";
            const a = document.createElement("a");
            a.className = "ad-guide-open";
            a.textContent = "📖 Guide";
            a.title = "Open the ADetailer Guide";
            a.setAttribute("role", "button");
            a.addEventListener("click", openGuideTab);
            // Insert "📖 Guide · " before the version text (the overlay is
            // right-anchored, so the link sits to the left of the version).
            p.insertBefore(document.createTextNode(" · "), p.firstChild);
            p.insertBefore(a, p.firstChild);
        }
    }

    function boot() {
        injectGuideLinks();
        // Gradio can re-render the accordion (preset load / paste / tab switch);
        // the observer re-injects. Idempotent thanks to the dataset guard.
        const obs = new MutationObserver(() => injectGuideLinks());
        obs.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
