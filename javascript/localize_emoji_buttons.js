/* ADetailer Ultimate — emoji-aware translation patch.
 *
 * Forge's bundled `javascript/localization.js` skips any text node whose
 * content matches its `re_emoji` regex (Extended_Pictographic + skin-tone
 * + hair modifiers). All of this extension's action buttons carry an
 * emoji prefix (📂 Load, ✏️ Rename, 🗑 Delete, 💾 Save preset, 🆕 Reset,
 * 📋 Copy settings, 📥 Paste settings, 🔍 Run detection preview, 🔄 Reset
 * ADetailer settings to defaults, 📤 Esport, 📥 Import), so the walker
 * bails on every single one — they stay English even when the user
 * picks a non-English locale via Forge's localization setting or via
 * the Language Diffusion extension.
 *
 * This patch re-applies translations only to the BUTTON text nodes the
 * core walker skipped — it deliberately does not touch labels,
 * accordion headers, info hints, placeholders, or any other element
 * type Forge already handles. Matching is byte-identical to Forge's
 * own logic: `text.trim()` → `window.localization[text]`. Skips on
 * English ("None") locale where `window.localization` is empty.
 *
 * Timing:
 *   - Initial sweep on DOMContentLoaded so the buttons translate as
 *     soon as the accordion mounts.
 *   - MutationObserver on document.body to re-translate buttons that
 *     Gradio rebuilds (preset load, paste, tab switch, reload UI).
 *
 * Idempotent: `if (tl !== text)` guards against re-assigning the same
 * value, so the observer doesn't loop on its own mutations.
 */
(function () {
    "use strict";

    function translateButtonsIn(root) {
        if (!window.localization) return;
        const walker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        let node;
        while ((node = walker.nextNode())) {
            const parent = node.parentElement;
            if (!parent || parent.tagName !== "BUTTON") continue;
            const text = node.textContent.trim();
            if (!text) continue;
            const tl = window.localization[text];
            if (tl !== undefined && tl !== text) {
                node.textContent = tl;
            }
        }
    }

    function bootstrap() {
        if (
            !window.localization ||
            Object.keys(window.localization).length === 0
        ) {
            // English / no locale → core walker already a no-op, so are we.
            return;
        }

        // Initial sweep over whatever has already mounted at load time.
        translateButtonsIn(document.body);

        // Re-translate whenever new nodes are added — covers the lazy
        // mount of the ADetailer accordion, preset Load, cross-tab paste,
        // and "Reload UI" without a full page reload.
        const obs = new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const addedNode of m.addedNodes) {
                    if (addedNode.nodeType === 1) {
                        translateButtonsIn(addedNode);
                    }
                }
            }
        });
        obs.observe(document.body, {
            childList: true,
            subtree: true,
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", bootstrap);
    } else {
        bootstrap();
    }
})();
