/* ADetailer Ultimate — translation patch for elements Forge's walker skips.
 *
 * Forge's bundled `javascript/localization.js` walks DOM text nodes and
 * the `placeholder` + `title` attributes, but deliberately bails on
 * - any text node whose content matches its `re_emoji` regex
 *   (Extended_Pictographic + skin-tone + hair modifiers) — caps every
 *   emoji-prefixed button label in this fork (📂 Load, ✏️ Rename, …)
 * - the `value` attribute of `<input>` elements that Gradio uses to
 *   render the *selected value* of dropdown widgets — caps the
 *   "Use same checkpoint / VAE / sampler / scheduler" sentinel values
 *   in our Inpainting section.
 *
 * Both kinds of element stay English even when the user picks a
 * non-English locale via Forge's localization setting or via the
 * Language Diffusion extension. This patch re-applies the dict lookup
 * to those two specific element types only — it deliberately does not
 * touch labels, accordion headers, info hints, placeholders, or any
 * other element Forge already handles correctly.
 *
 * Matching mirrors Forge's own logic verbatim: `text.trim()` →
 * `window.localization[text]`. No-op on English ("None") locale where
 * `window.localization` is empty.
 *
 * Timing:
 *   - Initial sweep on DOMContentLoaded so the elements translate as
 *     soon as the accordion mounts.
 *   - MutationObserver on document.body to re-translate elements that
 *     Gradio rebuilds (preset Load, cross-tab Paste, dropdown re-render,
 *     Reload UI without full page reload).
 *
 * Idempotent: `if (tl !== text)` guards against re-assigning the same
 * value, so the observer doesn't loop on its own mutations.
 */
(function () {
    "use strict";

    function translateText(text) {
        if (!text) return null;
        const trimmed = text.trim();
        if (!trimmed) return null;
        const tl = window.localization[trimmed];
        if (tl === undefined || tl === trimmed) return null;
        return tl;
    }

    function translateInTree(root) {
        if (!window.localization) return;

        // (1) Text nodes whose parent is a <button> — covers emoji-
        // prefixed button labels that Forge's re_emoji filter skips.
        const textWalker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        let node;
        while ((node = textWalker.nextNode())) {
            const parent = node.parentElement;
            if (!parent || parent.tagName !== "BUTTON") continue;
            const tl = translateText(node.textContent);
            if (tl !== null) {
                node.textContent = tl;
            }
        }

        // (2) <input value="..."> for Gradio dropdown selected values —
        // Forge's walker only handles `placeholder` and `title`, not
        // `value`. Scope to inputs that look like dropdown selectors
        // (Gradio renders them inside a wrapper with role="combobox"
        // or class containing "wrap"); apply the dict lookup to their
        // current value. We don't filter by id because every dropdown
        // in any extension benefits from the same fix, but the lookup
        // is a no-op when the value isn't a known translation key, so
        // it's safe to apply broadly.
        const inputs = (root.nodeType === 1 ? root : document).querySelectorAll(
            'input[type="text"]'
        );
        for (const inp of inputs) {
            // Only target inputs that are inside a dropdown wrapper —
            // skip text fields the user types into (prompts, preset
            // name) where overwriting `value` would clobber input.
            const wrapper = inp.closest(".wrap, [role='combobox'], .secondary-wrap");
            if (!wrapper) continue;
            const tl = translateText(inp.value);
            if (tl !== null) {
                inp.value = tl;
            }
        }
    }

    function bootstrap() {
        if (
            !window.localization ||
            Object.keys(window.localization).length === 0
        ) {
            // English / no locale → core walker is already a no-op, so are we.
            return;
        }

        // Initial sweep over whatever has already mounted at load time.
        translateInTree(document.body);

        // Re-translate whenever new nodes are added — covers the lazy
        // mount of the ADetailer accordion, preset Load, cross-tab paste,
        // dropdown re-render after model selection, and "Reload UI"
        // without a full page reload.
        const obs = new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const addedNode of m.addedNodes) {
                    if (addedNode.nodeType === 1) {
                        translateInTree(addedNode);
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
