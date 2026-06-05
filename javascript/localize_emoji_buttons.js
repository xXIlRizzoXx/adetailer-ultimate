/* ADetailer Ultimate — translation patch for elements Forge's walker skips.
 *
 * Forge's bundled `javascript/localization.js` walks DOM text nodes and the
 * `placeholder` + `title` attributes, but deliberately bails on:
 *   - any text node whose content matches its `re_emoji` regex
 *     (Extended_Pictographic + skin-tone + hair modifiers) — so EVERY
 *     emoji-prefixed label stays English: the action buttons (📂 Load,
 *     ✏️ Rename, …) AND the "🔁 Combine all tabs" detection-preview checkbox.
 *   - the `value` attribute of `<input>` elements that Gradio uses to render
 *     the *selected value* of dropdown widgets — caps the "Use same
 *     checkpoint / VAE / sampler / scheduler" sentinel values.
 *
 * It also can't translate the Settings → ADetailer "Reset" help blurb: it's
 * an OptionHTML whose nested <b>/<code>/<i> tags split the sentence into
 * separate text nodes, and some fragments are generic words ("are", "not")
 * that are unsafe to key individually (they'd clobber unrelated UI text).
 *
 * This patch re-applies the dictionary to exactly those three blind spots:
 *   (1) emoji-prefixed text nodes      — strict complement to Forge's
 *                                         re_emoji skip; key-gated.
 *   (2) input[role=listbox] .value     — dropdown selected values.
 *   (3) #setting_ad_reset_info block   — whole-block innerHTML swap, matched
 *                                         on the block's tag-stripped text.
 * It deliberately leaves everything Forge already handles (labels, accordion
 * headers, .info() hints, placeholders) untouched, and adds NO keys of its
 * own — the dictionaries live entirely in the Language Diffusion extension.
 *
 * The transient detection-preview status messages ("⚠️ Pick a detector
 * first.", "✅ 3 detection(s).", …) are intentionally NOT translated: the
 * count/error variants are built with interpolation, so a dictionary lookup
 * could only ever cover some of them — leaving a mixed-language status line.
 * They have no keys, so step (1) walks past them as a no-op.
 *
 * All matching mirrors Forge: `text.trim()` → `window.localization[text]`.
 * No-op on the English ("None") locale where `window.localization` is empty.
 *
 * Idempotent: a node is only rewritten when its current text is a known
 * source key; after translation the text is the target string (not a key),
 * so the MutationObserver never loops on its own writes.
 */
(function () {
    "use strict";

    // Mirror of Forge's own re_emoji (javascript/localization.js). Forge's
    // canBeTranslated() returns false for any text node matching this, so
    // these are precisely the nodes Forge leaves for us — translating them
    // here can never collide with Forge's own pass.
    const re_emoji =
        /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F9B0}-\u{1F9B3}]/u;

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

        // (1) Emoji-prefixed text nodes — buttons, checkbox/radio labels, and
        // any other element Forge's re_emoji filter skips. Not restricted to
        // <button> anymore (that missed the "🔁 Combine all tabs" checkbox).
        // Gated twice: the node must carry an emoji (so we only ever touch
        // Forge's blind spot) AND its trimmed text must be a known key (so
        // un-keyed emoji text — e.g. the dynamic preview status line — is
        // left exactly as-is).
        const textWalker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        let node;
        while ((node = textWalker.nextNode())) {
            const raw = node.textContent;
            if (!raw || !re_emoji.test(raw)) continue;
            const tl = translateText(raw);
            if (tl !== null) {
                node.textContent = tl;
            }
        }

        // (2) Gradio dropdown selected values. Forge's walker handles
        // `placeholder` and `title` but not `value`. Gradio's dropdown
        // <input> has no `type` attribute and carries `role="listbox"`; we
        // target by that role so we never overwrite a free-text field the
        // user is typing into. The lookup is a no-op for non-key values, so
        // applying it broadly is safe.
        const inputs = (root.nodeType === 1 ? root : document).querySelectorAll(
            'input[role="listbox"]'
        );
        for (const inp of inputs) {
            const tl = translateText(inp.value);
            if (tl !== null) {
                inp.value = tl;
            }
        }
    }

    // (3) Whole-block translation for OptionHTML settings. Forge can't match
    // these because their inner tags fragment the text. We read the block's
    // full visible text (textContent, tags stripped — exactly what the LD key
    // is) and, if it's a known key whose value is HTML, replace the innerHTML
    // of the smallest element that still contains the whole block (Gradio's
    // content wrapper). Idempotent: once swapped, the visible text is the
    // translation, which isn't itself a key, so further runs return early.
    function translateHtmlBlock(id) {
        if (!window.localization) return;
        const host = document.getElementById(id);
        if (!host) return;
        const key = host.textContent.trim();
        if (!key) return;
        const value = window.localization[key];
        if (value === undefined || value === key) return;
        let target = host;
        for (const el of host.querySelectorAll("*")) {
            if (el.textContent.trim() === key) target = el;
        }
        target.innerHTML = value;
    }

    // OptionHTML settings this fork registers that need block-level handling.
    const HTML_BLOCK_IDS = ["setting_ad_reset_info"];

    function sweep() {
        translateInTree(document.body);
        for (const id of HTML_BLOCK_IDS) translateHtmlBlock(id);
    }

    function bootstrap() {
        if (
            !window.localization ||
            Object.keys(window.localization).length === 0
        ) {
            // English / no locale → Forge's core walker is a no-op, so are we.
            return;
        }

        // Initial sweep over whatever has already mounted at load time.
        sweep();

        // Re-translate as new nodes mount — lazy ADetailer accordion, preset
        // Load, cross-tab paste, dropdown re-render, the Settings tab opening
        // (where the reset block lives), and "Reload UI" without a full reload.
        const obs = new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const addedNode of m.addedNodes) {
                    if (addedNode.nodeType === 1) {
                        translateInTree(addedNode);
                    }
                }
            }
            for (const id of HTML_BLOCK_IDS) translateHtmlBlock(id);
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
