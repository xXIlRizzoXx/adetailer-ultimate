/* ADetailer Ultimate — native HTML title tooltips on the fork's action
 * buttons.
 *
 * Why: gr.Button doesn't expose an `info=` parameter the way Checkbox /
 * Dropdown / Slider do, so there's no built-in way to attach hover
 * documentation to a button. We inject the standard `title` HTML
 * attribute via JavaScript — the browser then renders a native tooltip
 * after ~1 s of hover. No new dependencies, no custom CSS, no z-index
 * stacking battles with Gradio's reactive store.
 *
 * Forge Neo / A1111 auto-load any .js file in the extension's
 * `javascript/` folder.
 *
 * If Gradio re-renders a button (rare, but happens during preset
 * load / paste), the MutationObserver re-applies the tooltips so they
 * survive the DOM swap.
 */

(function () {
    "use strict";

    // Map: elem_id substring → tooltip text. The substring match catches
    // both txt2img / img2img variants AND the suffixed-by-tab versions
    // (e.g. ad_copy_settings_2nd, _3rd, …). Per-tab buttons all carry the
    // same base id fragment, so one mapping covers every instance.
    const TOOLTIPS = {
        // Top-of-tab clipboard
        adetailer_ad_copy_settings:
            "Copy this tab's processing settings to the clipboard. Detector, class filter and per-tab enable are excluded so each tab can target a different region.",
        adetailer_ad_paste_settings:
            "Paste the clipboard's settings into this tab. Only enabled after a Copy has been done on a different tab.",

        // Preset library
        adetailer_ad_preset_load:
            "Load the selected preset into this tab's widgets (every field, including the classes dropdown).",
        adetailer_ad_preset_rename:
            "Rename the selected preset using the name in the textbox below. The file on disk is updated atomically.",
        adetailer_ad_preset_delete:
            "Delete the selected preset from disk. Dropdown refreshes across all tabs.",
        adetailer_ad_preset_save:
            "Save the current tab's full widget state as a named preset using the name in the textbox to the left.",
        adetailer_ad_preset_reset:
            "Reset this tab's widgets to the pydantic defaults. Preset selection and clipboard state are not touched.",

        // Preset library export / import
        adetailer_ad_preset_export_btn:
            "Export the entire preset library to a JSON file you can back up or share with other installs.",
        adetailer_ad_preset_import_btn:
            "Import a previously-exported preset JSON. Tick 'Overwrite on conflict' to replace existing presets with the same name; otherwise duplicates are skipped.",

        // Detection preview
        adetailer_ad_preview_btn:
            "Run the configured detector against the input image WITHOUT inpainting. The result shows bounding boxes / masks so you can tune confidence + class filter before committing to a full generation.",
    };

    function applyTooltips() {
        for (const [idFragment, tooltipText] of Object.entries(TOOLTIPS)) {
            // Match any element whose id contains the fragment — covers
            // <button id="..._adetailer_ad_copy_settings_2nd"> as well as
            // div wrappers (e.g. gr.DownloadButton renders as a button
            // inside a div, depending on Gradio version).
            const candidates = document.querySelectorAll(
                `[id*="${idFragment}"]`
            );
            for (const el of candidates) {
                // Prefer the inner <button> if the matched element is a
                // wrapper (cases where Gradio nests the actual click target).
                const target =
                    el.tagName === "BUTTON" ? el : el.querySelector("button") || el;
                if (!target.title) {
                    target.title = tooltipText;
                }
            }
        }
    }

    function boot() {
        applyTooltips();
        // Re-apply on DOM changes. Gradio sometimes re-renders pieces of
        // the tree (preset load, paste, tab switch) and the new nodes
        // come back without our title attributes. Cheap; the function is
        // idempotent thanks to the `!target.title` guard.
        const docObserver = new MutationObserver(() => applyTooltips());
        docObserver.observe(document.body, {
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
