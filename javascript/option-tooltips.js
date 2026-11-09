/* ADetailer Ultimate — hover tooltips on the panel's options.
 *
 * Why: most options of a tab (confidence, mask blur, padding, …) have only a
 * label, and adding `info=` text under each of them would make every tab
 * twice as long. Like button-tooltips.js, this sets the standard `title`
 * attribute, here on the option's whole block (label and control), so the
 * browser shows the explanation after about a second of hover. Options that
 * already show an `info=` line under them are left out.
 *
 * Forge Neo / A1111 auto-load any .js file in the extension's
 * `javascript/` folder.
 */

(function () {
    "use strict";

    // Map: elem_id key (without the txt2img / img2img prefix and the tab
    // suffix _2nd, _3rd, …) → tooltip text. One entry covers every tab.
    const TOOLTIPS = {
        // Top of the panel and of each tab
        ad_skip_img2img:
            "img2img only. Skips the img2img generation and runs only ADetailer on your input image.",
        ad_tab_enable:
            "Runs this tab. The tab also needs a detector other than None.",
        ad_preset_dropdown:
            "Your saved presets. Pick one, then Load puts its settings into this tab.",
        ad_preset_name:
            "The name used by Save preset, and the new name used by Rename.",
        ad_preset_reset_all:
            "With this ticked, Reset puts every tab of the panel back to its defaults, not only this one.",
        ad_paste_swap:
            "With this ticked, Paste swaps the two tabs: this tab gets the copied settings and the copied tab gets this tab's settings. The clipboard is emptied afterwards.",
        ad_paste_all:
            "With this ticked, Paste puts the copied settings into every tab of the panel except the copied one.",
        ad_preset_import_overwrite:
            "When importing, replace a saved preset that has the same name. Unticked, those presets are skipped.",

        // Detector and classes
        ad_model_classes:
            "YOLO-World only: the names to look for, separated by commas, for example person, hand. Left empty, it looks for its 80 default classes.",
        ad_model_classes_exclude:
            "Turns the class filter around: every class except the ones you pick is detailed. With no class picked, nothing is left out.",
        ad_classes_sequential:
            "Runs a separate detect and inpaint pass for each selected class, in the order you picked them, each on the result of the previous one. Needs two or more classes; it does not run in Exclude (NOT) mode, with Merge and Invert, or with MediaPipe detectors other than face features.",

        // Inpaint prompts
        ad_prompt:
            "The prompt for the detailer pass. Blank uses the main prompt. [SEP] gives each region its own part, [SKIP] skips a region and [PROMPT] puts the main prompt at that spot.",
        ad_negative_prompt:
            "The negative prompt for the detailer pass. Blank uses the main negative prompt. [SEP] and [PROMPT] work here too.",
        ad_prompt_append:
            "Words always added to the end of the detailer prompt, also when it takes the main prompt. With [SEP] they are added to every part.",
        ad_negative_prompt_append:
            "Words always added to the end of the detailer negative prompt, also when it takes the main negative prompt.",
        ad_class_prompts:
            "One line per class, as class: prompt | negative prompt. Used with Process classes sequentially and two or more classes selected.",
        ad_use_main_loras:
            "Adds the LoRA tags of the main prompt to this tab's own prompt. A LoRA that the tab prompt already names keeps the tab's weight.",

        // Detection
        ad_confidence:
            "Only detections the detector is more sure of than this are used. Lower finds more regions, and more false ones.",
        ad_mask_filter_method:
            "What 'Mask only the top k' ranks the detections by: their area or their confidence.",
        ad_mask_k:
            "Keeps only the k largest, or most confident, detections. 0 keeps them all.",
        ad_mask_min_ratio:
            "Leaves out detections smaller than this share of the whole image. About 0.01 leaves out small things in the background.",
        ad_mask_max_ratio:
            "Leaves out detections larger than this share of the whole image. 1 keeps every size.",

        // Mask preprocessing (applied in this order: offset, erosion or dilation, merge mode)
        ad_x_offset:
            "Moves the mask left (negative values) or right (positive values), in pixels.",
        ad_y_offset:
            "Moves the mask down (negative values) or up (positive values), in pixels.",
        ad_dilate_erode:
            "Shrinks the mask (negative values) or grows it (positive values), after the offset.",

        // Inpainting
        ad_mask_blur:
            "Softens the edge of the mask, for a smoother blend with the rest of the picture.",
        ad_denoising_strength:
            "How much the region changes: lower keeps more of the original, higher redraws more. About 0.3 to 0.5 is typical.",
        ad_inpaint_only_masked:
            "On: only the region and some of its surroundings are redrawn, at full resolution, then pasted back. Off: the whole picture is inpainted.",
        ad_inpaint_only_masked_padding:
            "How many pixels around the region are redrawn with it when Inpaint only masked is on. More gives the model more context.",
        ad_use_inpaint_width_height:
            "Redraws the region, or the whole picture with Inpaint only masked off, at the inpaint width and height below. It wins over Scale inpaint to bbox.",
        ad_inpaint_width:
            "The width the region, or the whole picture with Inpaint only masked off, is redrawn at, with Use separate width/height on.",
        ad_inpaint_height:
            "The height the region, or the whole picture with Inpaint only masked off, is redrawn at, with Use separate width/height on.",
        ad_resolution_scale:
            "With Scale inpaint to bbox on, each region is redrawn at its box size times this value: 1.5 redraws a 256x320 box at 384x480. Needs Inpaint only masked on.",
        ad_use_steps:
            "Uses the steps set below for the detailer pass only. Off, it uses the main steps.",
        ad_steps:
            "The steps of the detailer pass, with Use separate steps on.",
        ad_use_cfg_scale:
            "Uses the CFG scale set below for the detailer pass only. Off, it uses the main CFG scale.",
        ad_cfg_scale:
            "The CFG scale of the detailer pass, with Use separate CFG scale on.",
        ad_use_checkpoint:
            "Uses another checkpoint for the detailer pass only. Your main checkpoint comes back after the pass.",
        ad_checkpoint:
            "The checkpoint of the detailer pass, with Use separate checkpoint on.",
        ad_use_vae:
            "Uses another VAE for the detailer pass only. Your main VAE comes back after the pass.",
        ad_vae:
            "The VAE of the detailer pass, with Use separate VAE on.",
        ad_use_text_encoder:
            "Forge and Forge Neo only: uses another text encoder for the detailer pass. On AUTOMATIC1111 it does nothing.",
        ad_text_encoder:
            "The text encoder of the detailer pass. None lets the detailer checkpoint use its own.",
        ad_use_sampler:
            "Uses the sampler, and the scheduler where the WebUI has them, set below for the detailer pass only.",
        ad_sampler:
            "The sampler of the detailer pass, with Use separate sampler on.",
        ad_scheduler:
            "The scheduler of the detailer pass, with Use separate sampler on.",
        ad_use_noise_multiplier:
            "Uses the noise multiplier set below for the detailer pass only.",
        ad_noise_multiplier:
            "The noise multiplier for img2img of the detailer pass, with Use separate noise multiplier on.",
        ad_use_clip_skip:
            "Uses the CLIP skip set below for the detailer pass only. Your main CLIP skip comes back after the pass.",
        ad_clip_skip:
            "The CLIP skip of the detailer pass, with Use separate CLIP skip on.",
        ad_restore_face:
            "Runs the WebUI's face restoration on the result of the detailer pass, where the WebUI has one.",

        // ControlNet
        ad_controlnet_model:
            "A ControlNet model for the detailer pass, separate from the ControlNet of the main generation. Passthrough uses the ControlNet settings made outside ADetailer.",
        ad_controlnet_module:
            "The preprocessor for the ControlNet model. Choosing a model picks its default preprocessor, unless the current one fits the new model.",
        ad_controlnet_weight:
            "How strongly ControlNet guides the detailer pass.",
        ad_controlnet_guidance_start:
            "When ControlNet starts guiding, as a share of the steps: 0 is the first step.",
        ad_controlnet_guidance_end:
            "When ControlNet stops guiding, as a share of the steps: 1 is the last step.",

        // Detection preview and Run ADetailer on an image
        ad_preview_all_tabs:
            "With this ticked, Run detection preview runs the detector of every tab that has one, enabled or not, and draws all the results on one image, one colour per tab.",
        ad_apply_save:
            "Also saves the result in the ADetailer-Inpaint folder of your outputs. A folder batch always saves its results.",
    };

    // script_txt2img_adetailer_<key> plus an optional tab suffix (_2nd, _11th, …).
    const ID = /^script_(?:txt2img|img2img)_adetailer_(.+?)(?:_\d+(?:st|nd|rd|th))?$/;

    function applyTooltips() {
        const l10n = window.localization;
        for (const el of document.querySelectorAll('[id*="_adetailer_ad_"]')) {
            const match = ID.exec(el.id);
            const text = match && TOOLTIPS[match[1]];
            if (!text) continue;
            if (!el.title) {
                // The WebUI's localizer translates a title only when the node
                // is added, before this runs: look it up here, as
                // button-tooltips.js and the WebUI's own hints.js do.
                const translated = l10n && l10n[text];
                el.title = typeof translated === "string" && translated ? translated : text;
            }
            // AUTOMATIC1111's hints.js empties the title of a dropdown's input
            // each time its list opens, and an empty title hides the block's:
            // give the inputs the block's text, except where hints.js has a
            // text of its own for the value (such as "Euler a" or "None"),
            // which it puts there a second later.
            const hints = window.titles;
            for (const input of el.querySelectorAll("input, textarea")) {
                if (input.title || (hints && hints[input.value])) continue;
                input.title = el.title;
            }
        }
    }

    function boot() {
        applyTooltips();
        // Gradio re-renders pieces of the panel (preset load, paste, tab
        // switch) without our titles. The page also changes many times a
        // second while generating, so one pass per animation frame at most.
        let pending = false;
        const docObserver = new MutationObserver(() => {
            if (pending) return;
            pending = true;
            window.requestAnimationFrame(() => {
                pending = false;
                applyTooltips();
            });
        });
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
