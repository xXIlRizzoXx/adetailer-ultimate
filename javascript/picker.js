/* ADetailer Ultimate — click the boxes of the Detection preview to pick them.
 *
 * A single-tab "🔍 Run detection preview" numbers every box (#1, #2, …), and
 * the tab's "Inpaint only these detections" field keeps only the numbers typed
 * in it. This script lets you click the boxes instead of typing: a click on a
 * box adds its number to that field, a second click takes it out, and the boxes
 * whose numbers are in the field are marked. An empty field still means "every
 * detection".
 *
 * HOW
 * ---
 * The preview button's existing .click also writes the boxes (JSON: picture
 * size and [x1, y1, x2, y2] per box, in the order of the numbers) into a hidden
 * Textbox, `<tab>_ad_preview_boxes`. This script lays a transparent box over
 * the preview picture for each of them, scaled to where the picture is drawn
 * (object-fit letterbox), and writes the numbers into `<tab>_ad_inpaint_indices`
 * with an `input` event, as class-sync.js writes its fields.
 *
 * SAFETY
 * ------
 * Pure client-side DOM: no Gradio component of its own and no Python-side
 * event listener, so fn_index / dependency ordering is untouched (index-safe).
 * Whenever something is not as expected (no boxes, no picture, a picture whose
 * shape does not match the boxes, a field it cannot find) it shows nothing and
 * writes nothing: the preview then works as before, with the numbers typed by
 * hand. Works on A1111 (Gradio 3) and Forge / Forge Neo / reForge (Gradio 4).
 */

(function () {
    "use strict";

    const OUT_KEY = "_ad_preview_output";
    const LAYER_CLASS = "ad-pick-layer";

    function fieldOf(wrapperId) {
        const box = document.getElementById(wrapperId);
        return box ? box.querySelector("textarea, input") : null;
    }

    function writeField(field, value) {
        if (!field || field.value === value) return;
        field.value = value;
        field.dispatchEvent(new Event("input", { bubbles: true }));
    }

    // The 1-based numbers in an "Inpaint only these detections" text, read as
    // adetailer/mask.py parse_indices reads it: "1,3,5", "1 3 5", "1-3,5",
    // "#2", full-width digits and the CJK punctuation an IME types. A range
    // is clipped to the `n` boxes shown; a single number is kept as it is.
    function parseNumbers(text, n) {
        let spec = String(text || "");
        try {
            spec = spec.normalize("NFKC");
        } catch (e) {
            /* keep it as it is */
        }
        spec = spec.replace(/[~〜‐-―−ー]/g, "-");
        spec = spec.replace(/#/g, "").replace(/\s*-\s*/g, "-");
        const out = [];
        const tokens = spec.split(/[,;\s、・]+/);
        for (let i = 0; i < tokens.length; i += 1) {
            const tok = tokens[i].trim();
            if (!tok) continue;
            const range = /^(\d+)-+(\d+)$/.exec(tok);
            if (range) {
                let lo = parseInt(range[1], 10);
                let hi = parseInt(range[2], 10);
                if (lo > hi) {
                    const t = lo;
                    lo = hi;
                    hi = t;
                }
                for (let v = Math.max(1, lo); v <= Math.min(n, hi); v += 1) {
                    if (out.indexOf(v) < 0) out.push(v);
                }
            } else if (/^\d+$/.test(tok)) {
                const v = parseInt(tok, 10);
                if (v >= 1 && out.indexOf(v) < 0) out.push(v);
            }
        }
        return out.sort(function (a, b) {
            return a - b;
        });
    }

    function readBoxes(outputId) {
        const field = fieldOf(outputId.replace(OUT_KEY, "_ad_preview_boxes"));
        if (!field || !field.value) return null;
        try {
            const data = JSON.parse(field.value);
            if (!data || !(data.w > 0) || !(data.h > 0) || !Array.isArray(data.boxes)) {
                return null;
            }
            return data;
        } catch (e) {
            return null;
        }
    }

    // The biggest loaded picture inside the preview output.
    function pictureOf(output) {
        const imgs = output.querySelectorAll("img");
        let best = null;
        for (let i = 0; i < imgs.length; i += 1) {
            const im = imgs[i];
            if (!im.naturalWidth || !im.naturalHeight) continue;
            if (!best || im.naturalWidth * im.naturalHeight > best.naturalWidth * best.naturalHeight) {
                best = im;
            }
        }
        return best;
    }

    // Where the picture's pixels are drawn inside its <img> element, relative
    // to `host`: the content box, letterboxed by object-fit "contain" or
    // "scale-down", or stretched by "fill". Null for any other fit.
    function drawnRect(img, host) {
        const r = img.getBoundingClientRect();
        const h = host.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) return null;
        const style = window.getComputedStyle(img);
        const fit = style.objectFit || "fill";
        const nw = img.naturalWidth;
        const nh = img.naturalHeight;
        let w = r.width;
        let ht = r.height;
        if (fit === "contain" || fit === "scale-down") {
            let s = Math.min(r.width / nw, r.height / nh);
            if (fit === "scale-down") s = Math.min(s, 1);
            w = nw * s;
            ht = nh * s;
        } else if (fit !== "fill") {
            return null;
        }
        return {
            left: r.left - h.left + (r.width - w) / 2,
            top: r.top - h.top + (r.height - ht) / 2,
            width: w,
            height: ht,
        };
    }

    function removeLayer(output) {
        const old = output.querySelector("." + LAYER_CLASS);
        if (old) old.remove();
    }

    function toggle(outputId, number, n) {
        const field = fieldOf(outputId.replace(OUT_KEY, "_ad_inpaint_indices"));
        if (!field) return;
        const picked = parseNumbers(field.value, n);
        const at = picked.indexOf(number);
        if (at >= 0) picked.splice(at, 1);
        else picked.push(number);
        picked.sort(function (a, b) {
            return a - b;
        });
        writeField(field, picked.join(","));
        update(document.getElementById(outputId));
    }

    function update(output) {
        if (!output || !output.id) return;
        const data = readBoxes(output.id);
        const img = data ? pictureOf(output) : null;
        const field = fieldOf(output.id.replace(OUT_KEY, "_ad_inpaint_indices"));
        // The boxes belong to the picture shown only when the shapes agree.
        if (
            !data ||
            !img ||
            !field ||
            Math.abs(img.naturalWidth / img.naturalHeight - data.w / data.h) > 0.02
        ) {
            removeLayer(output);
            return;
        }
        const rect = drawnRect(img, output);
        if (!rect) {
            removeLayer(output);
            return;
        }
        const n = data.boxes.length;
        const picked = parseNumbers(field.value, n);
        const key = [
            output.querySelector("." + LAYER_CLASS) ? 1 : 0,
            JSON.stringify(data.boxes),
            data.w,
            data.h,
            picked.join(","),
            Math.round(rect.left),
            Math.round(rect.top),
            Math.round(rect.width),
            Math.round(rect.height),
        ].join("|");
        if (output._adPickKey === key) return;
        output._adPickKey = key;

        removeLayer(output);
        if (window.getComputedStyle(output).position === "static") {
            output.style.position = "relative";
        }
        const layer = document.createElement("div");
        layer.className = LAYER_CLASS;
        const sx = rect.width / data.w;
        const sy = rect.height / data.h;
        // Larger boxes first, so a smaller box inside one (the eyes inside the
        // face) is drawn over it and still gets its clicks.
        const area = (b) =>
            Array.isArray(b) && b.length >= 4 ? (b[2] - b[0]) * (b[3] - b[1]) : 0;
        const order = data.boxes
            .map((b, k) => k)
            .sort((i, j) => area(data.boxes[j]) - area(data.boxes[i]) || i - j);
        for (const k of order) {
            const b = data.boxes[k];
            if (!Array.isArray(b) || b.length < 4) continue;
            const number = k + 1;
            const box = document.createElement("div");
            box.className = "ad-pick-box" + (picked.indexOf(number) >= 0 ? " ad-picked" : "");
            box.style.left = rect.left + b[0] * sx + "px";
            box.style.top = rect.top + b[1] * sy + "px";
            box.style.width = Math.max(4, (b[2] - b[0]) * sx) + "px";
            box.style.height = Math.max(4, (b[3] - b[1]) * sy) + "px";
            box.title = "#" + number;
            box.addEventListener("click", function (e) {
                e.preventDefault();
                e.stopPropagation();
                toggle(output.id, number, n);
            });
            layer.appendChild(box);
        }
        output.appendChild(layer);
    }

    // A tab's preview output: "<mode>_ad_preview_output" plus, from the 2nd
    // tab on, a "_2nd"-like ending.
    const OUT_ID = /_ad_preview_output(_\w+)?$/;

    function updateAll() {
        const outputs = document.querySelectorAll('div[id*="' + OUT_KEY + '"]');
        for (let i = 0; i < outputs.length; i += 1) {
            if (!OUT_ID.test(outputs[i].id)) continue;
            try {
                update(outputs[i]);
            } catch (e) {
                removeLayer(outputs[i]);
            }
        }
    }

    function boot() {
        // The hidden boxes field and the picture change without an event this
        // script could hear (Gradio sets them), and the picture moves when the
        // window or the panel is resized: a light check a few times a second,
        // which redraws only when something changed.
        setInterval(updateAll, 300);
        document.addEventListener(
            "input",
            function (e) {
                const t = e.target;
                if (t && t.closest && t.closest('div[id*="_ad_inpaint_indices"]')) updateAll();
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
