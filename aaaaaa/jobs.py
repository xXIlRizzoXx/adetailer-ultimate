"""Host job boundaries for ADetailer's standalone UI actions."""

from __future__ import annotations

from functools import wraps


def wrap_adetailer_job(func):
    """Run one image or an entire folder under the WebUI's shared GPU lock.

    Keep the callback's return values unchanged: the host's Gradio GPU wrapper
    adds HTML statistics to its last output, but our last output is Markdown.
    Only the outer action owns the state lifecycle; individual images and masks
    must preserve cancellation until this action has finished.
    """
    @wraps(func)
    def run(*args, **kwargs):
        from modules import shared
        from modules.call_queue import queue_lock

        with queue_lock:
            try:
                shared.state.begin(job="ADetailer")
            except TypeError:  # a host whose begin() takes no job label
                shared.state.begin()
            try:
                return func(*args, **kwargs)
            finally:
                try:
                    shared.state.end()
                finally:
                    shared.state.skipped = False
                    shared.state.interrupted = False
                    shared.state.stopping_generation = False

    return run


def wrap_adetailer_detection(func):
    """Serialize detector previews with generation without resetting its state."""
    @wraps(func)
    def run(*args, **kwargs):
        from modules.call_queue import queue_lock

        with queue_lock:
            return func(*args, **kwargs)

    return run
