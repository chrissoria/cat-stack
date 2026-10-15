"""
Raw-reply capture for classify(test=True).

A diagnostic aid: with test=True, classify() keeps what each model actually
replied for each row, next to the parsed 0/1 matrix, so a row that came back
"success" with every category 0 can be traced to the reply behind it. Off by
default; nothing here runs unless test=True.

Columns added to the output (suffixed with "_" + the model's sanitized name
when more than one model is used):

    raw_reply     the text the client returned for the final attempt
                  (chunked runs: each chunk's final reply, joined under a
                  "----- chunk i/n -----" header)
    raw_attempts  LLM calls made for the row (JSON retries, batch retries,
                  chunks and verification calls included)
    parsed_json   the JSON string handed to aggregation
    raw_error     the error string, if any
"""

import threading

RAW_COLUMNS = ("raw_reply", "raw_attempts", "parsed_json", "raw_error")


def _chunk_header(i, n):
    return f"----- chunk {i}/{n} -----"


class RawReplyRecorder:
    """Records every LLM call made while classifying one (row, model) pair.

    One recorder per classify_single() call, so a recorder is never shared
    across rows or models; the lock only guards against a caller that does.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.attempts = 0
        self._segments = [[]]
        self._chunked = False
        self._chunk_total = None

    def start_chunk(self, total=None):
        """Mark the start of a new category chunk (chunked path only).
        `total` is the number of chunks planned for the row."""
        with self._lock:
            if total is not None:
                self._chunk_total = total
            if not self._chunked:
                self._chunked = True
                if self._segments == [[]]:
                    return
            self._segments.append([])

    def record(self, reply):
        with self._lock:
            self.attempts += 1
            self._segments[-1].append(reply)

    def _wrap(self, fn):
        def wrapped(*args, **kwargs):
            try:
                out = fn(*args, **kwargs)
            except BaseException:
                self.record(None)
                raise
            reply = out[0] if isinstance(out, tuple) and out else out
            self.record(reply)
            return out
        return wrapped

    def wrap_client(self, client):
        """Route client.complete() through the recorder (instance attribute,
        so the class and every other client are untouched)."""
        client.complete = self._wrap(client.complete)
        return client

    def wrap_fn(self, fn):
        """Wrap a function that returns (reply, error) like client.complete."""
        return self._wrap(fn)

    @property
    def raw_reply(self):
        with self._lock:
            if not self._chunked:
                seg = self._segments[0]
                return seg[-1] if seg else None
            n = self._chunk_total or len(self._segments)
            parts = []
            for i, seg in enumerate(self._segments, 1):
                last = seg[-1] if seg else None
                parts.append(_chunk_header(i, n))
                parts.append("" if last is None else str(last))
            return "\n".join(parts)


class RawRowStore:
    """Thread-safe store of raw-reply records keyed by (row index, model)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.rows = {}

    def put(self, row_idx, model_name, raw_reply, attempts, parsed_json, error,
            accumulate_attempts=True):
        key = (row_idx, model_name)
        with self._lock:
            prev = self.rows.get(key)
            if accumulate_attempts and prev is not None:
                attempts = attempts + (prev["raw_attempts"] or 0)
            self.rows[key] = {
                "raw_reply": raw_reply,
                "raw_attempts": attempts,
                "parsed_json": parsed_json,
                "raw_error": error,
            }

    def put_recorder(self, row_idx, model_name, recorder, parsed_json, error):
        self.put(row_idx, model_name, recorder.raw_reply, recorder.attempts,
                 parsed_json, error)


def attach_raw_columns(df, store, model_names):
    """Append the raw-reply columns to df (in place order: after existing
    columns). Unsuffixed for a single model, `<col>_<model>` otherwise."""
    rows = store.rows if isinstance(store, RawRowStore) else store
    single = len(model_names) == 1
    n = len(df)
    for model_name in model_names:
        for col in RAW_COLUMNS:
            values = []
            for row_idx in range(n):
                rec = rows.get((row_idx, model_name))
                if rec is None:
                    values.append(0 if col == "raw_attempts" else None)
                else:
                    values.append(rec[col])
            name = col if single else f"{col}_{model_name}"
            df[name] = values
    return df
