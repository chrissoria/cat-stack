"""
Browser-based review UI for CatLLM classification corrections.

Opens a local web page where users can toggle category checkboxes for each
classified item. Submissions are posted back to a temporary local server.
Uses only Python standard library — no extra dependencies.
"""

import html
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs


def open_review_ui(items, categories):
    """
    Launch a browser-based review UI and block until the user submits.

    Args:
        items: list of dicts, each with:
            - "input": str — the input text
            - "values": dict — {category_name: 0 or 1}
        categories: list of str — category names (for column ordering)

    Returns:
        list of dicts, each with:
            - "input": str
            - "original": dict — {category_name: 0/1} before corrections
            - "corrected": dict — {category_name: 0/1} after corrections
            - "changed": list of str — category names that were flipped
        Returns None if the user cancels (closes browser without submitting).
    """
    result_holder = {"data": None}
    server_ready = threading.Event()

    page_html = _build_html(items, categories)

    class ReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page_html.encode("utf-8"))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")

            try:
                submitted = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                return

            # Build corrections from submitted data
            corrections = []
            for idx, item in enumerate(items):
                original = dict(item["values"])
                corrected = {}
                changed = []

                for cat in categories:
                    key = f"item_{idx}_{cat}"
                    new_val = 1 if submitted.get(key) else 0
                    corrected[cat] = new_val
                    if new_val != original[cat]:
                        changed.append(cat)

                corrections.append({
                    "input": item["input"],
                    "original": original,
                    "corrected": corrected,
                    "changed": changed,
                })

            result_holder["data"] = corrections

            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

            # Shut down server after response is sent
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, format, *args):
            pass  # Suppress request logging

    # Find an available port
    server = HTTPServer(("127.0.0.1", 0), ReviewHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"

    print(f"[CatLLM] Opening review UI in browser: {url}")
    print("[CatLLM] Waiting for corrections... (submit the form to continue)\n")

    # Start server in background, open browser
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    webbrowser.open(url)

    # Block until server shuts down (user submitted or closed)
    server_thread.join(timeout=3600)  # 1 hour timeout
    server.server_close()

    return result_holder["data"]


def _build_html(items, categories):
    """Build the self-contained HTML page for the review UI."""

    # Build item rows
    item_rows = []
    for idx, item in enumerate(items):
        input_text = html.escape(str(item["input"]))
        checkboxes = []
        for cat in categories:
            val = item["values"].get(cat, 0)
            checked = "checked" if val == 1 else ""
            cat_escaped = html.escape(cat)
            field_name = f"item_{idx}_{cat}"
            checkboxes.append(
                f'<label class="cb-label">'
                f'<input type="checkbox" name="{html.escape(field_name)}" {checked}>'
                f' {cat_escaped}</label>'
            )

        cb_html = "\n".join(checkboxes)
        item_rows.append(f"""
        <div class="item" id="item-{idx}">
            <div class="item-header">Item {idx + 1}</div>
            <div class="item-text">{input_text}</div>
            <div class="checkboxes">
                {cb_html}
            </div>
        </div>""")

    items_html = "\n".join(item_rows)
    n_items = len(items)
    n_cats = len(categories)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CatLLM — Review Classifications</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5;
        color: #333;
        padding: 20px;
    }}
    .container {{
        max-width: 900px;
        margin: 0 auto;
    }}
    h1 {{
        font-size: 1.5rem;
        margin-bottom: 4px;
    }}
    .subtitle {{
        color: #666;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }}
    .item {{
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    .item.changed {{
        border-color: #f0ad4e;
        background: #fffdf5;
    }}
    .item-header {{
        font-weight: 600;
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 6px;
    }}
    .item-text {{
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 12px;
        padding: 10px;
        background: #fafafa;
        border-radius: 4px;
        border-left: 3px solid #ddd;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 150px;
        overflow-y: auto;
    }}
    .checkboxes {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px 16px;
    }}
    .cb-label {{
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
        border: 1px solid #e0e0e0;
        background: #fafafa;
        transition: background 0.15s, border-color 0.15s;
    }}
    .cb-label:hover {{
        background: #f0f0f0;
    }}
    .cb-label.active {{
        background: #e8f5e9;
        border-color: #4caf50;
    }}
    .cb-label.was-changed {{
        box-shadow: 0 0 0 2px #ff9800;
    }}
    input[type="checkbox"] {{
        width: 16px;
        height: 16px;
        cursor: pointer;
        accent-color: #4caf50;
    }}
    .actions {{
        position: sticky;
        bottom: 0;
        background: #f5f5f5;
        padding: 16px 0;
        border-top: 1px solid #ddd;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .stats {{
        font-size: 0.9rem;
        color: #666;
    }}
    .stats .changed-count {{
        color: #f0ad4e;
        font-weight: 600;
    }}
    button {{
        padding: 10px 32px;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.15s;
    }}
    .submit-btn {{
        background: #4caf50;
        color: white;
    }}
    .submit-btn:hover {{
        background: #43a047;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>Review Classifications</h1>
    <p class="subtitle">
        {n_items} item(s), {n_cats} categor{"y" if n_cats == 1 else "ies"}.
        Toggle checkboxes to correct, then submit.
    </p>

    <form id="review-form">
        {items_html}

        <div class="actions">
            <div class="stats">
                Changes: <span class="changed-count" id="change-count">0</span>
            </div>
            <button type="submit" class="submit-btn">Submit Corrections</button>
        </div>
    </form>
</div>

<script>
(function() {{
    const form = document.getElementById('review-form');
    const changeCount = document.getElementById('change-count');

    // Track original values
    const originals = {{}};
    form.querySelectorAll('input[type="checkbox"]').forEach(cb => {{
        originals[cb.name] = cb.checked;

        cb.addEventListener('change', () => {{
            // Update label styling
            const label = cb.closest('.cb-label');
            label.classList.toggle('active', cb.checked);
            const changed = cb.checked !== originals[cb.name];
            label.classList.toggle('was-changed', changed);

            // Update item card styling
            const item = cb.closest('.item');
            const anyChanged = Array.from(item.querySelectorAll('input[type="checkbox"]'))
                .some(c => c.checked !== originals[c.name]);
            item.classList.toggle('changed', anyChanged);

            // Update total change count
            let total = 0;
            form.querySelectorAll('input[type="checkbox"]').forEach(c => {{
                if (c.checked !== originals[c.name]) total++;
            }});
            changeCount.textContent = total;
        }});

        // Set initial active state
        const label = cb.closest('.cb-label');
        label.classList.toggle('active', cb.checked);
    }});

    form.addEventListener('submit', async (e) => {{
        e.preventDefault();

        // Collect values
        const data = {{}};
        form.querySelectorAll('input[type="checkbox"]').forEach(cb => {{
            data[cb.name] = cb.checked;
        }});

        try {{
            const resp = await fetch(window.location.href, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(data),
            }});

            if (resp.ok) {{
                document.body.innerHTML = `
                    <div style="display:flex;align-items:center;justify-content:center;
                                height:80vh;font-family:sans-serif;">
                        <div style="text-align:center;">
                            <h2 style="color:#4caf50;">Corrections submitted</h2>
                            <p style="color:#666;margin-top:8px;">
                                You can close this tab.
                            </p>
                        </div>
                    </div>`;
            }} else {{
                alert('Submission failed. Please try again.');
            }}
        }} catch (err) {{
            alert('Could not reach server: ' + err.message);
        }}
    }});
}})();
</script>
</body>
</html>"""
