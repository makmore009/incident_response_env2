"""FastAPI server for the Incident Response Environment."""

import os
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import IncidentAction, IncidentObservation
    from .incident_environment import IncidentEnvironment
    from .scenarios import list_tasks as get_available_tasks
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import IncidentAction, IncidentObservation
    from server.incident_environment import IncidentEnvironment
    from server.scenarios import list_tasks as get_available_tasks

app = create_app(
    IncidentEnvironment,
    IncidentAction,
    IncidentObservation,
    env_name="incident_env",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)


@app.get("/api_info")
async def root():
    """API information page."""
    return {
        "name": "Incident Response Environment",
        "description": "On-call incident response RL environment for training AI agents",
        "endpoints": {
            "POST /reset": "Reset environment (accepts task_name parameter)",
            "POST /step": "Execute an action (accepts IncidentAction)",
            "GET /state": "Get current environment state",
            "GET /schema": "Get action/observation JSON schemas",
            "GET /tasks": "List available tasks",
            "GET /ui": "Minimal browser UI for manual testing",
            "WS /ws": "WebSocket endpoint for persistent sessions",
        },
        "tasks": get_available_tasks(),
    }


@app.get("/")
async def index():
    """Root endpoint for Spaces App tab: redirect to human UI."""
    return RedirectResponse(url="/ui", status_code=307)


@app.get("/tasks")
async def tasks():
    """Return available task names for this environment."""
    return {"tasks": get_available_tasks()}


@app.get("/healthz")
async def healthz():
    """Quick health check."""
    try:
        env = IncidentEnvironment()
        obs = env.reset()
        return {"status": "ok", "alert": obs.alert_summary}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
async def health():
    """Compatibility health endpoint used by container healthcheck."""
    return await healthz()


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Minimal browser UI for manual review and demo flows."""
    return """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Incident Env UI</title>
    <style>
        :root {
            --bg-a: #f5f7ff;
            --bg-b: #edf7f3;
            --ink: #1d2433;
            --muted: #5f6b82;
            --accent: #0f766e;
            --accent-2: #1d4ed8;
            --card: #ffffff;
            --border: #d9e0ef;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            font-family: "Segoe UI", "Trebuchet MS", sans-serif;
            background: radial-gradient(circle at 15% 20%, var(--bg-a), transparent 45%),
                                    radial-gradient(circle at 85% 10%, #e4f2ff, transparent 40%),
                                    linear-gradient(135deg, #f7fafc, var(--bg-b));
            min-height: 100vh;
        }
        .wrap { max-width: 1100px; margin: 0 auto; padding: 28px 16px 40px; }
        h1 { margin: 0 0 6px; font-weight: 700; letter-spacing: 0.2px; }
        .sub { margin: 0 0 18px; color: var(--muted); }
        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 14px; }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 8px 24px rgba(20, 30, 55, 0.05);
        }
        label { display: block; font-size: 13px; color: var(--muted); margin: 8px 0 5px; }
        select, input, textarea, button {
            width: 100%;
            border: 1px solid #cfd8ea;
            border-radius: 10px;
            padding: 9px 10px;
            font: inherit;
            background: #fff;
            color: var(--ink);
        }
        textarea { min-height: 110px; resize: vertical; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .btn {
            cursor: pointer;
            border: none;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #fff;
            font-weight: 600;
            margin-top: 10px;
        }
        .btn:hover { filter: brightness(1.03); }
        pre {
            margin: 0;
            background: #111827;
            color: #d1e3ff;
            border-radius: 12px;
            padding: 12px;
            overflow: auto;
            min-height: 420px;
            max-height: 68vh;
            border: 1px solid #202b3c;
            font-size: 12px;
            line-height: 1.45;
        }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
            pre { min-height: 280px; }
        }
    </style>
</head>
<body>
    <div class=\"wrap\">
        <h1>Incident Response Environment</h1>
        <p class=\"sub\">Manual reviewer UI for reset/step/state API flow.</p>
        <div class=\"grid\">
            <div class=\"card\">
                <label>Task</label>
                <select id=\"task\">
                    <option value=\"easy_config_error\">easy_config_error</option>
                    <option value=\"medium_cascading_db\">medium_cascading_db</option>
                    <option value=\"hard_intermittent_auth\">hard_intermittent_auth</option>
                </select>
                <button class=\"btn\" onclick=\"doReset()\">Reset Episode</button>

                <hr style=\"border:none;border-top:1px solid #e6ebf7;margin:14px 0;\" />

                <label>Action Type</label>
                <select id=\"actionType\">
                    <option>query_logs</option>
                    <option>check_metrics</option>
                    <option>read_runbook</option>
                    <option>identify_root_cause</option>
                    <option>execute_remedy</option>
                    <option>escalate</option>
                    <option>get_status</option>
                    <option>communicate_status</option>
                    <option>noop</option>
                </select>

                <div class=\"row\">
                    <div>
                        <label>Target</label>
                        <input id=\"target\" placeholder=\"payment-service\" />
                    </div>
                    <div>
                        <label>Key=Value params</label>
                        <input id=\"kv\" placeholder=\"service=payment-service,remedy=rollback_config\" />
                    </div>
                </div>

                <button class=\"btn\" onclick=\"doStep()\">Send Step</button>
                <button class=\"btn\" style=\"margin-top:8px\" onclick=\"getState()\">Get State</button>
            </div>

            <div class=\"card\">
                <pre id=\"out\">Ready. Click Reset Episode.</pre>
            </div>
        </div>
    </div>

    <script>
        const out = document.getElementById('out');

        function parseParams(src) {
            const result = {};
            if (!src.trim()) return result;
            src.split(',').forEach(part => {
                const [k, ...rest] = part.split('=');
                if (!k) return;
                result[k.trim()] = rest.join('=').trim();
            });
            return result;
        }

        function write(label, data) {
            out.textContent = `${label}\n${JSON.stringify(data, null, 2)}`;
        }

        async function doReset() {
            const task = document.getElementById('task').value;
            const r = await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_name: task })
            });
            write('POST /reset', await r.json());
        }

        async function doStep() {
            const action_type = document.getElementById('actionType').value;
            const target = document.getElementById('target').value;
            const parameters = parseParams(document.getElementById('kv').value);
            const r = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action_type, target, parameters })
            });
            write('POST /step', await r.json());
        }

        async function getState() {
            const r = await fetch('/state');
            write('GET /state', await r.json());
        }
    </script>
</body>
</html>
"""


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
