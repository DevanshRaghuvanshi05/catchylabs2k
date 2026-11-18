# CatchyLabs Read-Only Demo Guide

This guide describes how to run a safe, read-only CatchyLabs demonstration on Windows
with cloudflared tunneling to a remote frontend (e.g., Replit).

---

## Prerequisites

1. Ensure a `.env` file exists in the project folder (copy from `.env.example`).
   `run_demo.ps1` loads it automatically -- you do **not** need to set variables by hand.

2. At minimum, `.env` must contain:
   ```
   DATABASE_URL=postgresql://user:password@host:5432/dbname
   CATCHYLAB_API_KEY=your-api-key
   ```
   If `CATCHYLAB_API_KEY` is missing the script will offer to generate one for you.

3. Optional -- configure CORS for your frontend by adding to `.env`:
   ```
   ALLOWED_ORIGINS=https://your-frontend-app.replit.app
   ```

---

## Script Parameters

| Parameter | Description |
|-----------|-------------|
| `-CheckOnly` | Load .env, validate settings, print OK or the exact problem, then exit without starting anything. |
| `-WithWorker` | Also start the RQ worker in a separate PowerShell window. Without this flag the worker is NOT started. |
| `-ShowKey` | Print the CATCHYLAB_API_KEY value when reporting it (omitted by default). |
| `-ExpectProject <id>` | Abort unless the DATABASE_URL host contains this Supabase project id. |

---

## Step-by-Step Demo Execution

### Step 1: Validate settings (dry run)

Use `-CheckOnly` to confirm everything is correct before starting any process:

```powershell
.\run_demo.ps1 -CheckOnly
```

To also verify the connected Supabase project:

```powershell
.\run_demo.ps1 -ExpectProject yfxttjycopwmvwyidvbi -CheckOnly
```

Expected output (all checks pass):

```
=== CatchyLabs Read-Only Demo Launcher ===
Database Host : yfxttjycopwmvwyidvbi.supabase.co
Project check : OK (host contains 'yfxttjycopwmvwyidvbi')

OK -- all checks passed. Exiting (--CheckOnly).
```

---

### Step 2: Start the Demo Server

```powershell
.\run_demo.ps1
```

To also start the RQ worker in a separate window:

```powershell
.\run_demo.ps1 -WithWorker
```

To start with project verification:

```powershell
.\run_demo.ps1 -ExpectProject yfxttjycopwmvwyidvbi
```

This script:
- Reads `.env` from the script folder and loads variables not already set in the shell.
- If `CATCHYLAB_API_KEY` is missing, prompts to generate one with Python and appends it to `.env`.
- Displays the database host name only (credentials and passwords are never shown).
- Optionally launches the RQ worker in a separate window (`-WithWorker`).
- Starts `uvicorn` on port 8000 with `DEMO_MODE=1` (all mutating POST and PATCH endpoints return 403 "Disabled in demo mode").
- Uses the project `venv` Python if a `venv` folder exists, otherwise falls back to `python` on the PATH.

---

### Step 3: Start Cloudflare Tunnel

In a new terminal window, expose port 8000 via Cloudflare:

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated public URL (e.g. `https://xxxx.trycloudflare.com`).

---

### Step 4: Verify Demo Endpoints

In another terminal, run `check_demo.py` to test the endpoints and verify counts:

```bash
python check_demo.py http://localhost:8000
```

Or test against the public tunnel:

```bash
python check_demo.py https://xxxx.trycloudflare.com
```

This verifies:
- `GET /health` (Status 200)
- `GET /demo/status` (Status 200, demo_mode=True, run count, latest run ID)
- `GET /runs` (Status 200, total run count)
- `GET /research/<run_id>/full` (Status 200, evidence, insights, gaps, opportunities counts)

---

### Step 5: Open the Frontend

Open the frontend application (e.g. Replit frontend URL configured in `ALLOWED_ORIGINS`)
and connect it to your backend tunnel URL. All research reports, evidence, gaps, and
opportunities can be viewed in read-only mode safely without modifying any production
database.
