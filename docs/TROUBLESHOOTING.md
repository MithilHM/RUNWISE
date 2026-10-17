# RUNWISE Troubleshooting & Quick Fixes

Quick-reference runbook for common local setup and integration issues.

---

### 1. FalkorDB Connection Fails / Port 6379 Already in Use
**Symptom:** `Cannot connect to FalkorDB on 127.0.0.1:6379` or `port is already allocated`.  
**Cause:** Another local Redis or FalkorDB instance is already bound to port 6379.  
**Fix:**
```powershell
# Check what is using port 6379 on Windows
netstat -ano | findstr :6379

# Stop existing containers or restart Docker Compose
docker compose down
docker compose up -d
```

---

### 2. Reset / Wipe Graph State
**Symptom:** Ingestion fails due to previous duplicate test data or orphaned review items.  
**Fix:**
```powershell
# Reset graph and seed fresh 50 runs in one step:
python scripts/reset_graph.py --seed

# Nuclear wipe (cleans persistent Docker volume):
docker compose down -v
docker compose up -d
python scripts/seed_demo.py
```

---

### 3. Frontend Cannot Reach Backend (CORS / Network Error)
**Symptom:** Browser console reports `Access-Control-Allow-Origin` or `NetworkError when attempting to fetch resource`.  
**Fix:**
1. Check that backend is running on `http://127.0.0.1:8000`.
2. Ensure your `.env` contains:
   ```ini
   CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
   ```
3. Restart FastAPI backend to reload `.env` settings.

---

### 4. Zero-Key Deterministic Mode Check
**Symptom:** Wondering if an OpenAI / Anthropic API key is needed.  
**Answer:** **No.** Keep `RUNWISE_LLM_API_KEY=""` in your `.env`. RUNWISE is designed to operate deterministically using graph query templates and grounded explanations without third-party API keys.
