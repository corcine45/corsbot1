Deployment / Railway guide

This file documents quick steps to deploy Corsbot on Railway (or to set environment variables locally).

Required environment variables
- GROQ_API_KEY: your Groq API key (required)
- DISCORD_TOKEN: your Discord bot token (required)
- GIPHY_API_KEY: your Giphy API key (required)
- GEMINI_API_KEY: optional Gemini API key for smarter non-casual replies
- GEMINI_MODEL: optional Gemini model override (default: gemini-2.5-flash)
- HF_TOKEN: optional Hugging Face token for faster, authenticated model downloads (recommended)

Railway web dashboard (recommended)
1. Open your Railway project.
2. Go to Settings → Environment Variables.
3. Add the keys above and their values. Do NOT commit secrets to source control.
4. Restart or redeploy the service after saving variables.

Railway CLI (optional)
```bash
railway variables set GROQ_API_KEY "your_real_groq_key_here"
railway variables set DISCORD_TOKEN "your_discord_token_here"
railway variables set GIPHY_API_KEY "your_giphy_key_here"
railway variables set GEMINI_API_KEY "your_gemini_key_here"
railway variables set HF_TOKEN "your_hf_token_here"   # optional
railway up
```

Local development (PowerShell)
```powershell
# For current shell only
$env:GROQ_API_KEY = "your_real_groq_key_here"
$env:DISCORD_TOKEN = "your_discord_token_here"
$env:GIPHY_API_KEY = "your_giphy_key_here"
$env:GEMINI_API_KEY = "your_gemini_key_here"
$env:HF_TOKEN = "your_hf_token_here"   # optional
python main.py
```

Persisting on Windows (new shells)
```powershell
setx GROQ_API_KEY "your_real_groq_key_here"
setx DISCORD_TOKEN "your_discord_token_here"
setx GIPHY_API_KEY "your_giphy_key_here"
setx GEMINI_API_KEY "your_gemini_key_here"
setx HF_TOKEN "your_hf_token_here"   # optional
```

Verify the variables
- PowerShell quick check:
```powershell
echo $env:GROQ_API_KEY
echo $env:GEMINI_API_KEY
```
- Python quick check:
```powershell
python -c "import os; print(os.getenv('GROQ_API_KEY'), os.getenv('GEMINI_API_KEY'))"
```

Notes
- `config.py` performs placeholder detection and will exit with a clear error if common placeholder values are present.
- Do not store secrets in the repository. Use Railway environment variables or a secrets manager.
- After updating vars on Railway, trigger a redeploy/restart so the new env vars are loaded.

Troubleshooting
- 401 Invalid API Key: double-check `GROQ_API_KEY` value in Railway or local env; replace any placeholder text.
- Gemini 400/401 errors: double-check `GEMINI_API_KEY`; the bot will fall back to Groq if Gemini fails.
- HF Hub rate limits / slow downloads: add `HF_TOKEN` to authenticate downloads.
- If the bot still fails to start, capture logs in Railway and check `Missing env var` or `Invalid env var` messages emitted by `config.py`.

If you'd like, I can also add a `gitignore` reminder (do not commit `.env`) or a CI check to ensure env vars are configured before deploy.
