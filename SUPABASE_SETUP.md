# Supabase setup (Vision Radar)

1. Supabase -> New project. Save the **database password** you choose.
2. Click **Connect** -> **Session pooler** and copy the string. It looks like
   `postgres://postgres.<ref>:[YOUR-PASSWORD]@aws-0-<region>.pooler.supabase.com:5432/postgres`
   Replace `[YOUR-PASSWORD]` with your password (URL-encode special characters: `@`->`%40`, `#`->`%23`, `/`->`%2F`).
   Use *Session pooler*, not the direct `db.<ref>.supabase.co` host (IPv6-only unless you buy the IPv4 add-on).
3. **Prove the connection for real** (from any machine with internet):
   ```
   pip install "psycopg[binary]"
   export VISION_RADAR_DATABASE_URL='postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres'
   python -m vision_radar.data_layer.supabase_check
   ```
   Expect three `OK` lines (connected, write/read/delete round-trip, Row Level Security on).
4. Render -> your service -> Environment -> add `VISION_RADAR_DATABASE_URL` (same string). Redeploy.
   Tables are created automatically on first start. (`supabase/schema.sql` is the same SQL if you prefer the SQL Editor.)
5. Uptime ping target: `https://<your-app>.onrender.com/` every ~10 min.

Notes
- Never commit the URL/password to git. If it leaks: Supabase -> Settings -> Database -> reset password.
- Evidence images are still files on the server (`/tmp/evidence`), so they vanish on restart; the event rows survive in Supabase.
- Each database write is now a network round-trip (tens of ms); the pipeline persists entities only every N frames, so this should be fine, but watch FPS.
