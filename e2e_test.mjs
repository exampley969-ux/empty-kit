import { chromium } from "/home/claude/.npm-global/lib/node_modules/playwright/index.mjs";
import { spawn, execSync } from "node:child_process";

// fuser/lsof/ss/netstat are all unavailable in this sandbox, so find our own
// leftover processes by command line instead of by port.
function killLeftovers() {
  try {
    const out = execSync("ps -eo pid,cmd").toString();
    for (const line of out.split("\n")) {
      if (line.includes("e2e_backend.py") || line.includes("http.server 8766")) {
        const pid = parseInt(line.trim().split(/\s+/)[0], 10);
        if (pid && pid !== process.pid) { try { process.kill(pid, "SIGKILL"); } catch {} }
      }
    }
  } catch {}
}
killLeftovers();
try { execSync("rm -f /tmp/e2e_frontend/vr.sqlite"); } catch {}   // every run starts from a clean database

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

function startBackend() {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", ["/home/claude/e2e_backend.py"], { stdio: ["ignore", "pipe", "pipe"] });
    let buf = "";
    proc.stdout.on("data", (d) => {
      buf += d.toString();
      const m = buf.match(/READY (\d+)/);
      if (m) resolve({ proc, port: Number(m[1]) });
    });
    proc.stderr.on("data", (d) => process.stderr.write("[backend] " + d));
    proc.on("exit", (code) => { if (code !== 0) reject(new Error("backend exited early: " + code)); });
    setTimeout(() => reject(new Error("backend did not become ready in time")), 20000);
  });
}

const results = [];
function check(name, cond, detail = "") {
  results.push({ name, ok: !!cond, detail });
  console.log((cond ? "PASS " : "FAIL ") + name + (detail ? "  -- " + detail : ""));
}

function startStaticServer() {
  return new Promise((resolve, reject) => {
    const proc = spawn("python3", ["-m", "http.server", "8766", "--bind", "127.0.0.1"],
      { cwd: "/home/claude/vision_radar/frontend", stdio: ["ignore", "pipe", "pipe"] });
    let ready = false;
    const onData = () => { if (!ready) { ready = true; resolve(proc); } };
    proc.stdout.on("data", onData);
    proc.stderr.on("data", onData);   // http.server logs requests to stderr
    setTimeout(() => { if (!ready) { ready = true; resolve(proc); } }, 1000);
    proc.on("exit", (code) => { if (!ready) reject(new Error("static server exited early: " + code)); });
  });
}

async function main() {
  let proc, staticProc, browser;
  try {
    const started = await startBackend();
    proc = started.proc;
    const base = `http://127.0.0.1:${started.port}`;
    console.log("backend ready on", base);
    staticProc = await startStaticServer();
    console.log("static frontend server ready on http://127.0.0.1:8766");

    browser = await chromium.launch();
    await runChecks(browser, base);
  } finally {
    if (browser) await browser.close().catch(() => {});
    if (proc) proc.kill(9);
    if (staticProc) staticProc.kill(9);
    killLeftovers();
  }

  const failed = results.filter(r => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed.`);
  process.exit(failed.length ? 1 : 0);
}

async function runChecks(browser, base) {
  const page = await browser.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  page.on("console", (msg) => {
    console.log("[console]", msg.type(), msg.text());
    // "Failed to load resource" lines are Chromium's own network-level
    // diagnostic for any non-2xx response (here: the EXPECTED, explicitly
    // caught 404 from refreshLiveFrame polling before the pipeline's first
    // frame exists) - never a JS exception, so they don't belong in a
    // check for uncaught *script* errors. A real uncaught error would
    // either come through page.on("pageerror") or be a console.error call
    // with its own message, not this generic network-failure line.
    if (msg.type() === "error" && !msg.text().includes("Failed to load resource")) pageErrors.push(msg.text());
  });
  page.on("requestfailed", (r) => console.log("[reqfail]", r.url(), r.failure()?.errorText));
  page.on("response", async (r) => { if (!r.ok()) { let body = ""; try { body = await r.text(); } catch {} console.log("[http]", r.status(), r.url(), body); } });

  await page.goto(`http://127.0.0.1:8766/index.html?api=${encodeURIComponent(base)}`);

  // ---- login ----
  await page.fill("#login-user", "root");
  await page.fill("#login-pass", "bootstrap-pw-1");
  await page.click("#login-btn");
  await page.waitForSelector("header", { timeout: 5000 });
  check("login succeeds and shows the dashboard shell", await page.$("header") !== null);
  check("username shown in header", (await page.textContent("header .who")).includes("root"));

  // ---- overview / status ----
  await page.waitForTimeout(500);
  const statusText = await page.textContent("#status-body");
  check("status panel shows all five Section-26 components",
    ["camera", "vision", "tracking", "events", "database"].every(c => statusText.includes(c)));
  check("radar SVG is present (Section 16)", await page.$("svg#radar") !== null);

  // ---- cameras: add one pointing at the real video ----
  await page.click("nav >> text=Cameras");
  await page.fill("#cam-name", "Garage");
  await page.fill("#cam-uri", "/tmp/e2e_frontend/pan.mp4");
  await page.fill("#cam-w", "640");
  await page.fill("#cam-h", "500");
  await page.click("#cam-add");
  await page.waitForTimeout(300);
  check("camera appears in the table", (await page.textContent("table")).includes("Garage"));

  // ---- zones: draw a real polygon by clicking the canvas (Section 8) ----
  await page.click("nav >> text=Zones");
  await page.fill("#zone-name", "Restricted-Left");
  const camOptions = await page.$$eval("#zone-camera option", opts => opts.map(o => ({ value: o.value, text: o.textContent })));
  const garageOpt = camOptions.find(o => o.text.includes("Garage"));
  check("camera is selectable in the zone editor", !!garageOpt, JSON.stringify(camOptions));
  if (garageOpt) await page.selectOption("#zone-camera", garageOpt.value);
  const svgBox = await page.locator("#zone-canvas").boundingBox();
  const clickAt = async (fx, fy) => page.mouse.click(svgBox.x + svgBox.width * fx, svgBox.y + svgBox.height * fy);
  // Left third of the real 640x500 frame - exactly where the real motorcycle
  // ends up, matching the geometry already confirmed against the raw API.
  await clickAt(0.02, 0.02); await clickAt(0.35, 0.02); await clickAt(0.35, 0.98); await clickAt(0.02, 0.98);
  const pointCount = await page.textContent("#zone-point-count");
  check("four points were placed on the canvas", pointCount.includes("4 point"), pointCount);
  await page.click("#zone-save");
  await page.waitForTimeout(300);
  check("zone appears in the zones table", (await page.textContent("table")).includes("Restricted-Left"));

  // ---- rules: create the default draft rule (motorcycle zone entry) ----
  await page.click("nav >> text=Rules");
  const ruleJson = JSON.stringify({
    name: "Vehicle enters restricted zone",
    when: { event_type: "ZONE_ENTRY", entity_type: "motorcycle" },
    if: [{ field: "confidence", operator: ">", value: 0.5 }],
  });
  await page.fill("#rule-json", ruleJson);
  await page.click("#rule-save");
  await page.waitForTimeout(300);
  check("rule appears in the rules table", (await page.textContent("table")).includes("Vehicle enters restricted zone"));

  // ---- start a real session via the UI ----
  await page.click("nav >> text=Cameras");
  await page.click("text=Start session");
  await page.waitForTimeout(300);
  const startFlash = await page.textContent("body");
  check("starting the session produced no error banner", !startFlash.includes("CAMERA OFFLINE"));

  // wait for the real pipeline (13 frames) to finish and persist results
  let sawEntity = false, sawAlert = false;
  for (let i = 0; i < 20; i++) {
    await wait(500);
    await page.click("nav >> text=Overview");
    const overview = await page.textContent("#main");
    if (overview.includes("motorcycle")) sawEntity = true;
    await page.click("nav >> text=Alerts");
    const alertsText = await page.textContent("#main");
    if (alertsText.includes("Vehicle enters restricted zone")) { sawAlert = true; break; }
  }
  check("a real motorcycle entity showed up in the running dashboard", sawEntity);
  check("the real rule produced a real alert, shown in the Alerts tab", sawAlert);

  // ---- alert -> event detail -> real evidence image actually loads ----
  await page.click("nav >> text=Alerts");
  await page.click("text=View event");
  await page.waitForTimeout(400);
  const img = page.locator("#evidence-img");
  check("clicking an alert opens its event with a real evidence image", await img.count() > 0);
  if (await img.count() > 0) {
    const natural = await img.evaluate(el => ({ w: el.naturalWidth, h: el.naturalHeight, complete: el.complete }));
    check("the evidence image actually decoded (not a broken image)", natural.complete && natural.w > 0,
      JSON.stringify(natural));
  }

  // ---- event history filters ----
  await page.click("nav >> text=Event History");
  await page.selectOption("#f-type", "ZONE_ENTRY");
  await page.click("#f-apply");
  await page.waitForTimeout(300);
  const filteredRows = await page.$$eval("table tbody tr", rows => rows.map(r => r.textContent));
  check("filtering by ZONE_ENTRY shows only zone-entry rows",
    filteredRows.length > 0 && filteredRows.every(r => r.includes("ZONE_ENTRY")), JSON.stringify(filteredRows));

  // ---- natural-language search, grounded answer, honest about what's ignored ----
  await page.click("nav >> text=Search");
  await page.fill("#search-q", "did anybody loiter suspiciously near the gate today");
  await page.click("#search-go");
  await page.waitForTimeout(400);
  const searchText = await page.textContent("#search-results");
  check("search honestly reports unrecognised words as ignored, not hallucinated",
    searchText.includes("Not understood"));
  await page.fill("#search-q", "what happened in Restricted-Left today");
  await page.click("#search-go");
  await page.waitForTimeout(400);
  const zoneSearch = await page.textContent("#search-results");
  check("a real, understood zone query returns a real result grounded in stored events",
    zoneSearch.includes("ZONE_ENTRY") && !zoneSearch.includes("Not understood"));

  // ---- live camera view (Section 17-B), using a long clip so the real
  // pipeline is still running when we look, instead of racing a clip short
  // enough to finish before any poll tick observes it ----
  await page.click("nav >> text=Cameras");
  await page.fill("#cam-name", "LongCam");
  await page.fill("#cam-uri", "/tmp/e2e_frontend/pan_long.mp4");
  await page.click("#cam-add");
  await page.waitForTimeout(300);
  await page.locator("tr", { hasText: "LongCam" }).getByText("Start session").click();
  await page.waitForTimeout(300);

  await page.click("nav >> text=Overview");
  let sawLiveFrame = false;
  // 208 real frames through real YOLO inference take several seconds, but
  // model load / thread startup time varies - give this a generous, not
  // tightly-timed, window rather than treating normal variance as a bug.
  for (let i = 0; i < 40; i++) {
    const img = page.locator("#live-camera-img");
    if (await img.count() > 0) {
      const natural = await img.evaluate(el => ({ w: el.naturalWidth, complete: el.complete, visible: el.offsetParent !== null }));
      if (natural.visible && natural.complete && natural.w > 0) { sawLiveFrame = true; break; }
    }
    await wait(300);
    await page.click("nav >> text=Overview");   // re-trigger the view's eager refresh
  }
  check("the live camera panel shows a real decoded frame while a session is actually running", sawLiveFrame);

  // stop it the same way any user would: via the UI
  await page.click("nav >> text=Cameras");
  const stopBtns = page.locator("button", { hasText: "Stop" });
  if (await stopBtns.count() > 0) await stopBtns.first().click();
  await page.waitForTimeout(300);

  // ---- admin: users tab ----
  await page.click("nav >> text=Users");
  await page.fill("#u-name", "newviewer");
  await page.fill("#u-pass", "longenough-pw-1");
  await page.selectOption("#u-role", "viewer");
  await page.click("#u-add");
  await page.waitForTimeout(300);
  check("new user appears in the users table", (await page.textContent("table")).includes("newviewer"));

  // ---- role separation: log in as the new viewer, confirm no Users tab / write controls ----
  await page.click("text=log out");
  await page.waitForSelector("#login-user");
  await page.fill("#login-user", "newviewer");
  await page.fill("#login-pass", "longenough-pw-1");
  await page.click("#login-btn");
  await page.waitForSelector("header", { timeout: 5000 });
  const navText = await page.textContent("nav");
  check("a viewer does not see the Users tab (Section 28 role separation)", !navText.includes("Users"));
  await page.click("nav >> text=Cameras");
  check("a viewer sees no camera-add form", await page.$("#cam-add") === null);

  check("no uncaught JS errors occurred during the whole run", pageErrors.length === 0, pageErrors.join(" | "));
}

main().catch(e => { console.error("E2E RUN FAILED:", e); process.exit(1); });
