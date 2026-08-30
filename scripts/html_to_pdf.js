// Renders a statically-exported Pluto notebook (baked-state HTML, with the
// Pluto frontend-dist assets vendored locally alongside it -- see
// scripts/html_to_pdf.sh) to PDF with a real headless Chromium instance,
// waiting for Pluto's client-side "loading" progress bar to actually finish
// before printing (a plain `chromium --print-to-pdf` fires on window.onload,
// long before Pluto has parsed/rendered the baked cell outputs).
const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

async function renderOne(browser, baseUrl, htmlFile, pdfPath) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 1600 } });
  const url = baseUrl + "/" + htmlFile;
  await page.goto(url, { waitUntil: "load", timeout: 120000 });

  // Pluto's static-export shell puts a "loading" class on <pluto-editor>
  // while it parses/renders the embedded notebook state, and removes it
  // once every cell output is in the DOM.
  await page.waitForFunction(() => {
    const editor = document.querySelector("pluto-editor");
    return !!editor && !editor.classList.contains("loading");
  }, null, { timeout: 120000 });

  // extra settle time for plot / katex rendering
  await page.waitForTimeout(6000);

  fs.mkdirSync(path.dirname(pdfPath), { recursive: true });
  await page.pdf({
    path: pdfPath,
    format: "A4",
    printBackground: true,
    margin: { top: "12mm", bottom: "12mm", left: "10mm", right: "10mm" },
  });
  await page.close();
}

async function main() {
  const [, , htmlDir, pdfDir, baseUrl] = process.argv;
  const files = fs.readdirSync(htmlDir).filter((f) => f.endsWith(".html")).sort();
  const browser = await chromium.launch({
    executablePath: "/opt/pw-browsers/chromium",
    args: ["--no-sandbox", "--disable-background-networking"],
  });
  for (const f of files) {
    const base = f.replace(/\.html$/, "");
    const pdfPath = path.join(pdfDir, base + ".pdf");
    console.log("Rendering", baseUrl + "/" + f, "->", pdfPath);
    await renderOne(browser, baseUrl, f, pdfPath);
  }
  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
