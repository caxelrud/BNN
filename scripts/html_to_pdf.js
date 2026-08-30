// Renders a statically-exported Pluto notebook (baked-state HTML, with the
// Pluto frontend-dist assets vendored locally alongside it -- see
// scripts/html_to_pdf.sh) to PDF with a real headless Chromium instance,
// waiting for Pluto's client-side "loading" progress bar to actually finish
// before printing (a plain `chromium --print-to-pdf` fires on window.onload,
// long before Pluto has parsed/rendered the baked cell outputs).
//
// Chromium's normal paginated `format: "A4"` print layout inserts a blank
// leading page for Pluto's notebook shell (some CSS interaction with the
// editor's flex layout / fixed-position ToC -- print pagination, not
// screen layout, is affected). We sidestep the whole pagination engine by
// printing to a single page sized to the notebook's actual rendered
// height instead of paginating it.
const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

const VIEWPORT_WIDTH = 1400;

async function renderOne(browser, baseUrl, htmlFile, pdfPath) {
  const page = await browser.newPage({ viewport: { width: VIEWPORT_WIDTH, height: 1600 } });
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

  const contentHeight = await page.evaluate(() => document.documentElement.scrollHeight);

  fs.mkdirSync(path.dirname(pdfPath), { recursive: true });

  // Chromium's px->pt rounding can push a sliver of content onto a second
  // page even when height == scrollHeight exactly; pad until it settles on
  // a single page rather than risk cropping real content.
  let pad = 8;
  for (let attempt = 0; attempt < 6; attempt++) {
    await page.pdf({
      path: pdfPath,
      printBackground: true,
      width: `${VIEWPORT_WIDTH}px`,
      height: `${contentHeight + pad}px`,
    });
    if (countPdfPages(pdfPath) === 1) {
      await page.close();
      return;
    }
    pad *= 4;
  }
  await page.close();
  throw new Error(`${pdfPath}: could not settle on a single page (content height ${contentHeight})`);
}

function countPdfPages(pdfPath) {
  const buf = fs.readFileSync(pdfPath, "latin1");
  const matches = buf.match(/\/Type\s*\/Page[^s]/g);
  return matches ? matches.length : 0;
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
