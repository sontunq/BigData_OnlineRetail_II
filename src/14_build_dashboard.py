"""Build the self-contained HTML dashboard.

Inlines plotly.js (from the installed plotly Python package, no CDN needed) and the JSON
exported by 13_export_dashboard.py into src/dashboard_template.html, and writes
outputs/dashboard/retail_dashboard.html - a single file that opens offline in any browser.
"""
import json
import os

import plotly

from common import ROOT

SRC = ROOT / "src" / "dashboard_template.html"
DATA = ROOT / "outputs" / "dashboard" / "dashboard_data.json"
OUT = ROOT / "outputs" / "dashboard" / "retail_dashboard.html"


def main():
    js_path = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
    plotly_js = open(js_path, encoding="utf-8").read()
    data = DATA.read_text(encoding="utf-8")
    json.loads(data)   # fail early on a broken export
    # never let a literal "</script" inside the inlined code close the <script> element
    plotly_js = plotly_js.replace("</script", "<\\/script")
    data = data.replace("</", "<\\/")
    html = SRC.read_text(encoding="utf-8")
    assert html.count("/*__PLOTLY__*/") == 1 and html.count("/*__DATA__*/") == 1
    html = html.replace("/*__PLOTLY__*/", plotly_js).replace("/*__DATA__*/", data)
    OUT.write_text(html, encoding="utf-8")
    js_version = plotly_js[:200].split("plotly.js v")[1].split()[0] if "plotly.js v" in plotly_js[:200] else "?"
    print(f"plotly.js {js_version} (from Python plotly {plotly.__version__}); "
          f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
