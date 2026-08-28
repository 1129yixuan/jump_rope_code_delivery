from pathlib import Path


ROOT = Path(__file__).resolve().parent

LEGACY_DATA_TRANSLATIONS = {
    "../\u8f6e\u6362\u8df3/\u0032\u53f7\u573a\u5730_\u4e09\u7248\u7b97\u6cd5\u51c6\u786e\u7387.xlsx": "../jump_rope/venue_2_three_algorithm_accuracy.xlsx",
    "\u4ec5\u0032\u53f7\u573a\u5730\uff1b\u0031\u53f7\u573a\u5730\u4e3a\u8bad\u7ec3\u6570\u636e\uff0c\u4e0d\u7eb3\u5165\u8bc4\u4f30": "Venue 2 only; Venue 1 is training data and is excluded from evaluation",
    "\u0032\u53f7\u573a\u5730": "Venue 2",
    "\u4f4e\u7b49 0-100": "Low 0-100",
    "\u4e2d\u7b49 101-129": "Intermediate 101-129",
    "\u9ad8\u7b49 130+": "High 130+",
}


def load_data() -> str:
    data_path = ROOT / "data.js"
    if data_path.is_file():
        data = data_path.read_text(encoding="utf-8")
    else:
        public_path = ROOT / "public_dashboard.html"
        public = public_path.read_text(encoding="utf-8")
        start = public.index("window.DASHBOARD_DATA =")
        end = public.index("</script>", start)
        data = public[start:end].strip()

    for legacy, english in LEGACY_DATA_TRANSLATIONS.items():
        data = data.replace(legacy, english)
    return data


def main():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    styles = (ROOT / "styles.css").read_text(encoding="utf-8")
    data = load_data()
    app = (ROOT / "app.js").read_text(encoding="utf-8")

    html = html.replace(
        '<link rel="stylesheet" href="./styles.css" />',
        f"<style>\n{styles}\n</style>",
    )
    html = html.replace(
        '    <script src="./data.js"></script>',
        f"    <script>\n{data}\n    </script>",
    )
    html = html.replace(
        '    <script src="./app.js"></script>',
        f"    <script>\n{app}\n    </script>",
    )
    output = ROOT / "public_dashboard.html"
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
