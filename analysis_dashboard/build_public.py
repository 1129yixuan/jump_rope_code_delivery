from pathlib import Path


ROOT = Path(__file__).resolve().parent

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
