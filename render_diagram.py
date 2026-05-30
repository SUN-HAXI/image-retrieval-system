"""Render Mermaid diagram to PNG/SVG via kroki.io API (handles large diagrams)."""
import json
import urllib.request
from pathlib import Path

KROKI = "https://kroki.io"


def render(kroki_type: str, code: str, output_path: str):
    payload = json.dumps({"diagram_source": code}).encode()
    url = f"{KROKI}/{kroki_type}"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(output_path, "wb") as f:
            f.write(resp.read())
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"  [{size_kb:.0f} KB] Saved → {output_path}")


def main():
    md_file = Path(__file__).parent / "model_architecture.md"
    code = md_file.read_text(encoding="utf-8")

    start = code.find("```mermaid")
    end = code.find("```", start + 10)
    if start == -1 or end == -1:
        print("ERROR: No mermaid block found")
        return

    mermaid_code = code[start + 10 : end].strip()
    print(f"Diagram: {len(mermaid_code)} chars, {mermaid_code.count(chr(10))} lines")

    out_svg = md_file.with_suffix(".svg")
    print("Rendering SVG via kroki.io ...")
    render("mermaid/svg", mermaid_code, str(out_svg))

    out_png = md_file.with_suffix(".png")
    print("Rendering PNG via kroki.io ...")
    render("mermaid/png", mermaid_code, str(out_png))

    print(f"\nDone! Files:\n  {out_png}\n  {out_svg}")


if __name__ == "__main__":
    main()
