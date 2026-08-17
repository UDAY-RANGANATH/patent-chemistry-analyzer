"""Generate a synthetic chemistry patent PDF for testing the pipeline.

Usage: python generate_sample_patent.py [output.pdf] [--pages N]

Creates a realistic multi-page patent with: bibliographic header, abstract,
detailed description, examples (reactions), and claims. Used by the test suite
and as example_data. Purely synthetic — no real patent text.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz  # PyMuPDF

PAGE_TEXT: list[list[str]] = [
    # Page 1 — bibliographic + abstract
    [
        "UNITED STATES PATENT AND TRADEMARK OFFICE",
        "Patent No.: US 12,345,678 B2",
        "Title: PROCESS FOR THE PREPARATION OF METHYL 4-HYDROXYBENZOATE AND INTERMEDIATES THEREOF",
        "Assignee: Paracelsus Pharmaceuticals, Inc.",
        "Inventors: Alan T. Berzelius; Clara M. Hofmann",
        "Filed: March 4, 2023",
        "ABSTRACT",
        "Disclosed is an improved process for preparing methyl 4-hydroxybenzoate (paraben) from "
        "4-hydroxybenzoic acid by acid-catalyzed esterification in methanol, and a two-stage "
        "manufacturing route via an isolated intermediate.",
    ],
    # Page 2 — background + summary
    [
        "BACKGROUND OF THE INVENTION",
        "Parabens are widely used preservatives in pharmaceutical formulations. Known processes "
        "employ sulfuric acid or p-toluenesulfonic acid as catalysts with prolonged heating.",
        "SUMMARY OF THE INVENTION",
        "The present invention provides a process comprising: esterifying 4-hydroxybenzoic acid "
        "with methanol in the presence of sulfuric acid to give methyl 4-hydroxybenzoate; and "
        "purifying the product by distillation. Optionally, the ester is hydrolyzed to the "
        "sodium salt.",
    ],
    # Page 3 — detailed description + example 1
    [
        "DETAILED DESCRIPTION",
        "In a preferred embodiment, 4-hydroxybenzoic acid (CAS 99-96-7) is suspended in methanol. "
        "Concentrated sulfuric acid is added dropwise at 25 °C. The mixture is heated at reflux "
        "(about 65 °C) for 3 hours.",
        "EXAMPLE 1",
        "Preparation of methyl 4-hydroxybenzoate. 4-hydroxybenzoic acid (13.8 g, 100 mmol) and "
        "methanol (50 mL) were charged to a round-bottom flask equipped with a reflux condenser. "
        "Concentrated sulfuric acid (2 mL) was added dropwise at 25 °C. The mixture was heated at "
        "reflux for 3 hours under nitrogen. After cooling to 20 °C, the mixture was poured into "
        "water (200 mL) and extracted with ethyl acetate (3 × 100 mL). The combined organic layer "
        "was washed with saturated sodium bicarbonate solution, then brine, dried over anhydrous "
        "sodium sulfate, filtered and concentrated under reduced pressure. The residue was "
        "crystallized from hexane to give methyl 4-hydroxybenzoate as a white solid (14.2 g, "
        "93% yield). 1H NMR (CDCl3) delta 8.0 (d, 2H), 6.9 (d, 2H), 5.5 (s, 1H), 3.9 (s, 3H).",
    ],
    # Page 4 — example 2
    [
        "EXAMPLE 2",
        "Preparation of sodium 4-hydroxybenzoate. Methyl 4-hydroxybenzoate (7.6 g, 50 mmol) was "
        "dissolved in ethanol (30 mL). A solution of sodium hydroxide (2.2 g, 55 mmol) in water "
        "(10 mL) was added. The mixture was stirred at 60 °C for 1 hour. The solvent was removed "
        "under reduced pressure and the residue was crystallized from isopropanol to afford "
        "sodium 4-hydroxybenzoate as a white powder (8.0 g, quantitative).",
    ],
    # Page 5 — manufacturing process + claims
    [
        "INDUSTRIAL MANUFACTURING PROCESS",
        "At industrial scale, the esterification is carried out in a glass-lined reactor of 1000 L. "
        "The reactor is charged with 4-hydroxybenzoic acid and methanol, sulfuric acid is metered in "
        "over 30 minutes, and the batch is heated by a steam jacket to 65 °C for 3 hours. After "
        "cooling, the batch is transferred to a centrifuge for separation, then the organic phase is "
        "washed and dried in a vacuum tray dryer at 45 °C for 12 hours.",
        "WHAT IS CLAIMED IS:",
        "1. A process for preparing methyl 4-hydroxybenzoate comprising esterifying "
        "4-hydroxybenzoic acid with methanol in the presence of a strong acid catalyst.",
        "2. The process of claim 1, wherein the strong acid is sulfuric acid.",
        "3. The process of claim 1, further comprising purifying by crystallization.",
    ],
]


def build_pdf(pages: list[list[str]], out_path: Path) -> None:
    doc = fitz.open()
    for block in pages:
        page = doc.new_page(width=612, height=792)  # US Letter
        y = 60
        for line in block:
            fontsize = 13 if line.isupper() else 10.5
            page.insert_text((60, y), line, fontsize=fontsize, fontname="helv")
            y += 20
    doc.save(str(out_path))
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="sample_patent.pdf")
    parser.add_argument("--pages", type=int, default=5)
    args = parser.parse_args()

    pages = PAGE_TEXT[: args.pages]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(pages, out)
    print(f"wrote {len(pages)}-page sample patent to {out}")


if __name__ == "__main__":
    main()
