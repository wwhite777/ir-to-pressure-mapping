#!/usr/bin/env python3
"""
09_create_drawio_with_images.py - Create DrawIO files with embedded PNG images

This script embeds actual experimental PNG images into DrawIO XML format
so they can be edited at https://app.diagrams.net/
"""

import base64
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/wjeong/cc")
FIGURE_DIR = PROJECT_ROOT / "result" / "figure"
DRAWIO_DIR = FIGURE_DIR / "drawio"
DRAWIO_DIR.mkdir(parents=True, exist_ok=True)


def png_to_base64(png_path):
    """Convert PNG file to base64 string."""
    with open(png_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_drawio_with_embedded_image(image_path, output_path, title, width=800, height=600):
    """Create a DrawIO file with an embedded PNG image."""

    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found")
        return False

    base64_data = png_to_base64(image_path)

    drawio_xml = f'''<mxfile host="app.diagrams.net" modified="{datetime.now().isoformat()}" agent="Claude" version="21.0.0">
  <diagram name="{title}" id="embedded-image">
    <mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{width + 100}" pageHeight="{height + 150}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="title-cell" value="{title}" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=14;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="50" y="20" width="{width}" height="30" as="geometry"/>
        </mxCell>
        <mxCell id="image-cell" value="" style="shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;image=data:image/png,{base64_data};" vertex="1" parent="1">
          <mxGeometry x="50" y="60" width="{width}" height="{height}" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    with open(output_path, 'w') as f:
        f.write(drawio_xml)

    print(f"Created: {output_path}")
    return True


def create_figure2_drawio():
    """Create Figure 2 - Adaptive EMA Analysis with embedded image."""
    image_path = FIGURE_DIR / "exp2_uncertainty_ema" / "adaptive_ema_analysis.png"
    output_path = DRAWIO_DIR / "figure2_adaptive_ema.drawio"
    return create_drawio_with_embedded_image(
        image_path, output_path,
        "Figure 2: Uncertainty-Guided Adaptive EMA Analysis",
        width=900, height=700
    )


def create_figure3_drawio():
    """Create Figure 3 - Qualitative Results with embedded image."""
    image_path = FIGURE_DIR / "exp4_qualitative" / "qualitative_results.png"
    output_path = DRAWIO_DIR / "figure3_qualitative.drawio"
    return create_drawio_with_embedded_image(
        image_path, output_path,
        "Figure 3: Qualitative Results Across Occlusion Conditions",
        width=900, height=700
    )


def create_figure4_drawio():
    """Create Figure 4 - Retrieval Visualization with embedded image."""
    image_path = FIGURE_DIR / "exp3_retrieval" / "retrieval_visualization.png"
    output_path = DRAWIO_DIR / "figure4_retrieval.drawio"
    return create_drawio_with_embedded_image(
        image_path, output_path,
        "Figure 4: Cross-Modal Retrieval Visualization",
        width=1000, height=600
    )


def create_figure5_drawio():
    """Create Figure 5 - Baseline Comparison with embedded image."""
    image_path = FIGURE_DIR / "consolidated" / "baseline_comparison.png"
    output_path = DRAWIO_DIR / "figure5_baselines.drawio"
    return create_drawio_with_embedded_image(
        image_path, output_path,
        "Figure 5: Baseline Architecture Comparison",
        width=900, height=350
    )


def main():
    print("Creating DrawIO files with embedded PNG images...")
    print(f"Output directory: {DRAWIO_DIR}")
    print()

    # Create all figures
    create_figure2_drawio()
    create_figure3_drawio()
    create_figure4_drawio()
    create_figure5_drawio()

    print()
    print("Done! Open these .drawio files at https://app.diagrams.net/")
    print("You can add annotations, labels, arrows, etc. to the embedded images.")


if __name__ == "__main__":
    main()
