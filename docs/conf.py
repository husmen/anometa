"""Sphinx configuration for the Anometa documentation."""

project = "Anometa"
author = "Houssem Menhour"
copyright = "2026, Houssem Menhour"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

# Docstrings use single backticks for code names, as in Markdown.
default_role = "code"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Attribute types come from the annotations; keep them out of the text (numpy.str_ etc.).
napoleon_attr_annotations = False

# Local-only planning files and the separately built site, report and video sources.
exclude_patterns = [
    "_build",
    "plans",
    "wayfinder",
    "HACKATHON.md",
    "video",
    "site",
    "report",
    "provenance",
]

# Mermaid: neutral "base" theme in both modes; colours come from _static/mermaid-zed.css.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
# Size diagrams to their content instead of a fixed 500px box.
mermaid_height = "auto"
mermaid_init_config = {
    "startOnLoad": False,
    "themeVariables": {
        "fontFamily": "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
        "fontSize": "14px",
    },
    "flowchart": {"curve": "basis", "padding": 14, "nodeSpacing": 40, "rankSpacing": 48},
}

html_theme = "furo"
html_title = "Anometa"
html_logo = "site/assets/logo.webp"
html_favicon = "site/assets/logo.webp"
html_static_path = ["_static"]
html_css_files = ["mermaid-zed.css"]
