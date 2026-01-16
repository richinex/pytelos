"""Theme definitions for the TUI.

This module hides the design decisions about:
- Color palettes and visual appearance
- Theme variables (borders, scrollbars, etc.)
- Dark/light mode configuration

To add a new theme, define it here and register in the app.
"""

from textual.theme import Theme

# Refined Catppuccin Mocha theme with enhanced contrast and polish
CATPPUCCIN_MOCHA = Theme(
    name="catppuccin-mocha",
    # Core palette - refined for better contrast
    primary="#89b4fa",      # Blue (Sapphire) - main accent
    secondary="#cba6f7",    # Mauve/Purple - secondary accent
    accent="#f9e2af",       # Yellow/Gold - highlights
    foreground="#cdd6f4",   # Light text
    background="#11111b",   # Crust - deepest background
    success="#a6e3a1",      # Green - success states
    warning="#fab387",      # Peach - warnings
    error="#f38ba8",        # Red/Pink - errors
    surface="#1e1e2e",      # Base - main surface
    panel="#181825",        # Mantle - panel backgrounds
    dark=True,
    variables={
        # Cursor styling
        "block-cursor-foreground": "#11111b",
        "block-cursor-background": "#f5e0dc",
        "block-cursor-text-style": "bold",
        "block-cursor-blurred-foreground": "#cdd6f4",
        "block-cursor-blurred-background": "#45475a",
        "block-hover-background": "#313244 20%",

        # Input styling
        "input-cursor-background": "#cdd6f4",
        "input-cursor-foreground": "#11111b",
        "input-selection-background": "#89b4fa 30%",

        # Border colors
        "border": "#45475a",
        "border-blurred": "#313244",

        # Scrollbar styling - subtle and refined
        "scrollbar": "#313244",
        "scrollbar-hover": "#45475a",
        "scrollbar-active": "#89b4fa",
        "scrollbar-background": "#181825",
        "scrollbar-corner-color": "#181825",

        # Footer styling
        "footer-foreground": "#bac2de",
        "footer-background": "#11111b",
        "footer-key-foreground": "#f9e2af",
        "footer-key-background": "#313244",
        "footer-description-foreground": "#a6adc8",

        # Text variants
        "text-muted": "#6c7086",
        "text-disabled": "#45475a",

        # Semantic text colors
        "text-success": "#a6e3a1",
        "text-warning": "#fab387",
        "text-error": "#f38ba8",
        "text-primary": "#89b4fa",
        "text-secondary": "#cba6f7",
        "text-accent": "#f9e2af",

        # Link styling
        "link-color": "#89b4fa",
        "link-style": "underline",
        "link-background-hover": "#89b4fa 15%",
        "link-color-hover": "#b4befe",
        "link-style-hover": "bold",

        # Button styling
        "button-foreground": "#cdd6f4",
        "button-color-foreground": "#11111b",
        "button-focus-text-style": "bold reverse",
    },
)
