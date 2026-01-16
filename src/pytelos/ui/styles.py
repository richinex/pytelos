"""CSS styles for the TUI.

Hides layout and styling decisions from the application logic.
Uses Textual CSS features: nesting, pseudo-classes, variables.

Design Philosophy:
- Clean, modern aesthetic with subtle depth
- Clear visual hierarchy through color and spacing
- Smooth interactive states with glow effects
- Consistent border treatments and rounded corners
"""

APP_CSS = """
/* ============================================
   CSS Variables - Design Tokens
   ============================================ */

/* Panel borders with subtle glow capability */
$panel-border: tall $border;
$panel-border-focus: tall $primary;
$panel-border-active: tall $accent;

/* Subtle glow effects for focus states */
$glow-primary: $primary 20%;
$glow-accent: $accent 20%;
$glow-success: $success 20%;
$glow-warning: $warning 20%;
$glow-error: $error 20%;

/* ============================================
   Main Screen Layout - 2x2 Grid
   ============================================ */
Screen {
    layout: grid;
    grid-size: 2 2;
    grid-columns: 3fr 2fr;
    grid-rows: 1fr auto;
    background: $background;
}

/* ============================================
   Chat History Panel - Primary Focus Area
   ============================================ */
#chat-history {
    height: 100%;
    background: $panel;
    border: round $primary 60%;
    border-title-color: $primary;
    border-title-style: bold;
    border-title-align: left;
    border-subtitle-color: $text-muted;
    border-subtitle-align: right;
    padding: 0 1;
    scrollbar-gutter: stable;

    /* Focus state with glow */
    &:focus {
        border: round $primary;
    }

    /* Focus-within for child focus */
    &:focus-within {
        border: round $primary-lighten-1;
    }
}

/* ============================================
   Right Panel Container
   ============================================ */
#right-panel {
    height: 100%;
    background: transparent;
    padding: 0;
}

/* ============================================
   Execution Log Panel
   ============================================ */
#execution-log {
    height: 2fr;
    background: $panel;
    border: round $secondary 60%;
    border-title-color: $secondary;
    border-title-style: bold;
    border-title-align: left;
    border-subtitle-color: $text-muted;
    border-subtitle-align: right;
    padding: 0 1;
    overflow-y: scroll;
    overflow-x: auto;
    scrollbar-gutter: stable;

    &:focus {
        border: round $secondary;
    }

    /* Executing state - animated border color */
    &.executing {
        border: round $warning;
        border-title-color: $warning;
    }
}

/* ============================================
   Plot Panel - Image Display
   ============================================ */
#plot-panel {
    height: 1fr;
    min-height: 10;
    background: $panel;
    border: round $accent 60%;
    border-title-color: $accent;
    border-title-style: bold;
    border-title-align: left;
    border-subtitle-color: $text-muted;
    border-subtitle-align: right;
    padding: 0 1;

    &:focus {
        border: round $accent;
    }
}

/* Plot image - auto-size within bounds */
#plot-image {
    width: auto;
    height: auto;
}

/* ============================================
   Debug/Log Panel
   ============================================ */
#debug-panel {
    height: auto;
    min-height: 6;
    max-height: 14;
    background: $panel;
    border: round $warning 60%;
    border-title-color: $warning;
    border-title-style: bold;
    border-title-align: left;
    border-subtitle-color: $text-muted;
    border-subtitle-align: right;
    padding: 0 1;
    overflow-y: scroll;
    overflow-x: auto;
    scrollbar-gutter: stable;
    margin-top: 1;

    &:focus {
        border: round $warning;
    }
}

/* ============================================
   Bottom Bar - Metrics + Input
   ============================================ */
#bottom-bar {
    column-span: 2;
    height: auto;
    padding: 1;
    background: $panel;
    border-top: solid $border;
}

/* ============================================
   Metrics Panel - Compact Stats Display
   ============================================ */
#metrics {
    height: 1;
    padding: 0 2;
    margin-bottom: 1;
    background: $surface;
    border: none;
    color: $foreground;
}

/* ============================================
   Chat Input Bar - Text Entry + Send
   ============================================ */
ChatInputBar {
    height: 5;
    border: round $primary 60%;
    background: $panel;

    &:focus-within {
        border: round $primary;
    }
}

/* Text input area */
#chat-input {
    width: 1fr;
    height: 100%;
    border: none;
    padding: 0 1;
    background: transparent;

    &:focus {
        background: transparent;
    }
}

/* Send button with prominent styling */
#send-btn {
    width: 10;
    height: 100%;
    margin: 0 0 0 1;
    min-width: 8;
    border: tall $success;
    background: $success;
    color: $background;
    text-style: bold;

    &:hover {
        background: $success-lighten-1;
        border: tall $success-lighten-1;
    }

    &:focus {
        background: $success-lighten-2;
        border: tall $success-lighten-2;
        text-style: bold reverse;
    }
}

/* ============================================
   Chat Messages - Conversation Display
   ============================================ */
.chat-message {
    width: 100%;
    height: auto;
    margin: 0 0 1 0;
    padding: 1 2;
    border: none;
    background: transparent;
}

/* User messages - Green accent with subtle background */
.user-message {
    border-left: tall $success;
    background: $success 8%;

    & .message-header {
        color: $success;
        text-style: bold;
    }

    & .message-content {
        color: $foreground;
    }

    &:hover {
        background: $success 12%;
        border-left: tall $success-lighten-1;
    }
}

/* Assistant messages - Purple/Mauve accent */
.assistant-message {
    border-left: tall $secondary;
    background: $secondary 8%;

    & .message-header {
        color: $secondary;
        text-style: bold;
    }

    & .message-content {
        color: $foreground;
    }

    &:hover {
        background: $secondary 12%;
        border-left: tall $secondary-lighten-1;
    }
}

/* Message header with timestamp */
.message-header {
    height: auto;
    padding: 0;
    margin-bottom: 0;
}

/* Message content area */
.message-content {
    height: auto;
    padding: 0;
    margin: 0;
}

/* ============================================
   Maximized Panel States
   ============================================ */
#chat-history.-maximized,
#execution-log.-maximized {
    column-span: 2;
    row-span: 1;
}

/* ============================================
   Tooltip Styling
   ============================================ */
Tooltip {
    background: $panel;
    color: $foreground;
    border: tall $border;
    padding: 0 1;
}

/* ============================================
   Notification Toasts
   ============================================ */
Toast {
    background: $surface;
    border: tall $border;
    padding: 0 1;

    &.-information {
        border: tall $primary;
        background: $primary 12%;
        color: $foreground;
    }

    &.-error {
        border: tall $error;
        background: $error 12%;
        color: $foreground;
    }

    &.-warning {
        border: tall $warning;
        background: $warning 12%;
        color: $foreground;
    }
}

/* ============================================
   Scrollbar Styling - Subtle and Refined
   ============================================ */
* {
    scrollbar-background: $panel;
    scrollbar-color: $surface-lighten-1;
    scrollbar-color-hover: $primary 50%;
    scrollbar-color-active: $primary;
    scrollbar-size: 1 1;
}

/* ============================================
   Header - App Title Bar
   ============================================ */
Header {
    background: $panel;
    color: $foreground;
    dock: top;
    height: 1;
}

HeaderTitle {
    color: $primary;
    text-style: bold;
}

/* ============================================
   Footer - Keyboard Shortcuts
   ============================================ */
Footer {
    background: $panel;
    height: auto;
}

FooterKey {
    background: $surface;
    color: $foreground;
    padding: 0 1;

    & > .footer-key--key {
        background: $primary 80%;
        color: $background;
        text-style: bold;
    }

    &:hover {
        background: $primary 15%;
    }

    &:focus {
        background: $primary 25%;
    }
}

/* ============================================
   Global Button Variants
   ============================================ */
Button {
    min-width: 8;
    height: 3;
    border: tall $border;
    background: $surface;
    color: $foreground;

    &:hover {
        text-style: bold;
        background: $surface-lighten-1;
    }

    &:focus {
        border: tall $primary;
    }
}

/* Primary action button */
Button.-primary {
    background: $primary;
    color: $background;
    border: tall $primary;

    &:hover {
        background: $primary-lighten-1;
        border: tall $primary-lighten-1;
    }

    &:focus {
        background: $primary-lighten-2;
        text-style: bold reverse;
    }
}

/* Success/Confirm button */
Button.-success {
    background: $success;
    color: $background;
    border: tall $success;

    &:hover {
        background: $success-lighten-1;
        border: tall $success-lighten-1;
    }

    &:focus {
        background: $success-lighten-2;
        text-style: bold reverse;
    }
}

/* Error/Danger button */
Button.-error {
    background: $error;
    color: $background;
    border: tall $error;

    &:hover {
        background: $error-lighten-1;
        border: tall $error-lighten-1;
    }

    &:focus {
        background: $error-lighten-2;
        text-style: bold reverse;
    }
}

/* Warning button */
Button.-warning {
    background: $warning;
    color: $background;
    border: tall $warning;

    &:hover {
        background: $warning-lighten-1;
        border: tall $warning-lighten-1;
    }

    &:focus {
        background: $warning-lighten-2;
        text-style: bold reverse;
    }
}

/* ============================================
   Markdown Content Styling
   ============================================ */
Markdown {
    margin: 0;
    padding: 0;
}

MarkdownH1 {
    color: $primary;
    text-style: bold;
    margin: 1 0;
}

MarkdownH2 {
    color: $secondary;
    text-style: bold;
    margin: 1 0;
}

MarkdownH3 {
    color: $accent;
    text-style: bold;
    margin: 1 0;
}

MarkdownFence {
    background: $panel;
    border: round $border;
    margin: 1 0;
    padding: 1;
}

MarkdownBlockQuote {
    border-left: wide $primary;
    background: $primary 8%;
    padding: 0 1;
    margin: 1 0;
}

/* ============================================
   DataTable Styling (if used)
   ============================================ */
DataTable {
    background: $surface;
    border: tall $border;
}

DataTable > .datatable--header {
    background: $panel;
    color: $primary;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: $primary 30%;
}

/* ============================================
   OptionList Styling (if used)
   ============================================ */
OptionList {
    background: $surface;
    border: tall $border;
    padding: 0 1;
}

OptionList > .option-list--option-highlighted {
    background: $primary 20%;
}

OptionList > .option-list--option-hover {
    background: $surface-lighten-1;
}
"""
