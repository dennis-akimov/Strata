//! Input for the browser pane the way a person gives it: AppKit events into the web view's window, so the page gets
//! trusted DOM events.  Google's editors (Sheets, My Maps) ignore what a page script dispatches (isTrusted false); a
//! paste through the pasteboard, a key press and a click as NSEvents reach them as typing and clicking would.
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::{NSPoint, NSString};
use wry::{WebView, WebViewExtMacOS};

const KEY_DOWN: u64 = 10;
const KEY_UP: u64 = 11;
const MOUSE_DOWN: u64 = 1;
const MOUSE_UP: u64 = 2;

/// A key by name: (macOS virtual key code, the characters it types).
pub fn key_code(name: &str) -> Option<(u16, &'static str)> {
    Some(match name.to_ascii_lowercase().as_str() {
        "enter" | "return" => (36, "\r"),
        "tab" => (48, "\t"),
        "escape" | "esc" => (53, "\u{1b}"),
        "backspace" => (51, "\u{7f}"),
        "delete" => (117, "\u{f728}"),
        "up" => (126, "\u{f700}"),
        "down" => (125, "\u{f701}"),
        "left" => (123, "\u{f702}"),
        "right" => (124, "\u{f703}"),
        "home" => (115, "\u{f729}"),
        "end" => (119, "\u{f72b}"),
        "a" => (0, "a"),
        "c" => (8, "c"),
        "v" => (9, "v"),
        "z" => (6, "z"),
        _ => return None,
    })
}

/// Modifier names ("cmd", "shift", "alt", "ctrl") -> NSEventModifierFlags.
pub fn modifier_flags(mods: &[&str]) -> u64 {
    mods.iter().map(|m| match m.to_ascii_lowercase().as_str() {
        "shift" => 1 << 17,
        "ctrl" | "control" => 1 << 18,
        "alt" | "option" => 1 << 19,
        "cmd" | "command" | "meta" => 1 << 20,
        _ => 0,
    }).fold(0, |a, b| a | b)
}

fn window_of(web: &WebView) -> (Retained<AnyObject>, isize) {
    let wv = web.webview();
    let win: Retained<AnyObject> = unsafe { msg_send![&*wv, window] };
    let n: isize = unsafe { msg_send![&*win, windowNumber] };
    // the web view takes the keyboard (as a click into it would)
    let _: bool = unsafe { msg_send![&*win, makeFirstResponder: &*wv] };
    (win, n)
}

/// One key press (down and up) with modifiers, into the focused element of the page.
pub fn press(web: &WebView, name: &str, mods: &[&str]) -> Result<(), String> {
    let (code, chars) = key_code(name).ok_or_else(|| format!("no key named {name:?}"))?;
    let flags = modifier_flags(mods);
    let (win, n) = window_of(web);
    let s = NSString::from_str(chars);
    for kind in [KEY_DOWN, KEY_UP] {
        let ev: Option<Retained<AnyObject>> = unsafe {
            msg_send![class!(NSEvent), keyEventWithType: kind, location: NSPoint::new(0.0, 0.0), modifierFlags: flags,
                      timestamp: 0.0f64, windowNumber: n, context: std::ptr::null::<AnyObject>(), characters: &*s,
                      charactersIgnoringModifiers: &*s, isARepeat: false, keyCode: code]
        };
        let ev = ev.ok_or("the key event was not made")?;
        let _: () = unsafe { msg_send![&*win, sendEvent: &*ev] };
    }
    Ok(())
}

/// Paste `text` (tab-separated rows paste as cells in a spreadsheet) where the page's focus is: on the pasteboard, then
/// the web view's own Paste (Edit > Paste).  The person's clipboard is put back after.
pub fn paste(web: &WebView, text: &str) -> Result<(), String> {
    let pb: Retained<AnyObject> = unsafe { msg_send![class!(NSPasteboard), generalPasteboard] };
    let ty = NSString::from_str("public.utf8-plain-text");
    let before: Option<Retained<NSString>> = unsafe { msg_send![&*pb, stringForType: &*ty] };
    let set = |t: &str| unsafe {
        let _: isize = msg_send![&*pb, clearContents];
        let _: bool = msg_send![&*pb, setString: &*NSString::from_str(t), forType: &*ty];
    };
    set(text);
    let _ = window_of(web);
    let wv = web.webview();
    let _: () = unsafe { msg_send![&*wv, paste: std::ptr::null::<AnyObject>()] };
    // the paste reads the pasteboard as it runs: the person's text back a moment later
    let back = before.map(|b| b.to_string());
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1500));
        if let Some(b) = back {
            unsafe {
                let pb: Retained<AnyObject> = msg_send![class!(NSPasteboard), generalPasteboard];
                let ty = NSString::from_str("public.utf8-plain-text");
                let _: isize = msg_send![&*pb, clearContents];
                let _: bool = msg_send![&*pb, setString: &*NSString::from_str(&b), forType: &*ty];
            }
        }
    });
    Ok(())
}

/// A click at (x, y) in the page's CSS pixels (from the top left of the web view), as a mouse would.
pub fn click(web: &WebView, x: f64, y: f64) -> Result<(), String> {
    let (win, n) = window_of(web);
    let wv = web.webview();
    // the web view's point (flipped: top-left) in the window's coordinates (bottom-left)
    let p: NSPoint = unsafe { msg_send![&*wv, convertPoint: NSPoint::new(x, y), toView: std::ptr::null::<AnyObject>()] };
    for kind in [MOUSE_DOWN, MOUSE_UP] {
        let ev: Option<Retained<AnyObject>> = unsafe {
            msg_send![class!(NSEvent), mouseEventWithType: kind, location: p, modifierFlags: 0u64, timestamp: 0.0f64,
                      windowNumber: n, context: std::ptr::null::<AnyObject>(), eventNumber: 0isize, clickCount: 1isize,
                      pressure: 1.0f32]
        };
        let ev = ev.ok_or("the mouse event was not made")?;
        let _: () = unsafe { msg_send![&*win, sendEvent: &*ev] };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_modifiers_by_name() {
        assert_eq!(key_code("Enter"), Some((36, "\r")));
        assert_eq!(key_code("tab"), Some((48, "\t")));
        assert_eq!(key_code("F13"), None);
        assert_eq!(modifier_flags(&["cmd", "shift"]), (1 << 20) | (1 << 17));
        assert_eq!(modifier_flags(&[]), 0);
    }
}
