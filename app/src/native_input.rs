//! Clicks and typing in the browser pane the way a person's mouse and keyboard give them: AppKit events sent to the
//! web view's window, so the page gets trusted DOM events (a page script's are isTrusted false, and sites such as
//! Zillow act only on real ones).  The pointer goes to the element, then presses there: what a click is, so menus that
//! open on hover open.  No invented paths or timing: a site's "are you a person" check is the person's to answer.
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::{NSPoint, NSString};
use wry::{WebView, WebViewExtMacOS};

const KEY_DOWN: u64 = 10;
const KEY_UP: u64 = 11;
const MOUSE_DOWN: u64 = 1;
const MOUSE_UP: u64 = 2;
const MOUSE_MOVED: u64 = 5;

/// A key by name: (macOS virtual key code, the characters it types).
pub fn key_code(name: &str) -> Option<(u16, &'static str)> {
    Some(match name.to_ascii_lowercase().as_str() {
        "enter" | "return" => (36, "\r"),
        "tab" => (48, "\t"),
        "escape" | "esc" => (53, "\u{1b}"),
        "backspace" => (51, "\u{7f}"),
        "a" => (0, "a"),
        _ => return None,
    })
}

/// Modifier names -> NSEventModifierFlags.
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
    // the web view takes the keyboard, as a click into it would
    let _: bool = unsafe { msg_send![&*win, makeFirstResponder: &*wv] };
    (win, n)
}

/// One key press (down, up) with modifiers, into the page's focused element.
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

/// Put `text` where the page's focus is, as Edit > Paste does (the selection, if any, is replaced).  The person's
/// clipboard text is put back a moment later.
pub fn paste(web: &WebView, text: &str) -> Result<(), String> {
    let ty = NSString::from_str("public.utf8-plain-text");
    let pb: Retained<AnyObject> = unsafe { msg_send![class!(NSPasteboard), generalPasteboard] };
    let before: Option<Retained<NSString>> = unsafe { msg_send![&*pb, stringForType: &*ty] };
    unsafe {
        let _: isize = msg_send![&*pb, clearContents];
        let _: bool = msg_send![&*pb, setString: &*NSString::from_str(text), forType: &*ty];
    }
    let _ = window_of(web);
    let wv = web.webview();
    let _: () = unsafe { msg_send![&*wv, paste: std::ptr::null::<AnyObject>()] };
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

/// A click at (x, y), the page's CSS pixels from the web view's top left: the pointer moves there, then presses.
pub fn click(web: &WebView, x: f64, y: f64) -> Result<(), String> {
    let (win, n) = window_of(web);
    let wv = web.webview();
    // the web view's point (flipped: top-left) in the window's coordinates (bottom-left)
    let p: NSPoint = unsafe { msg_send![&*wv, convertPoint: NSPoint::new(x, y), toView: std::ptr::null::<AnyObject>()] };
    let moved: Option<Retained<AnyObject>> = unsafe {
        msg_send![class!(NSEvent), mouseEventWithType: MOUSE_MOVED, location: p, modifierFlags: 0u64, timestamp: 0.0f64,
                  windowNumber: n, context: std::ptr::null::<AnyObject>(), eventNumber: 0isize, clickCount: 0isize, pressure: 0.0f32]
    };
    if let Some(ev) = moved {
        // through the window, as the mouse's own moves come (sent to the web view alone, the page saw no mousemove:
        // tried on the M1), and to the web view for a window that is not key and passes no moves on
        let _: () = unsafe { msg_send![&*win, sendEvent: &*ev] };
        let _: () = unsafe { msg_send![&*wv, mouseMoved: &*ev] };
    }
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
        assert_eq!(key_code("F13"), None);
        assert_eq!(modifier_flags(&["cmd", "shift"]), (1 << 20) | (1 << 17));
    }
}
