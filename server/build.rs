// The commit this server is built from, "-dirty" when tracked files differ from it: what a running server is (/health
// "build", its first log line).  2026-11-07: a session saw changed files on the M1 and took them for another
// session's patched server; they were copies of this Mac's - nothing said which code ran.
fn main() {
    let git = |args: &[&str]| {
        std::process::Command::new("git").args(args).output().ok().filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    // a build from a git archive (make linux-package's Docker build) has no .git: the commit comes in STRATA_BUILD_SHA
    println!("cargo:rerun-if-env-changed=STRATA_BUILD_SHA");
    let given = std::env::var("STRATA_BUILD_SHA").ok().filter(|s| !s.is_empty());
    let sha = given.clone().or_else(|| git(&["rev-parse", "--short=8", "HEAD"])).unwrap_or_else(|| "unknown".into());
    let dirty = given.is_none() && git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    println!("cargo:rustc-env=STRATA_BUILD={sha}{}", if dirty { "-dirty" } else { "" });
    for p in ["../.git/HEAD", "../.git/index", "src", "web", "build.rs"] {
        println!("cargo:rerun-if-changed={p}");
    }
}
