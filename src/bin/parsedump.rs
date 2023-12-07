//! Debugging tool. Reads stdin, parses, and dumps to stdout.

use std::io::stdin;

use anyhow::Result;

fn main() -> Result<()> {
    let mut input = String::new();

    for line in stdin().lines() {
        let line = line?;
        input.push_str(&line);
        input.push('\n');
    }

    let script = souvenir::parse(&input)?;

    println!("{script:#?}");

    Ok(())
}
