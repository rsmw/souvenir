use std::io::stdin;

use anyhow::Result;

fn main() -> Result<()> {
    stderrlog::new()
    .verbosity(5)
    .init().unwrap();

    let mut input = String::new();

    for line in stdin().lines() {
        let line = line?;
        input.push_str(&line);
        input.push('\n');
    }

    let script = souvenir::parse(&input)?;
    let compiled = script.compile()?;

    println!("{compiled}");

    Ok(())
}
