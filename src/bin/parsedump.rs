//! Debugging tool. Reads stdin, parses, and dumps to stdout.

use std::{error::Error, io::stdin};

fn main() -> Result<(), Box<dyn Error>> {
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
