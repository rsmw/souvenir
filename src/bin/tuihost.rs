use std::env::args;
use std::io::{stdin, stdout, Write};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};

use log::warn;

use souvenir::eval::GlobalHandle;
use souvenir::interpret::{Actor, IoPayload, ActorStatus};
use souvenir::value::Value;

fn main() -> Result<()> {
    let source = match args().nth(1) {
        Some(path) => std::fs::read_to_string(path)?,
        None => include_str!("../../example.svr").to_owned(),
    };

    let script = Arc::new(souvenir::parse(&source)?.compile()?);

    let globals = GlobalHandle::with_values(&[])?;

    let mut actor = Actor::from_script(script, globals);

    let mut instant = Instant::now();

    loop {
        let now = Instant::now();
        let dt = now - instant;
        instant = now;

        actor.tick(dt);

        if let Some(io) = actor.poll_io() {
            let response = match io.payload() {
                IoPayload::Quote { speaker, line } => {
                    if speaker.is_some() {
                        warn!("Not implemented: Speakers");
                    }

                    println!("{line}");

                    Value::FALSE
                },

                IoPayload::Menu { items } => {
                    for (index, item) in items.iter().enumerate() {
                        let index = index + 1;
                        println!("{index}.\t{item}");
                    }

                    Value::Int(prompt_digit()? - 1)
                },

                IoPayload::HostFfi { fn_name, args } => match (fn_name.as_ref(), args.as_slice()) {
                    ("music", [path]) => {
                        let path: &str = path.try_into()?;
                        eprintln!("** Playing music: {path:?} **");
                        Value::TRUE
                    },

                    (fn_name, args) => {
                        eprintln!("Not yet implemented: Host FFI: {fn_name}({args:?})");
                        break;
                    },
                },
            };

            if !(actor.fulfill(io, response)?) {
                eprintln!("Fulfillment failed");
            }

            continue;
        }

        match actor.status() {
            ActorStatus::Running => continue,
            ActorStatus::Sleeping { .. } => continue,

            ActorStatus::Retiring => break,
            ActorStatus::Hibernating => break,

            ActorStatus::Loading { path, args } => {
                bail!("Not implemented: Loading script {path}({args:?})");
            },

            ActorStatus::Blocked => {
                bail!("Actor execution stalled:\n{actor:#?}");
            },

            ActorStatus::Killed { error } => {
                eprintln!("Actor died: {error}");
                break;
            },
        }
    }

    Ok(())
}

fn prompt_digit() -> Result<i64> {
    loop {
        print!("> ");
        stdout().flush()?;

        let mut buf = String::new();
        stdin().read_line(&mut buf)?;

        match buf.trim().parse::<i64>() {
            Ok(int) => return Ok(int),
            Err(err) => eprintln!("{err}"),
        }
    }
}
