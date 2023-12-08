use std::fmt;

use std::collections::HashSet;

use crate::ast::{Expr, Splice};
use crate::interpret::{Op, Script, TaskLabel, MenuItem};

impl fmt::Display for Script {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut used_labels = HashSet::<TaskLabel>::new();

        for (index, op) in self.body.as_ref().iter().enumerate() {
            match op {
                &Op::CancelHandler { label } |
                &Op::PushHandler { label, .. } => {
                    used_labels.insert(label);
                },

                &Op::EnterBlock { offset } |
                &Op::Jz { offset, .. } |
                &Op::Jump { offset } => {
                    let label = TaskLabel(index + offset);
                    used_labels.insert(label);
                },

                Op::Menu { choices } => {
                    for &MenuItem { target, .. } in choices.as_ref() {
                        used_labels.insert(target);
                    }
                },

                _ => continue,
            }
        }

        for (index, op) in self.body.as_ref().iter().enumerate() {
            let label = TaskLabel(index);

            for (name, info) in self.pages.iter() {
                if info.entry_point == label {
                    writeln!(f, "{name}:")?;
                }
            }

            if used_labels.contains(&label) || index == 0 {
                writeln!(f, "{label}:")?;
            }

            write!(f, "\t")?;

            match op {
                Op::Eval { expr, dst } => {
                    writeln!(f, "Eval ({dst:?}) = {expr}")?;
                },

                Op::EnterBlock { offset } => {
                    let target = TaskLabel(index + offset);
                    writeln!(f, "EnterBlock until {target}")?;
                },

                Op::PushHandler { pattern, label, cancel } => {
                    let cancel = if *cancel { " (autocancel)" } else { "" };
                    writeln!(f, "PushHandler{cancel} {label}, {pattern:?}")?;
                },

                Op::CancelHandler { label } => {
                    writeln!(f, "CancelHandler {label}")?;
                },

                Op::Jz { guard, offset } => {
                    let target = TaskLabel(index + offset);
                    writeln!(f, "Jz {target} unless {guard}")?;
                },

                Op::Jump { offset } => {
                    let target = TaskLabel(index + offset);
                    writeln!(f, "Jump {target}")?;
                },

                Op::Menu { choices } => {
                    writeln!(f, "Menu:")?;

                    for MenuItem { guard, prompt, target } in choices.iter() {
                        write!(f, "\t\t{target} ")?;

                        if let Some(guard) = guard {
                            write!(f, "if {guard} ")?;
                        }

                        writeln!(f, "{prompt:?}")?;
                    }
                },

                Op::Quote { speaker, lines } => {
                    write!(f, "Quote")?;

                    if let Some(speaker) = speaker {
                        write!(f, "({speaker})")?;
                    }

                    writeln!(f, ":")?;

                    for line in lines.iter() {
                        match line {
                            Splice::Expr { expr } => writeln!(f, "\t\t{expr}")?,
                            Splice::Verbatim { value } => writeln!(f, "\t\t{value:?}")?,
                        }
                    }
                },

                Op::Trace { on } => writeln!(f, "Trace {on}")?,

                Op::Tailcall { path, args } => {
                    write!(f, "Tailcall {path:?}(")?;

                    for (i, arg) in args.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }

                        write!(f, "{arg}")?;
                    }

                    writeln!(f, ")")?;
                },

                Op::Wait { amount } => {
                    writeln!(f, "Wait {amount}")?;
                },

                Op::Return => writeln!(f, "Return")?,

                Op::Retire => writeln!(f, "Retire")?,

                Op::Hcf { reason } => writeln!(f, "Hcf {reason:?}")?,
            }
        }

        let end = TaskLabel(self.body.len());
        if used_labels.contains(&end) {
            writeln!(f, "{end}:")?;
        }

        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Local { name } => write!(f, "{name}"),

            Expr::Global { name } => write!(f, "${name}"),

            Expr::Atom { name } => write!(f, "#{name}"),

            Expr::Member { lhs, name } => write!(f, "({lhs}).{name}"),

            Expr::FnCall { lhs, args } => {
                write!(f, "({lhs})(")?;
                let mut first = true;
                for arg in args {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }

                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            },

            Expr::Int { value } => write!(f, "{value}"),

            Expr::String { value } => write!(f, "{value:?}"),

            Expr::Infix { lhs, op, rhs } => write!(f, "({lhs}) {op} ({rhs})"),

            Expr::Neg { rhs } => write!(f, "- ({rhs})"),

            Expr::Not { rhs } => write!(f, "not ({rhs})"),

            Expr::Paren { value } => write!(f, "{value}"),
        }
    }
}

impl fmt::Display for TaskLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &TaskLabel(addr) = self;
        write!(f, "{addr:08x}")
    }
}

#[test]
fn if_then() {
    let src = r#"
        if X ?= 1
            > Branch one
        else if X ?= 2
            > Branch two
        else if X ?= 3
            > Branch three
        else
            > Fallback
        ;;
        "#;

    let script = crate::parse(src).unwrap()
    .compile().unwrap();

    println!("{script}");

    println!("OK");
}

#[test]
fn beer() {
    let src = r#"
    -- Print the lyrics of 99 Bottles of Beer

    global $count
    $count = 99

    :: verse

    -- Local variables are declared before use
    let S = "s"
    if $count ?= 1
        S = ""
    ;;

    > {$count} bottle{S} of beer on the wall
    > {$count} bottle{S} of beer
    > Take one down
    > Pass it around

    $count -= 1

    if $count ?= 1
        S = ""
    else
        S = "s"
    ;;

    let Count = $count
    if Count ?= 0
        Count = "No more"
    ;;

    > {Count} bottle{S} of beer on the wall
    >

    if not ($count ?= 0)
        -> verse
    ;;

    bye
    "#;

    let script = crate::parse(src).unwrap();
    eprintln!("{script:#?}\n\n\n");

    let script = script.compile().unwrap();
    println!("{script}\n\n\n");

    println!("OK");
}
