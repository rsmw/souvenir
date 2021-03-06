use std::fmt::*;

use ast;

impl Display for ast::Modpath {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.0.join(":"))
    }
}

impl Display for ast::Label {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &ast::Label::Qualified(ref qfd) => write!(f, "{}", qfd),
            &ast::Label::Local { ref name } => write!(f, "'{}", name),
            &ast::Label::Anonymous => write!(f, ""),
        }
    }
}

impl Display for ast::Ident {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &ast::Ident { ref name } = self;
        write!(f, "{}", name)
    }
}

impl Display for ast::Stmt {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &ast::Stmt::Empty => writeln!(f, ""),

            &ast::Stmt::Disarm { ref target } => {
                write!(f, "disarm {}", target)
            },

            &ast::Stmt::Let { ref name, ref value } => {
                write!(f, "let {} = {}", name, value)
            },

            &ast::Stmt::Listen { ref name, ref arms } => {
                writeln!(f, "listen {}", name)?;
                for arm in arms.iter() {
                    writeln!(f, "{}", arm)?;
                }
                writeln!(f, ";;")
            },

            &ast::Stmt::Say { ref message } => {
                writeln!(f, "say {}", message)
            },

            _ => write!(f, "STATEMENT"),
        }
    }
}

impl Display for ast::TrapArm {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &ast::TrapArm {
            ref pattern,
            ref origin,
            ref guard,
            ref body,
        } = self;

        writeln!(f, "| {} from {} when {}", pattern, origin, guard)?;
        for stmt in body.0.iter() {
            write!(f, "{}", format!("{}", stmt).indent_lines())?;
        }
        Ok(())
    }
}

impl Display for ast::Expr {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &ast::Expr::Id(ref id) => write!(f, "{}", id),

            &ast::Expr::Str(_) => write!(f, "> I don't think so"),

            _ => write!(f, "EXPRESSION"),
        }
    }
}

impl Display for ast::Cond {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &ast::Cond::True => write!(f, "#yes"),
            &ast::Cond::False => write!(f, "#no"),

            _ => write!(f, "CONDITION"),
        }
    }
}

impl Display for ast::Atom {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &ast::Atom::User(ref name) => write!(f, "#{}", name),
        }
    }
}

impl Display for ast::Pat {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &ast::Pat::Hole => write!(f, "_"),

            &ast::Pat::Assign(ref id) => write!(f, "{}", id),

            &ast::Pat::Match(ref expr) => write!(f, "{}", expr),

            &ast::Pat::List(ref items) => write!(f, "[{}]", {
                items.iter()
                    .map(|i| format!("{}", i))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
        }
    }
}

impl Display for ast::SceneName {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &ast::SceneName { ref name, ref in_module } = self;

        match in_module.as_ref() {
            Some(path) => write!(f, "{}:{}", path, name),
            None => write!(f, "{}", name),
        }
    }
}

impl Display for ast::QfdSceneName {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &ast::QfdSceneName { ref name, ref in_module } = self;
        write!(f, "{}:{}", in_module, name)
    }
}

impl Display for ast::QfdLabel {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &ast::QfdLabel { ref name, ref in_scene } = self;
        write!(f, "{}'{}", in_scene, name)
    }
}

impl Display for ast::Call {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &ast::Call(ref name, ref args) = self;

        let args = args.iter()
            .map(|expr| format!("{}", expr))
            .collect::<Vec<_>>();

        write!(f, "{}({})", name, args.join(", "))
    }
}

pub trait IndentLines {
    fn indent_lines(&self) -> String;
}

impl<'a> IndentLines for &'a str {
    fn indent_lines(&self) -> String {
        self.lines()
            .map(|line| format!("    {}", line))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl IndentLines for String {
    fn indent_lines(&self) -> String {
        self.as_str().indent_lines()
    }
}

use ast::tokens::{Tok, TokErr};

impl Display for TokErr {
    fn fmt(&self, f: &mut Formatter) -> Result {
        use ast::tokens::ErrReason;

        let &TokErr { ref reason, .. } = self;

        match reason {
            &ErrReason::UnrecognizedToken => {
                write!(f, "Unrecognized token")
            },

            &ErrReason::InvalidStringLiteral => {
                write!(f, "Invalid string literal")
            },

            &ErrReason::InvalidNumberLiteral => {
                write!(f, "Invalid number literal")
            },

            &ErrReason::InvalidCamelCase => {
                write!(f, "Variable names must be in CamelCase")
            },

            &ErrReason::InvalidSnakeCase => {
                write!(f, "Scene and atom names must be in snake_case")
            },

            &ErrReason::InvalidScreamingCase => {
                write!(f, "Macro names must be in SCREAMING_CASE")
            },
        }
    }
}

impl<'a> Display for Tok<'a> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", match self {
            &Tok::EndLn => ";",
            &Tok::EndBlk => ";;",

            &Tok::KwDisarm => "disarm",
            &Tok::KwFrom => "from",
            &Tok::KwGiven => "given",
            &Tok::KwIf => "if",
            &Tok::KwLet => "let",
            &Tok::KwListen => "listen",
            &Tok::KwSpawn => "spawn",
            &Tok::KwThen => "then",
            &Tok::KwTrace => "trace",
            &Tok::KwTrap => "trap",
            &Tok::KwWait => "wait",
            &Tok::KwWeave => "weave",
            &Tok::KwWhen => "when",

            &Tok::NmScene(ref s) => s,
            &Tok::NmLabel(ref s) => s,
            &Tok::NmMacro(ref s) => s,
            &Tok::NmVar(ref s) => s,

            &Tok::LitAtom(ref s) => s,
            &Tok::LitInt(ref s) => s,
            &Tok::LitRoll(ref s) => s,
            &Tok::LitStr(ref s) => s,

            &Tok::OpAssign => "=",
            &Tok::OpComma => ",",
            &Tok::OpDot => ".",
            &Tok::OpSend => "<-",
            &Tok::OpColon => ":",
            &Tok::OpMul => "*",
            &Tok::OpDiv => "/",
            &Tok::OpAdd => "+",
            &Tok::OpSub => "-",

            &Tok::Pipe => "|",
            &Tok::Hole => "_",
            &Tok::Scene => "==",
            &Tok::Divert => "->",

            &Tok::LParen => "(",
            &Tok::RParen => ")",
            &Tok::LSquare => "[",
            &Tok::RSquare => "]",
            &Tok::LCurly => "{",
            &Tok::RCurly => "}",
            &Tok::LAngle => "<",
            &Tok::RAngle => ">",
        })
    }
}

use driver::*;

impl Display for LoadErr {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &LoadErr::Description(ref s) => write!(f, "{}", s),
            &LoadErr::Parse(ref s) => write!(f, "{}", s),

            &LoadErr::Io(ref err) => {
                use std::error::Error;
                write!(f, "{}", err.description())
            },

            &LoadErr::PathIsNotLoadable(ref path) => {
                write!(f, "Couldn't find modules in {}", path)
            },

            &LoadErr::ModpathIsNotUnicode(ref path) => {
                write!(f, "Unable to decode {:?}", path)
            },

            &LoadErr::ModpathIsNotValid(ref path) => {
                write!(f, "{:?} is not a valid module path", path)
            },
        }
    }
}

impl Display for CompileErr {
    fn fmt(&self, f: &mut Formatter) -> Result {
        match self {
            &CompileErr::Internal(ICE(ref ice)) => {
                write!(f, "INTERNAL ERROR!!! {}", ice)
            },

            &CompileErr::Load(ref err) => write!(f, "{}", err),

            &CompileErr::BuildErrs(ref errs) => {
                for err in errs.iter() {
                    writeln!(f, "{}", err)?;
                }

                Ok(())
            },
        }
    }
}

impl Display for BuildErrWithCtx {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let &BuildErrWithCtx(ref cause, ref ctx) = self;

        match cause {
            &BuildErr::SceneWasOverqualified(ref name) => {
                writeln!(f, "Scene names shouldn't be qualified in their definitions:")?;
                write!(f, "{}", name.in_module.as_ref().unwrap())?;
            },

            &BuildErr::NoSuchModule(ref path) => {
                write!(f, "The module {} was not found.", path)?;
            },

            &BuildErr::NoSuchScene(ref name) => {
                write!(f, "The scene {:?} was not found in the module {}.", &name.name, name.in_module)?;
            },

            &BuildErr::WrongNumberOfArgs { ref call, ref wanted, ref got } => {
                writeln!(f, "In the expression:\n{}", call)?;
                write!(f, "The function {} needs {} args, but was called with {}", &call.0.name, wanted, got)?;
            },

            &BuildErr::InvalidNumber(ref s) => {
                write!(f, "The number {} could not be parsed", s)?;
            },

            &BuildErr::IoInPrelude => {
                writeln!(f, "IO not allowed in module prelude")?;
            },

            &BuildErr::SelfInPrelude => {
                writeln!(f, "Special variable Self cannot be used in prelude")?;
            },

            &BuildErr::LabelInPrelude(ref _label) => {
                writeln!(f, "Traps not allowed in module prelude")?;
            },

            e => write!(f, "Can't describe this error yet: {:?}", e)?,
        };

        write!(f, "{}", ctx)
    }
}

impl Display for ErrCtx {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let stack = match self {
            &ErrCtx::Global(ref modpath, ref stack) => {
                writeln!(f, "  In module {}:", modpath)?;
                stack

            },

            &ErrCtx::Local(ref scene_name, ref stack) => {
                writeln!(f, "  In the definition of scene {}", scene_name)?;
                stack
            },

            &ErrCtx::NoContext => {
                return Ok(()) // Write nothing!
            },
        };

        if let Some(first) = stack.first() {
            writeln!(f, "{}", first.to_string().indent_lines())?;
        }

        if stack.len() > 1 {
            if let Some(last) = stack.last() {
                writeln!(f, "{}", last.to_string().indent_lines())?;
            }
        }

        Ok(())
    }
}
