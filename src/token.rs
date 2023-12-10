use std::{iter::Peekable, str::Chars, fmt};

use super::ast::Binop;

pub(crate) struct Lexer<'a> {
    pub(crate) src: &'a str,
    depth: St,
    pub(crate) start: usize,
    pub(crate) input: Peekable<Chars<'a>>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct Token<'src> {
    pub content: &'src str,
    pub offset: usize,
    pub kind: Tok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Tok {
    QuoteStart,
    QuoteVerbatim(String),
    QuoteEscape(char),
    // QuoteInterpOpen -- use CurlyOpen
    // QuoteInterpClose -- use CurlyClose
    // QuoteEnd -- use LineBreak
    LitString(String),
    LitInt(u32),
    Atom(String),
    Ident(String),
    GlobalName(String),
    KwdGlobal,
    KwdEnum,
    KwdLet,
    KwdIf,
    KwdElse,
    KwdOn,
    KwdListen,
    KwdMenu,
    KwdMusic,
    KwdTrace,
    KwdContinue,
    KwdHibernate,
    KwdReturn,
    KwdWait,
    KwdBye,
    KwdNot,
    Less,
    Equal,
    Greater,
    Stitch,
    Plus,
    Minus,
    Splat,
    Slash,
    Assignment(Option<Binop>),
    GotoArrow,
    LabelMarker,
    OptionPipe,
    Semicolons,
    LineBreak,
    Dot,
    Ellipsis,
    Comment,
    Whitespace,
    ParenOpen,
    ParenClose,
    CurlyOpen,
    CurlyClose,
    InvalidComment,
    InvalidInt(String),
    Unrecognized(char),
}

/// Tracks current depth in the string interpolation hierarchy
#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
enum St {
    /// Outermost layer: Statements, expressions, etc.
    #[default]
    Root,

    /// Inside a string
    Quote,

    /// Inside an interpolated expression
    Interp,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src,
            depth: St::Root,
            start: 0,
            input: src.chars().peekable(),
        }
    }

    pub(crate) fn text_at(&self, start: usize) -> &'src str {
        &self.src[.. self.start][start ..]
    }

    pub(crate) fn bump(&mut self) -> Option<char> {
        let ch = self.input.next()?;
        self.start += ch.len_utf8();
        Some(ch)
    }

    pub(crate) fn bump_while(&mut self, f: impl Fn(char) -> bool) {
        while let Some(&c) = self.input.peek() {
            if f(c) {
                self.bump().unwrap();
            } else {
                break;
            }
        }
    }

    pub(crate) fn bump_if_equal(&mut self, ch: char) -> bool {
        match self.input.peek().cloned() {
            Some(c) if c == ch => {
                self.bump().unwrap();
                true
            },

            _ => false,
        }
    }

    pub fn skip_whitespace(self) -> impl Iterator<Item=Token<'src>> {
        self.filter(|t| match t.kind {
            Tok::Comment | Tok::Whitespace => false,
            _ => true,
        })
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Token<'src>;

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.start;
        let ch = self.bump()?;

        let kind: Tok = match ch {
            // Order is very important for these first few cases

            '{' => {
                self.depth = match self.depth {
                    St::Quote => St::Interp,
                    same => same,
                };

                Tok::CurlyOpen
            },

            '}' => {
                self.depth = match self.depth {
                    St::Interp => St::Quote,
                    same => same,
                };

                Tok::CurlyClose
            },

            '\n' => {
                self.depth = St::Root;
                Tok::LineBreak
            },

            '\\' if self.depth == St::Quote => {
                match self.input.peek() {
                    None | Some('\n') => Tok::Unrecognized('\\'),

                    Some(&e) => {
                        self.bump();
                        Tok::QuoteEscape(e)
                    },
                }
            },

            _ if self.depth == St::Quote => {
                self.bump_while(|c| {
                    !"\\{}\n".contains(c)
                });

                Tok::QuoteVerbatim(self.text_at(offset).into())
            },

            ',' => Tok::LineBreak,

            '|' => Tok::OptionPipe,

            '(' => Tok::ParenOpen,

            ')' => Tok::ParenClose,

            '=' => Tok::Assignment(None),

            '?' => if self.bump_if_equal('=') {
                Tok::Equal
            } else if self.bump_if_equal('<') {
                Tok::Less
            } else if self.bump_if_equal('>') {
                Tok::Greater
            } else {
                Tok::Unrecognized('?')
            },

            '.' => if self.bump_if_equal('.') {
                if self.bump_if_equal('.') {
                    Tok::Ellipsis
                } else {
                    Tok::Unrecognized('.')
                }
            } else {
                Tok::Dot
            },

            ':' => if self.bump_if_equal(':') {
                Tok::LabelMarker
            } else {
                Tok::Unrecognized(':')
            },

            ';' => if self.bump_if_equal(';') {
                Tok::Semicolons
            } else {
                Tok::Unrecognized(';')
            },

            '+' => if self.bump_if_equal('=') {
                Tok::Assignment(Some(Binop::Add))
            } else {
                Tok::Plus
            },

            '*' => if self.bump_if_equal('=') {
                Tok::Assignment(Some(Binop::Mul))
            } else {
                Tok::Splat
            },

            '/' => if self.bump_if_equal('=') {
                Tok::Assignment(Some(Binop::Div))
            } else {
                Tok::Slash
            },

            '-' => if self.bump_if_equal('>') {
                Tok::GotoArrow
            } else if self.bump_if_equal('=') {
                Tok::Assignment(Some(Binop::Sub))
            } else if self.bump_if_equal('-') {
                self.bump_while(|c| c != '\n');

                match self.depth {
                    St::Root => Tok::Comment,
                    _ => Tok::InvalidComment,
                }
            } else {
                Tok::Minus
            },

            '$' => {
                let start = self.start;
                self.bump_while(|c| c == '_' || c.is_alphabetic());
                self.bump_while(|c| c == '_' || c.is_alphabetic() || c.is_numeric());
                let name = self.text_at(start);

                if name.len() > 0 {
                    Tok::GlobalName(name.into())
                } else {
                    Tok::Unrecognized('$')
                }
            },

            '#' => {
                let start = self.start;
                self.bump_while(|c| c == '_' || c.is_alphabetic());
                self.bump_while(|c| c == '_' || c.is_alphabetic() || c.is_numeric());
                let name = self.text_at(start);

                if name.len() > 0 {
                    Tok::Atom(name.into())
                } else {
                    Tok::Unrecognized('.')
                }
            },

            '"' => {
                let mut text = String::new();

                loop {
                    let Some(c) = self.bump() else {
                        break Tok::Unrecognized('"');
                    };

                    match c {
                        '"' => break Tok::LitString(text),

                        '\\' => {
                            let Some(c) = self.bump() else {
                                break Tok::Unrecognized('\\');
                            };

                            // TODO: Non-literal escape sequences
                            text.push(c);
                        },

                        _ => text.push(c),
                    }
                }
            },

            '<' => if self.bump_if_equal('>') {
                if self.bump_if_equal('=') {
                    Tok::Assignment(Some(Binop::Stitch))
                } else {
                    Tok::Stitch
                }
            } else {
                Tok::Unrecognized('<')
            },

            '>' => {
                self.bump_while(|c| c != '\n' && c.is_whitespace());

                self.depth = St::Quote;

                Tok::QuoteStart
            },

            _ if ch.is_numeric() => {
                self.bump_while(|c| c.is_numeric());
                match self.text_at(offset).parse::<u32>() {
                    Ok(int) => Tok::LitInt(int),
                    Err(_) => Tok::InvalidInt(self.text_at(offset).into()),
                }
            },

            _ if ch.is_alphabetic() => {
                self.bump_while(|c| {
                    c == '_' || c.is_alphabetic() || c.is_numeric()
                });

                match self.text_at(offset) {
                    "global" => Tok::KwdGlobal,
                    "enum" => Tok::KwdEnum,
                    "let" => Tok::KwdLet,
                    "if" => Tok::KwdIf,
                    "else" => Tok::KwdElse,
                    "on" => Tok::KwdOn,
                    "listen" => Tok::KwdListen,
                    "menu" => Tok::KwdMenu,
                    "music" => Tok::KwdMusic,
                    "trace" => Tok::KwdTrace,
                    "continue" => Tok::KwdContinue,
                    "hibernate" => Tok::KwdHibernate,
                    "return" => Tok::KwdReturn,
                    "wait" => Tok::KwdWait,
                    "bye" => Tok::KwdBye,
                    "not" => Tok::KwdNot,
                    word => Tok::Ident(word.into()),
                }
            },

            _ if ch.is_whitespace() => {
                self.bump_while(|c| c != '\n' && c.is_whitespace());
                Tok::Whitespace
            },

            _ => Tok::Unrecognized(ch),
        };

        let content = self.text_at(offset);

        Some(Token { kind, offset, content })
    }
}

impl<'src> AsRef<str> for Token<'src> {
    fn as_ref(&self) -> &str {
        self.content
    }
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::QuoteStart => write!(f, "> "),
            Tok::QuoteVerbatim(content) => write!(f, "{content}"),
            Tok::QuoteEscape(char) => write!(f, "\\{char}"),
            Tok::LitString(s) => write!(f, "{:?}", s),
            Tok::LitInt(i) => write!(f, "{}", i),
            Tok::Atom(e) => write!(f, "#{}", e),
            Tok::Ident(w) => write!(f, "{}", w),
            Tok::GlobalName(g) => write!(f, "${}", g),
            Tok::KwdGlobal => write!(f, "global"),
            Tok::KwdEnum => write!(f, "enum"),
            Tok::KwdLet => write!(f, "let"),
            Tok::KwdIf => write!(f, "if"),
            Tok::KwdElse => write!(f, "else"),
            Tok::KwdOn => write!(f, "on"),
            Tok::KwdListen => write!(f, "listen"),
            Tok::KwdMenu => write!(f, "menu"),
            Tok::KwdMusic => write!(f, "music"),
            Tok::KwdTrace => write!(f, "trace"),
            Tok::KwdContinue => write!(f, "continue"),
            Tok::KwdHibernate => write!(f, "hibernate"),
            Tok::KwdReturn => write!(f, "return"),
            Tok::KwdWait => write!(f, "wait"),
            Tok::KwdBye => write!(f, "bye"),
            Tok::KwdNot => write!(f, "not"),
            Tok::Plus => write!(f, "+"),
            Tok::Minus => write!(f, "-"),
            Tok::Splat => write!(f, "*"),
            Tok::Slash => write!(f, "/"),
            Tok::Stitch => write!(f, "<>"),
            Tok::Less => write!(f, "?<"),
            Tok::Equal => write!(f, "?="),
            Tok::Greater => write!(f, "?>"),
            Tok::Assignment(None) => write!(f, "="),
            Tok::Assignment(Some(op)) => write!(f, "{}=", op),
            Tok::GotoArrow => write!(f, "->"),
            Tok::LabelMarker => write!(f, "::"),
            Tok::OptionPipe => write!(f, "|"),
            Tok::Semicolons => write!(f, ";;"),
            Tok::Dot => write!(f, "."),
            Tok::Ellipsis => write!(f, "..."),
            Tok::LineBreak => write!(f, ","),
            Tok::Comment | Tok::InvalidComment => write!(f, "-- ..."),
            Tok::Whitespace => write!(f, " "),
            Tok::ParenOpen => write!(f, "("),
            Tok::ParenClose => write!(f, ")"),
            Tok::CurlyOpen => write!(f, "{{"),
            Tok::CurlyClose => write!(f, "}}"),
            Tok::InvalidInt(s) => write!(f, "{}", s),
            Tok::Unrecognized(u) => write!(f, "{:?}", u),
        }
    }
}

#[test]
fn test_tokens() {
    let src = ":: word -> menu . music | return ;; , \"string literal\" > quoted {expr}";

    let tokens: Vec<_> = Lexer::new(src).skip_whitespace().map(|t| t.kind).collect();

    assert_eq!(tokens.as_slice(), &[
        Tok::LabelMarker,
        Tok::Ident("word".into()),
        Tok::GotoArrow,
        Tok::KwdMenu,
        Tok::Dot,
        Tok::KwdMusic,
        Tok::OptionPipe,
        Tok::KwdReturn,
        Tok::Semicolons,
        Tok::LineBreak,
        Tok::LitString("string literal".into()),
        Tok::QuoteStart,
        Tok::QuoteVerbatim("quoted ".into()),
        Tok::CurlyOpen,
        Tok::Ident("expr".into()),
        Tok::CurlyClose,
    ]);
}
