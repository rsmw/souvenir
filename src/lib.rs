pub mod ast;
pub mod token;
pub mod compile;
pub mod disas;
pub mod eval;

pub mod value;
pub mod interpret;

pub fn parse(input: &str) -> anyhow::Result<ast::Script> {
    ast::Parser {
        src: input,
        input: token::Lexer::new(input).skip_whitespace().peekable(),
        last_page: None,
    }.parse()
}

#[test]
fn compile_example() {
    let input = include_str!("../example.svr");

    let ast = parse(input).unwrap();

    assert_eq!(ast.pages.len(), 3);

    let _script = ast.compile().unwrap();
}
