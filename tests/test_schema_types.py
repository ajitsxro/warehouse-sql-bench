import pytest

from sqlbench.schema.types import LogicalType, StructField, TypeKind, TypeParseError, parse_type


@pytest.mark.parametrize(
    "text",
    [
        "int",
        "bigint",
        "double",
        "string",
        "bool",
        "date",
        "timestamp_ntz",
        "timestamp_tz",
        "variant",
        "decimal(12,2)",
        "decimal(38,0)",
        "array<string>",
        "map<string,double>",
        "map<bigint,array<int>>",
        "struct<source:string,medium:string>",
        "array<struct<sku:string,tags:array<string>>>",
    ],
)
def test_round_trips_to_canonical_text(text: str) -> None:
    assert str(parse_type(text)) == text


def test_whitespace_is_ignored() -> None:
    assert str(parse_type("  map< string , decimal( 10 , 2 ) >  ")) == "map<string,decimal(10,2)>"


def test_nested_struct_structure() -> None:
    parsed = parse_type("struct<city:string,zip:array<int>>")

    assert parsed == LogicalType(
        TypeKind.STRUCT,
        fields=(
            StructField("city", LogicalType(TypeKind.STRING)),
            StructField("zip", LogicalType(TypeKind.ARRAY, element=LogicalType(TypeKind.INT))),
        ),
    )


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("text", "unknown type"),
        ("decimal(39,2)", "precision"),
        ("decimal(5,6)", "scale"),
        ("decimal", "unexpected end"),
        ("array<string", "unexpected end"),
        ("array<>", "unknown type"),
        ("map<double,string>", "map keys"),
        ("struct<a:int,a:int>", "duplicate"),
        ("struct<>", "field name"),
        ("int int", "unexpected 'int'"),
        ("array<string>!", "unexpected character"),
    ],
)
def test_invalid_types_explain_the_problem(text: str, problem: str) -> None:
    with pytest.raises(TypeParseError, match=problem):
        parse_type(text)
