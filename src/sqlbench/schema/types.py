"""Logical column types, written in YAML as strings like "decimal(12,2)" or "array<string>"."""

import re
from dataclasses import dataclass
from enum import StrEnum


class TypeKind(StrEnum):
    INT = "int"
    BIGINT = "bigint"
    DECIMAL = "decimal"
    DOUBLE = "double"
    STRING = "string"
    BOOL = "bool"
    DATE = "date"
    TIMESTAMP_NTZ = "timestamp_ntz"
    TIMESTAMP_TZ = "timestamp_tz"
    VARIANT = "variant"
    ARRAY = "array"
    STRUCT = "struct"
    MAP = "map"


INTEGER_KINDS = frozenset({TypeKind.INT, TypeKind.BIGINT})
NUMERIC_KINDS = frozenset({TypeKind.INT, TypeKind.BIGINT, TypeKind.DECIMAL, TypeKind.DOUBLE})
TEMPORAL_KINDS = frozenset({TypeKind.DATE, TypeKind.TIMESTAMP_NTZ, TypeKind.TIMESTAMP_TZ})
NESTED_KINDS = frozenset({TypeKind.VARIANT, TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.MAP})
MAP_KEY_KINDS = frozenset({TypeKind.STRING, TypeKind.INT, TypeKind.BIGINT})
# Types written as a bare word, with no parameters.
SIMPLE_KINDS = frozenset(
    {
        TypeKind.INT,
        TypeKind.BIGINT,
        TypeKind.DOUBLE,
        TypeKind.STRING,
        TypeKind.BOOL,
        TypeKind.DATE,
        TypeKind.TIMESTAMP_NTZ,
        TypeKind.TIMESTAMP_TZ,
        TypeKind.VARIANT,
    }
)

MAX_DECIMAL_PRECISION = 38


@dataclass(frozen=True)
class StructField:
    name: str
    type: "LogicalType"


@dataclass(frozen=True)
class LogicalType:
    kind: TypeKind
    precision: int | None = None
    scale: int | None = None
    # Array element type, or map value type.
    element: "LogicalType | None" = None
    key: "LogicalType | None" = None
    fields: tuple[StructField, ...] = ()

    def __str__(self) -> str:
        if self.kind == TypeKind.DECIMAL:
            return f"decimal({self.precision},{self.scale})"
        if self.kind == TypeKind.ARRAY:
            return f"array<{self.element}>"
        if self.kind == TypeKind.MAP:
            return f"map<{self.key},{self.element}>"
        if self.kind == TypeKind.STRUCT:
            parts = [f"{field.name}:{field.type}" for field in self.fields]
            return f"struct<{','.join(parts)}>"
        return self.kind.value


class TypeParseError(ValueError):
    """Raised when a type string is not a valid logical type."""


TOKEN_PATTERN = re.compile(r"\s*([a-z_][a-z0-9_]*|\d+|[<>(),:])")


def tokenize(text: str) -> list[str]:
    text = text.strip()
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = TOKEN_PATTERN.match(text, position)
        if match is None:
            raise TypeParseError(f"unexpected character {text[position]!r} in type {text!r}")
        tokens.append(match.group(1))
        position = match.end()
    return tokens


class TypeParser:
    """Recursive-descent parser over the tokens of one type string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = tokenize(text)
        self.position = 0

    def fail(self, problem: str) -> TypeParseError:
        return TypeParseError(f"{problem} in type {self.text!r}")

    def next_token(self) -> str:
        if self.position >= len(self.tokens):
            raise self.fail("unexpected end")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def expect(self, expected: str) -> None:
        token = self.next_token()
        if token != expected:
            raise self.fail(f"expected {expected!r}, got {token!r}")

    def expect_int(self) -> int:
        token = self.next_token()
        if not token.isdigit():
            raise self.fail(f"expected a number, got {token!r}")
        return int(token)

    def parse_type(self) -> LogicalType:
        word = self.next_token()
        try:
            kind = TypeKind(word)
        except ValueError:
            raise self.fail(f"unknown type {word!r}") from None

        if kind in SIMPLE_KINDS:
            return LogicalType(kind)
        if kind == TypeKind.DECIMAL:
            return self.parse_decimal()
        if kind == TypeKind.ARRAY:
            self.expect("<")
            element = self.parse_type()
            self.expect(">")
            return LogicalType(kind, element=element)
        if kind == TypeKind.MAP:
            return self.parse_map()
        return self.parse_struct()

    def parse_decimal(self) -> LogicalType:
        self.expect("(")
        precision = self.expect_int()
        self.expect(",")
        scale = self.expect_int()
        self.expect(")")
        if not 1 <= precision <= MAX_DECIMAL_PRECISION:
            raise self.fail(f"decimal precision must be 1 to {MAX_DECIMAL_PRECISION}")
        if scale > precision:
            raise self.fail("decimal scale cannot exceed precision")
        return LogicalType(TypeKind.DECIMAL, precision=precision, scale=scale)

    def parse_map(self) -> LogicalType:
        self.expect("<")
        key = self.parse_type()
        if key.kind not in MAP_KEY_KINDS:
            raise self.fail(f"map keys must be string, int or bigint, not {key}")
        self.expect(",")
        value = self.parse_type()
        self.expect(">")
        return LogicalType(TypeKind.MAP, key=key, element=value)

    def parse_struct(self) -> LogicalType:
        self.expect("<")
        fields: list[StructField] = []
        seen_names: set[str] = set()
        while True:
            name = self.next_token()
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                raise self.fail(f"bad struct field name {name!r}")
            if name in seen_names:
                raise self.fail(f"duplicate struct field {name!r}")
            seen_names.add(name)
            self.expect(":")
            fields.append(StructField(name, self.parse_type()))
            separator = self.next_token()
            if separator == ">":
                break
            if separator != ",":
                raise self.fail(f"expected ',' or '>', got {separator!r}")
        return LogicalType(TypeKind.STRUCT, fields=tuple(fields))


def parse_type(text: str) -> LogicalType:
    parser = TypeParser(text)
    logical_type = parser.parse_type()
    if parser.position != len(parser.tokens):
        raise parser.fail(f"unexpected {parser.tokens[parser.position]!r}")
    return logical_type
