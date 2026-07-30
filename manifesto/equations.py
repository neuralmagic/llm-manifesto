"""Small expression evaluator for computed spec values such as scheduler limits."""

from __future__ import annotations

import ast
import operator
from typing import Any

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def eval_expr(expr: Any, context: dict[str, Any]) -> Any:
    if not isinstance(expr, str):
        return expr
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr
    return _eval_node(tree.body, context)


def render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: render_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [render_value(item, context) for item in value]
    return eval_expr(value, context)


def render_mapping(values: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    active_context = dict(context)
    for key, value in values.items():
        rendered_value = render_value(value, active_context)
        rendered[key] = rendered_value
        active_context[key] = rendered_value
    return rendered


def _eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"unknown variable in expression: {node.id}")
        return context[node.id]
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in OPS:
            raise ValueError(f"unsupported operator in expression: {op_type.__name__}")
        return OPS[op_type](_eval_node(node.left, context), _eval_node(node.right, context))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError(f"unsupported unary operator in expression: {op_type.__name__}")
        return UNARY_OPS[op_type](_eval_node(node.operand, context))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")
