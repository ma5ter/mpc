"""
Module for resolving constant values in the compiler.
This module provides functionality to evaluate constant expressions at compile time,
which allows for optimizations and early error detection.
"""

import sys
from typing import cast

from compiler.error import *
from compiler.entity import NamedConstant


class SimpleConstantResolver(ast.NodeVisitor):
	"""
	AST visitor for resolving constant expressions at compile time.
	This class evaluates expressions that can be determined at compile time,
	allowing for optimizations and early error detection.
	"""

	class Error(CompilationError):
		"""
		Exception raised when a constant expression cannot be resolved.
		This is typically due to unsupported operations or non-constant values.
		"""

		def __init__(self, node: ast.AST, message: str):
			"""
			Initialize the error with the AST node and error message.

			Args:
				node: The AST node where the error occurred.
				message: Description of the error.
			"""
			super().__init__(node, "can't resolve value as constant due to " + message)

	def __init__(self, node: ast.AST, hint: dict[str, NamedConstant] | None = None):
		"""
		Initialize the constant resolver with the AST node to evaluate.

		Args:
			node: The AST node representing the expression to evaluate.
			hint: Optional dictionary of named constants that may be referenced.
		"""
		super().__init__()
		self.hint = hint  # Dictionary of named constants that can be referenced
		self.result: int = self.visit(node)  # The resolved constant value

	def generic_visit(self, node: ast.AST) -> None:
		"""
		Handle unsupported AST nodes by raising an error.

		Args:
			node: The AST node that is not supported.

		Raises:
			SimpleConstantResolver.Error: Always raised for unsupported nodes.
		"""
		raise SimpleConstantResolver.Error(node, f"unexpected AST node {node.__class__.__name__} visited")

	def visit_Constant(self, node: ast.Constant) -> int:
		"""
		Visit a Constant node in the AST and return its value.

		Args:
			node: The Constant node to visit.

		Returns:
			The integer value of the constant.

		Raises:
			SimpleConstantResolver.Error: If the constant is not an integer.
		"""
		if not isinstance(node.value, int):
			raise SimpleConstantResolver.Error(node, f"{node.value.__class__.__name__} is not an int")
		return node.value

	def visit_Name(self, node: ast.Name) -> int:
		"""
		Visit a Name node in the AST and resolve it to a constant value.

		Args:
			node: The Name node to visit.

		Returns:
			The integer value of the named constant.

		Raises:
			SimpleConstantResolver.Error: If no hints are provided or the name is not found.
		"""
		if self.hint is None or node.id not in self.hint:
			raise SimpleConstantResolver.Error(node, f"no hints provided for {node.id} resolution")
		const = self.hint[node.id].const
		if not isinstance(const, int):
			raise SimpleConstantResolver.Error(node, f"named const {node.id} is not an int")
		return const

	def visit_BinOp(self, node: ast.BinOp) -> int:
		"""
		Visit a BinOp node in the AST and evaluate the binary operation.

		Args:
			node: The BinOp node to visit.

		Returns:
			The result of the binary operation.

		Raises:
			NotImplementedError: If the operation type is not supported.
			SimpleConstantResolver.Error: If the operation is not supported.
		"""
		# Evaluate left and right operands
		left = self.visit(cast(ast.AST, node.left))
		right = self.visit(cast(ast.AST, node.right))

		# Handle different binary operation types
		if isinstance(node.op, ast.Add):
			return left + right
		if isinstance(node.op, ast.Sub):
			return left - right
		if isinstance(node.op, ast.Mult):
			return left * right
		if isinstance(node.op, ast.FloorDiv):
			return left // right
		if isinstance(node.op, ast.Pow):
			return left ** right
		if isinstance(node.op, ast.BitAnd):
			return left & right
		if isinstance(node.op, ast.BitOr):
			return left | right
		if isinstance(node.op, ast.BitXor):
			return left ^ right
		if isinstance(node.op, ast.LShift):
			return left << right
		if isinstance(node.op, ast.RShift):
			return left << right
		if isinstance(node.op, ast.Div):
			# Handle division with potential precision loss
			result = left / right
			int_result = int(result)
			if result != int_result:
				# TODO: implement common warning and error reporting
				print(f"warning: floating point division cast to int {result} -> {int_result}, precision loss", file=sys.stderr)
			return int_result

		# Raise error for unsupported operations
		raise NotImplementedError

	def visit_Assign(self, node):
		"""
		Visit an Assign node in the AST.
		This method does nothing as assignments are not part of constant expressions.

		Args:
			node: The Assign node to visit.
		"""
		pass

	@staticmethod
	def resolve(node: ast.AST, hint: dict[str, NamedConstant] | None = None) -> int | None:
		"""
		Static method to resolve a constant expression.

		Args:
			node: The AST node representing the expression to evaluate.
			hint: Optional dictionary of named constants that may be referenced.

		Returns:
			The resolved constant value if successful, None otherwise.
		"""
		try:
			return SimpleConstantResolver(node, hint).result
		except SimpleConstantResolver.Error:
			return None
