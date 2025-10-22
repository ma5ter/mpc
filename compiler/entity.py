import ast
import abc
from typing import List, Dict, cast

import compiler.abstract


class Entity(compiler.abstract.Entity):
	def __init__(self):
		self.refcount: int = 1

	def incref(self) -> None:
		self.refcount += 1
		return


class Anchor(Entity):
	count = 0

	class CollisionError(Exception):
		def __init__(self) -> None:
			super().__init__()

	def __init__(self) -> None:
		super().__init__()
		self.op: compiler.abstract.OP | None = None
		Anchor.count += 1
		self.serial: int = Anchor.count

	def place_to(self, op):
		self.op = op
		if op.anchor is not None:
			raise Anchor.CollisionError()
		op.anchor = self

	def __str__(self):
		return f"label_{self.serial}"


class Named(Entity):
	def __init__(self, name: str) -> None:
		super().__init__()
		self.name: str = name


class IStackable(abc.ABC):
	@abc.abstractmethod
	def get_index(self) -> int:
		pass


class Stackable(Entity, IStackable):
	def __init__(self) -> None:
		super().__init__()
		self.index: int | None = None

	def get_index(self) -> int:
		return self.index

	def __str__(self):
		return f"{type(self).__name__} <{self.get_index()}>"


class Variable(Named, Stackable):
	def __init__(self, name: str) -> None:
		self.tuple = 0
		Named.__init__(self, name)
		Stackable.__init__(self)

	def __str__(self):
		return f"{'var' if self.tuple < 2 else 'tuple'} {self.name} <{self.get_index()}>"


class UnnamedConstant(Stackable):
	def __init__(self, value: int) -> None:
		super().__init__()
		self.value: int = value
		self.refcount: int = 0

	def __str__(self):
		return f"{type(self.value).__name__} = {self.value} <{self.get_index()}>"


class NamedConstant(Named, IStackable):
	def __init__(self, name: str, const: UnnamedConstant | int) -> None:
		Named.__init__(self, name)
		self.const: UnnamedConstant | int = const

	def incref(self) -> None:
		super().incref()
		if isinstance(self.const, UnnamedConstant):
			self.const.incref()

	def get_index(self) -> int:
		if not isinstance(self.const, UnnamedConstant):
			raise RuntimeError("no index")
		return self.const.get_index()

	def __str__(self):
		return f"const {self.name} = {self.const.value} <{self.get_index()}>"


class Function(Named, Stackable):
	class Preprocessor(ast.NodeVisitor):
		def __init__(self, nodes: List[ast.AST]) -> None:
			super().__init__()
			self.returns: int | None = None
			self.variables: Dict[str, Variable] = {}
			for node in nodes:
				self.visit(node)

		def visit_Name(self, node: ast.Name):
			name = node.id
			if isinstance(node.ctx, ast.Store) and name not in self.variables:
				self.variables[name] = Variable(name)

		def visit_Return(self, node: ast.Return):
			value = node.value
			if value is None:
				result = 0
			elif isinstance(value, ast.Tuple):
				result = len(cast(ast.Tuple, value).elts)
			else:
				result = 1
			if self.returns is None:
				self.returns = result
			elif result != self.returns:
				raise SyntaxError('Multiple return type')

		def get_variables(self) -> Dict[str, Variable]:
			return self.variables

	def __init__(self, name: str, params: List[str], block: List[ast.AST] | int | None = None, sys_lib_index: int | None = None) -> None:
		Named.__init__(self, name)
		Stackable.__init__(self)
		# start counting only from the first call, not upon construction
		self.code: List[compiler.abstract.OP] | None = None
		self.address: int | None = None
		self.anchors: List[Anchor] = []
		self.refcount: int = 0
		if len(params) > 0 and params[-1] == '*':
			self.params: List[str] = params[:-1]
			self.variadic: bool = True
		else:
			self.params: List[str] = params
			self.variadic: bool = False
		self.variables: Dict[str, Variable] = {x: Variable(x) for x in self.params}
		if isinstance(block, int):
			self.returns: int | None = block
			self.block: List[ast.AST] | int | None = None
		else:
			self.returns: int | None = None
			self.block: List[ast.AST] | int | None = block
		self.sys_lib_index = sys_lib_index
		if self.block is not None:
			preprocessor = self.Preprocessor(cast(List[ast.AST], self.block))
			for name in preprocessor.get_variables():
				if name not in self.variables:
					self.variables[name] = Variable(name)
			self.returns = preprocessor.returns

	def get_params_count(self) -> int:
		return len(self.params)

	def __str__(self):
		return f"func {self.name}({', '.join(self.params)}) <{self.get_index()}>"
