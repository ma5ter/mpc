"""
Module containing concrete implementations of compiler entities and operations.
This module defines various types of entities used in the compiler, including:
- Basic entities with reference counting
- Anchors for code positioning
- Named and unnamed entities
- Stackable entities for the virtual machine
- Variables and constants
- Function definitions with preprocessing capabilities
"""

import ast
import abc
from typing import List, Dict, cast

import compiler.abstract


class Entity(compiler.abstract.Entity):
	"""
	Base class for all compiler entities with reference counting.
	Provides basic reference counting functionality.
	"""

	def __init__(self):
		"""
		Initialize an entity with a reference count of 1.
		"""
		self.refcount: int = 1

	def incref(self) -> None:
		"""
		Increment the reference count of this entity.
		"""
		self.refcount += 1
		return


class Anchor(Entity):
	"""
	Represents a code anchor point in the compiler.
	Anchors are used to mark specific positions in the code for jumps and labels.
	"""
	count = 0  # Class variable to track anchor creation count

	class CollisionError(Exception):
		"""
		Exception raised when attempting to place an anchor on an operation
		that already has an anchor.
		"""

		def __init__(self) -> None:
			"""
			Initialize the collision error with a default message.
			"""
			super().__init__()

	def __init__(self) -> None:
		"""
		Initialize an anchor with a unique serial number.
		Each anchor gets a unique identifier based on creation order.
		"""
		super().__init__()
		self.op: compiler.abstract.OP | None = None  # Reference to the operation this anchor points to
		Anchor.count += 1
		self.serial: int = Anchor.count  # Unique identifier for this anchor

	def place_to(self, op: compiler.abstract.OP):
		"""
		Place this anchor on the specified operation.

		Args:
			op: The operation to anchor to.

		Raises:
			CollisionError: If the operation already has an anchor.
		"""
		self.op = op
		if op.anchor is not None:
			raise Anchor.CollisionError()
		op.anchor = self

	def __str__(self):
		"""
		Return a string representation of the anchor.
		"""
		return f"label_{self.serial}"


class Named(Entity):
	"""
	Base class for named entities in the compiler.
	Provides a name attribute for entities that need identification.
	"""

	def __init__(self, name: str) -> None:
		"""
		Initialize a named entity with the given name.

		Args:
			name: The name of the entity.
		"""
		super().__init__()
		self.name: str = name


class IStackable(abc.ABC):
	"""
	Interface for stackable entities in the virtual machine.
	Defines the contract for entities that can be placed on the VM stack.
	"""

	@abc.abstractmethod
	def get_index(self) -> int:
		"""
		Get the stack index of this entity.

		Returns:
			The stack index of the entity.
		"""
		pass


class Stackable(Entity, IStackable):
	"""
	Base class for entities that can be placed on the virtual machine stack.
	Provides stack index management functionality.
	"""

	def __init__(self) -> None:
		"""
		Initialize a stackable entity with no initial stack index.
		"""
		super().__init__()
		self.index: int | None = None  # Stack index, None if not on stack

	def get_index(self) -> int:
		"""
		Get the stack index of this entity.

		Returns:
			The stack index of the entity.
		"""
		return self.index

	def __str__(self):
		"""
		Return a string representation of the stackable entity.
		"""
		return f"{type(self).__name__} <{self.get_index()}>"


class Variable(Named, Stackable):
	"""
	Represents a variable in the compiler.
	Variables can be simple or tuple variables based on their usage.
	"""

	def __init__(self, name: str) -> None:
		"""
		Initialize a variable with the given name.

		Args:
			name: The name of the variable.
		"""
		self.tuple = 0  # Size of a tuple variable (1 = simple, 2+ = tuple)
		Named.__init__(self, name)
		Stackable.__init__(self)

	def __str__(self):
		"""
		Return a string representation of the variable.
		Differentiates between simple variables and tuple variables.
		"""
		return f"{'var' if self.tuple < 2 else 'tuple'} {self.name} <{self.get_index()}>"


class UnnamedConstant(Stackable):
	"""
	Represents an unnamed constant value in the compiler.
	These constants are typically used for immediate values in operations.
	"""

	def __init__(self, value: int) -> None:
		"""
		Initialize an unnamed constant with the given value.

		Args:
			value: The constant value.
		"""
		super().__init__()
		self.value: int = value
		self.refcount: int = 0  # Constants typically have special reference counting

	def __str__(self):
		"""
		Return a string representation of the constant.
		"""
		return f"{type(self.value).__name__} = {self.value} <{self.get_index()}>"


class NamedConstant(Named, IStackable):
	"""
	Represents a named constant in the compiler.
	Named constants can reference either unnamed constants or direct values.
	"""

	def __init__(self, name: str, const: UnnamedConstant | int) -> None:
		"""
		Initialize a named constant with the given name and value.

		Args:
			name: The name of the constant.
			const: The constant value, either as an UnnamedConstant or direct value.
		"""
		Named.__init__(self, name)
		self.const: UnnamedConstant | int = const

	def incref(self) -> None:
		"""
		Increment the reference count of this constant.
		If the constant references an unnamed constant, increment its reference count too.
		"""
		super().incref()
		if isinstance(self.const, UnnamedConstant):
			self.const.incref()

	def get_index(self) -> int:
		"""
		Get the stack index of this constant.

		Returns:
			The stack index of the constant.

		Raises:
			RuntimeError: If the constant doesn't reference an unnamed constant.
		"""
		if not isinstance(self.const, UnnamedConstant):
			raise RuntimeError("no index")
		return self.const.get_index()

	def __str__(self):
		"""
		Return a string representation of the named constant.
		"""
		return f"const {self.name} = {self.const.value} <{self.get_index()}>"


class Function(Named, Stackable):
	"""
	Represents a function in the compiler.
	Functions can be either user-defined or system library functions.
	"""

	class Preprocessor(ast.NodeVisitor):
		"""
		AST preprocessor for function analysis.
		Analyzes function blocks to determine variables and return types.
		"""

		def __init__(self, nodes: List[ast.AST]) -> None:
			"""
			Initialize the preprocessor with the AST nodes to analyze.

			Args:
				nodes: List of AST nodes representing the function body.
			"""
			super().__init__()
			self.returns: int | None = None  # Number of return values (None if not determined)
			self.variables: Dict[str, Variable] = {}  # Dictionary of variables found in the function
			for node in nodes:
				self.visit(node)

		def visit_Name(self, node: ast.Name):
			"""
			Visit a Name node in the AST.
			Tracks variable declarations and usage.

			Args:
				node: The Name node to visit.
			"""
			name = node.id
			if isinstance(node.ctx, ast.Store) and name not in self.variables:
				# If this is a variable declaration (Store context) and we haven't seen it before
				self.variables[name] = Variable(name)

		def visit_Return(self, node: ast.Return):
			"""
			Visit a Return node in the AST.
			Determines the return type of the function.

			Args:
				node: The Return node to visit.

			Raises:
				SyntaxError: If multiple return types are detected.
			"""
			value = node.value
			if value is None:
				# No return value (implicit return None)
				result = 0
			elif isinstance(value, ast.Tuple):
				# Tuple return value - count the elements
				result = len(cast(ast.Tuple, value).elts)
			else:
				# Single value return
				result = 1

			# Check for consistent return types
			if self.returns is None:
				self.returns = result
			elif result != self.returns:
				raise SyntaxError('Multiple return type')

		def get_variables(self) -> Dict[str, Variable]:
			"""
			Get the variables found in the function.

			Returns:
				Dictionary of variable names to Variable objects.
			"""
			return self.variables

	def __init__(self, name: str, params: List[str], block: List[ast.AST] | int | None = None, sys_lib_index: int | None = None) -> None:
		"""
		Initialize a function with the given name, parameters, and body.

		Args:
			name: The name of the function.
			params: List of parameter names.
			block: Either a list of AST nodes representing the function body,
				   an integer representing the number of return values,
				   or None for system library functions.
			sys_lib_index: Index of the function in the system library, if applicable.
		"""
		Named.__init__(self, name)
		Stackable.__init__(self)
		# Function code is generated later, not upon construction
		self.code: List[compiler.abstract.OP] | None = None
		self.address: int | None = None  # Memory address of the function code
		self.anchors: List[Anchor] = []  # List of anchors in this function
		self.refcount: int = 0  # Reference count for the function

		# Handle variadic functions (functions with *args)
		if len(params) > 0 and params[-1] == '*':
			self.params: List[str] = params[:-1]
			self.variadic: bool = True
		else:
			self.params: List[str] = params
			self.variadic: bool = False

		# Initialize parameter variables
		self.variables: Dict[str, Variable] = {x: Variable(x) for x in self.params}

		# Handle different initialization cases
		if isinstance(block, int):
			# Function with known return count (typically system library functions)
			self.returns: int | None = block
			self.block: List[ast.AST] | int | None = None
		else:
			# User-defined function with AST block
			self.returns: int | None = None
			self.block: List[ast.AST] | int | None = block

		self.sys_lib_index = sys_lib_index

		# Process the function block if provided
		if self.block is not None:
			preprocessor = self.Preprocessor(cast(List[ast.AST], self.block))
			# Add variables found in the block to our variables dictionary
			for name in preprocessor.get_variables():
				if name not in self.variables:
					self.variables[name] = Variable(name)
			self.returns = preprocessor.returns

	def get_params_count(self) -> int:
		"""
		Get the number of parameters for this function.

		Returns:
			The number of parameters.
		"""
		return len(self.params)

	def __str__(self):
		"""
		Return a string representation of the function.
		"""
		return f"func {self.name}({', '.join(self.params)}) <{self.get_index()}>"
