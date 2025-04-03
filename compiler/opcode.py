from typing import Any

import compiler.abstract
from compiler.entity import *


class OP(compiler.abstract.OP):
	def __init__(self, opcode: int, mask: int, source: str | None):
		self.opcode = opcode
		self.mask = mask
		self.source = source
		self.metadata = None
		self.anchor: Anchor | None = None
		self.entry_point = None

	def move_anchor_to(self, target) -> None:
		if self.anchor is not None:
			self.anchor.place_to(target)
			self.anchor = None

	def get_instruction(self) -> int:
		return self.opcode

	def append_source(self, string: str) -> str:
		if self.source is None:
			return string
		return string + (' ' * (30 - len(string))) + ' ; ' + self.source

	def __str__(self):
		return self.append_source(type(self).__name__)

	@staticmethod
	def bits(value: int) -> int:
		count = 0
		while value > 0:
			value >>= 1
			count += 1
		return count


class NOP(OP):
	def __init__(self):
		super().__init__(0, 0, None)

	def get_instruction(self) -> None:
		raise RuntimeError("NOP is not a real instruction it's just an anchor placeholder and should be optimized out")


def optimize(code: list[OP]) -> list[OP]:
	result: list[OP] = []
	skip = 0
	for i in range(len(code)):
		if skip > 0:
			skip -= 1
			continue
		curr = code[i]
		nxt = code[i + 1] if i < len(code) - 1 else None
		nxt2 = code[i + 2] if i < len(code) - 2 else None
		# remove NOPs
		if isinstance(nxt, NOP):
			try:
				nxt.move_anchor_to(curr)
			except Anchor.CollisionError:
				keep = cast(Anchor, curr.anchor)
				remove = cast(Anchor, nxt.anchor)
				for c in code:
					if isinstance(c, BranchOP):
						bra = cast(BranchOP, c)
						if bra.target == remove:
							bra.target = keep
			skip = 1
		# check for increment
		elif isinstance(nxt2, ADD):
			# TODO: properly check anchors in-between
			if isinstance(curr, PSH) and cast(PSH, curr).value == 1 and isinstance(nxt, LDV):
				var = nxt
			elif isinstance(nxt, PSH) and cast(PSH, nxt).value == 1 and isinstance(curr, LDV):
				var = curr
			else:
				var = None
			if var is not None:
				result.append(var)
				curr = INC(nxt2.source)
				nxt2.move_anchor_to(curr)
				skip = 2
		# check for decrement
		# TODO: properly check anchors in-between
		elif isinstance(curr, PSH) and cast(PSH, curr).value == 1 and isinstance(nxt, LDV) and isinstance(nxt2, SUB):
			result.append(nxt)
			curr = DEC(nxt2.source)
			nxt2.move_anchor_to(curr)
			skip = 2
		# remove consecutive variable STORE-LOAD
		# TODO: need more branching and calls analysis to find another store without load
		elif isinstance(curr, STV) and isinstance(nxt, LDV) and curr.anchor is None and nxt.anchor is None:
			stv = cast(STV, curr)
			ldv = cast(STV, nxt)
			if stv.value == ldv.value:
				# optimized
				curr.metadata = 'oo'
				# skip = 1
				# continue
				pass
		result.append(curr)
	return result


def optimize_branches(code: list[OP]) -> list[OP]:
	code_size = 0
	while code_size != len(code):
		code_size = len(code)
		address = 1
		while address < len(code):
			op = code[address]
			if isinstance(op, BranchOP):
				if op.target is None:
					raise AssertionError('branch or jump target is none')
				target_address = code.index(op.target.op)
				# NOTE: targets point to the instruction after they attached
				offset = target_address + 1 - code.index(op)
				# NOTE: min positive offset is 2 (no sense to jump next op)
				if offset >= 0:
					if offset < 2:
						raise ValueError("offset too small")
					offset -= 2
				# remove previous load complements
				while (address > 0 and isinstance(code[address - 1], LoadComplementary)
				and cast(LoadComplementary, code[address - 1]).complement == op):
					address -= 1
					code.pop(address)
				# JMP has own storage and may work without complements
				if isinstance(op, JMP):
					if offset >= 0:
						if offset < op.mask:
							op.offset = offset
							address += 1
							continue
						else:
							op.offset = op.mask
							offset -= op.mask
					else:
						# replace opcode with a jump back instruction and positive offset saving one byte
						jmb = JMB(op.target, op.source)
						op.move_anchor_to(jmb)
						code[address] = op = jmb
				# add complements
				psh:List[OP] = []
				if offset < 0:
					# extend offset when negative
					prev = 0
					curr = 1
					while prev != curr:
						psh = PSH(offset - curr + 1, op.source, op).expand()
						if isinstance(op, JMB):
							psh.pop()  # remove the last NEG instruction
						prev = curr
						curr = len(psh)
				else:
					psh = PSH(offset, op.source, op).expand()
				for p in psh:
					code.insert(address, p)
					address += 1
			address += 1
	return code


class IntegralOP(OP):
	def __init__(self, opcode: int, mask: int, value: Any, source: str | None):
		IntegralOP.assert_integral(value)
		if isinstance(value, int) and not isinstance(self, PSH) and value < 0:
			raise ValueError("value out of range")
		super().__init__(opcode, mask, source)
		self.value: Any = value

	def get_value(self) -> Any:
		return self.value.get_index() if isinstance(self.value, IStackable) else self.value

	def get_instruction(self) -> int:
		value = self.get_value()
		if not isinstance(value, int):
			raise ValueError("value is not integral")
		if value < 0 or value > self.mask:
			raise ValueError("value out of range")
		return self.opcode | (value & self.mask)

	def expand(self) -> list[OP]:
		value = self.get_value()
		if value >= 0:
			if value < self.mask:
				return [self]
			value -= self.mask
		elif not isinstance(self, JMP):
			raise ValueError("negative value")
		# NOTE: only derivatives may construct OPs, ignore parameters' error
		op = self.__class__(value=self.mask, source=self.source)
		if self.anchor is not None:
			self.anchor.place_to(op)
		psh = PSH(value, self.source, op)
		psh.entry_point = self.entry_point
		return psh.expand() + [op]

	@staticmethod
	def assert_integral(value) -> int:
		if isinstance(value, int):
			return value
		if isinstance(value, IStackable):
			return value.get_index()
		raise TypeError("argument must be integral")

	def __str__(self):
		return self.append_source(f"{type(self).__name__} {self.value}")


class BranchOP(OP):
	def __init__(self, opcode: int, mask: int, target: Anchor, source: str | None):
		super().__init__(opcode, mask, source)
		self.target: Anchor = target

	def __str__(self):
		return self.append_source(f"{type(self).__name__} {self.target}")


class LoadComplementary:
	def __init__(self, complement: OP):
		self.complement: OP = complement


class PSH(IntegralOP, LoadComplementary):
	def __init__(self, value: Any, source: str | None = None, complement: OP = None):
		LoadComplementary.__init__(self, complement)
		IntegralOP.__init__(self, 0, 0x7F, value, source)

	def expand(self) -> list[OP]:
		value = self.get_value()
		if value >= 0:
			if value < 0x80:
				return [self]
			sign = 1
		else:
			sign = -1
			value *= sign
		# generate instructions when overflowed or negative
		count = (OP.bits(value) - 7 + 4) // 5
		# for small negative values count will be calculated as -1
		if count < 0:
			count = 0
		psh = PSH((value >> (count * 5)) & 0x7F, self.source, self.complement)
		psh.entry_point = self.entry_point
		result = [psh]
		while count > 0:
			count -= 1
			result.append(PSC((value >> (count * 5)) & 0x1F, self.source, self.complement))
		if sign < 0:
			result.append(NEG(self.source, self.complement))
		if self.anchor is not None:
			self.anchor.place_to(result[-1])
		return result


class PSC(OP, LoadComplementary):
	def __init__(self, value: int, source: str | None = None, complement: OP = None):
		LoadComplementary.__init__(self, complement)
		OP.__init__(self, 0x80, 0x1F, source)
		self.value: int = value

	def get_instruction(self) -> int:
		value = self.value
		if not isinstance(value, int):
			raise ValueError("value is not integral")
		if value < 0 or value > self.mask:
			raise ValueError("value out of range")
		return self.opcode | (value & self.mask)

	def __str__(self):
		return self.append_source(f"PSC {self.value}")


class POP(OP):
	def __init__(self, count: int, source: str | None = None):
		super().__init__(0b10111100, 3, source)
		if count < 1 or count > self.mask + 1:
			raise ValueError("count out of range")
		self.count: int = count

	def get_instruction(self) -> int:
		count = self.count
		if not isinstance(count, int):
			raise ValueError("count is not integral")
		if count < 0 or count > self.mask:
			raise ValueError("count out of range")
		return self.opcode | ((count - 1) & self.mask)

	def __str__(self):
		return self.append_source(f"POP {self.count}")


# LOAD and STORE OPERATIONS

class LDC(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10110110, 0, source)


class LDV(IntegralOP):
	def __init__(self, value: Any, source: str | None = None):
		super().__init__(0b11100000, 0xF, value, source)


class STV(IntegralOP):
	def __init__(self, value: Any, source: str | None = None):
		super().__init__(0b11110000, 0xF, value, source)


# CONTROL FLOW OPERATIONS

class CAL(IntegralOP):
	def __init__(self, value: Any, source: str | None = None):
		super().__init__(0b11010000, 0xF, value, source)


class RET(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10110101, 0, source)


class SKZ(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10110000, 0, source)


class SNZ(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10110001, 0, source)


class SKN(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10110010, 0, source)


class SNN(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b1011011, 0, source)


class SLP(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10110100, 0, source)


class JMP(BranchOP):
	def __init__(self, target: Anchor, source: str | None = None):
		super().__init__(0b11000000, 0xF, target, source)
		self.offset = self.mask

	def get_instruction(self) -> int:
		if self.offset is None or self.offset <= 0:
			raise RuntimeError("offset not set by optimizer")
		if self.offset > self.mask:
			raise RuntimeError("offset is too large")
		return self.opcode | self.offset

	def __str__(self):
		return self.append_source(f"{type(self).__name__} {self.target} <{self.offset}>")


class JMB(BranchOP):
	def __init__(self, target: Anchor, source: str | None = None):
		super().__init__(0b10110111, 0, target, source)


# COMPARE-BRANCH OPERATIONS

class BZE(BranchOP):
	# pop and branch if zero
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100000, 0, anchor, source)


class BNZ(BranchOP):
	# pop and branch if zero
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100001, 0, anchor, source)


class BEQ(BranchOP):
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100010, 0, anchor, source)


class BNE(BranchOP):
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100011, 0, anchor, source)


class BGT(BranchOP):
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100100, 0, anchor, source)


class BLT(BranchOP):
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100101, 0, anchor, source)


class BGE(BranchOP):
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100110, 0, anchor, source)


class BLE(BranchOP):
	def __init__(self, anchor: Anchor, source: str | None = None):
		super().__init__(0b10100111, 0, anchor, source)


# BINARY OPERATIONS

class ADD(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101000, 0, source)


class SUB(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101001, 0, source)


class MUL(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101010, 0, source)


class DIV(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101011, 0, source)


class PWR(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101100, 0, source)


class AND(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101101, 0, source)


class IOR(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101110, 0, source)


class XOR(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10101111, 0, source)


# UNARY OPERATIONS

class NEG(OP, LoadComplementary):
	def __init__(self, source: str | None = None, complement: OP = None):
		LoadComplementary.__init__(self, complement)
		OP.__init__(self, 0b10111000, 0, source)


class INV(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10111001, 0, source)


class INC(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10111010, 0, source)


class DEC(OP):
	def __init__(self, source: str | None = None):
		super().__init__(0b10111011, 0, source)

