from enum import Enum


def pp(*args):
	print(*args)


class ConsoleColors(Enum):
	Reset = '\033[0m'
	Red = '\u001B[31m'
	Green = '\u001B[32m'
	Yellow = '\u001B[33m'
	Blue = '\u001B[34m'
	Magenta = '\u001B[35m'
	Cyan = '\u001B[36m'


def highlight(string: str, color: ConsoleColors = ConsoleColors.Cyan) -> str:
	return color.value + string + ConsoleColors.Reset.value
