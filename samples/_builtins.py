import builtins
import time
from enum import Enum
from typing import Tuple


class AlarmState(Enum):
	DISARMED = 0
	ARMED = 1
	ALARMING = 2


class Action(Enum):
	ACTION_OFF = 0
	ACTION_DISARM = ACTION_OFF
	ACTION_ON = 1
	ACTION_ARM = ACTION_ON
	ACTION_EXIT = 2
	ACTION_ENTRY = 3
	ACTION_ALARM = 4
	ACTION_KTS_CHECK_START = 5
	ACTION_KTS_CHECK_SUCCESS = 6
	ACTION_KTS_CHECK_FAIL = 7
	ACTION__ONCE = 0x80


# noinspection PyShadowingBuiltins
def print(*x: int | Enum) -> None:
	builtins.print(*x)
	return


def sleep(milliseconds: int) -> None:
	time.sleep(milliseconds / 1000)


def output(action: Action) -> None:
	builtins.print(f"OUT: {action.name}")


# noinspection PyShadowingBuiltins
def input() -> int:
	return 0


def get_tick() -> int:
	return 0


def get_time() -> int:
	return 0


def get_realtime(timezone_offset: int) -> Tuple[int, int, int]:
	return 0, 0, 0


def get_date(timezone_offset: int) -> Tuple[int, int, int]:
	return 0, 0, 0


def get_weekday(timezone_offset: int) -> int:
	return 0


def sh_get_alarm_timer() -> Tuple[int, Action]:
	"""
	Retrieves the current alarm timer value and its associated action.

	:return: A tuple containing:
	         - int: The alarm timer seconds to timeout or 0 if not active.
	         - Action: The action associated with the alarm, values are:
			   - Action.ACTION_OFF: No action/timer inactive.
			   - Action.ACTION_EXIT: Exit delay timer active.
			   - Action.ACTION_ENTRY: Entry delay timer active.
			   - Action.ACTION_ALARM: Alarm timer active.
	"""
	return 0, Action.ACTION_ALARM


def sh_get_panic_timer() -> int:
	return 0


def sh_get_section_state(section_number: int) -> AlarmState:
	return AlarmState(input(f"Provide section state for section {section_number}: "))
