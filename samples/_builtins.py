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


class RimVariableType(Enum):
	RIM_VARIABLE_BATTERY = 0
	RIM_VARIABLE_TEMPERATURE = 1
	RIM_VARIABLE_RSSI = 2
	RIM_VARIABLE_TEMPERATURE2 = 3


# noinspection PyShadowingBuiltins
def print(*x: int | Enum) -> None:
	builtins.print(*x)
	return


def sleep(milliseconds: int) -> None:
	time.sleep(milliseconds / 1000)


def output(action: Action) -> None:
	builtins.print(f"OUT: {action.name}")


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


def sh_get_zone_sensor(zone_number: int) -> int:
	"""
	Retrieves the current zone sensor mask value
	:param zone_number: number of the zone starting from 1
	:return: a bitmask of the following flags:
			- SENSOR_AFFECT_ALARM = 0
			- SENSOR_AFFECT_ATTENTION = 1
			- SENSOR_AFFECT_MALFUNCTION = 2
			- SENSOR_AFFECT_BREACH = 3
			- SENSOR_AFFECT_BATTERY = 4
			- SENSOR_AFFECT_BATTERY2 = 5
			- SENSOR_AFFECT_LOST = 6
			- SENSOR_AFFECT_PHYALARM = 7
			- SENSOR_STATE_ENTRY_DELAY = 8
			- SENSOR_STATE_EXIT_DELAY = 9
			- SENSOR_STATE_TIMEOUT = 10
			- SENSOR_STATE_LOWTEMP = 11
			- SENSOR_STATE_HIGHTEMP = 12
			- SENSOR_STATE_BYPASS = 13
	"""
	return int(input(f"Provide sensor flags state for zone {zone_number}: "))


def sh_get_zone_rim_variables(zone_number: int, variable_type: RimVariableType) -> int:
	"""
	Retrieves the current zone variables associated with a RI-M sensor
	:param zone_number: number of the zone starting from 1
	:param variable_type: one of:
			- RIM_VARIABLE_BATTERY = 0
			- RIM_VARIABLE_TEMPERATURE = 1
			- RIM_VARIABLE_RSSI = 2
			- RIM_VARIABLE_TEMPERATURE2 = 3
	:return: a variable value
	"""
	return int(input(f"Provide RI-M sensor variable {variable_type} for zone {zone_number}: "))
