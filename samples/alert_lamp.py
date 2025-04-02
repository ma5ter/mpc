from _builtins import *

# SCRIPT PARAMETERS
SECTION_MASK = 0
TIMEZONE_OFFSET = 3*60*60

# CONSTS
SECTION_MAX = 24

MAX_VALUE = 100
MIN_VALUE = 10
RANGE = MAX_VALUE - MIN_VALUE

def accumulated_state() -> AlarmState:
	accumulator = AlarmState.DISARMED
	index = 0
	# iterate all sections within the global mask
	while index <= SECTION_MAX:
		# check section mask
		if (2 ** index) & SECTION_MASK:
			state = sh_section_state(index)
			# the higher section state is the more important it is
			if accumulator < state:
				accumulator = state
		index += 1
	return accumulator


def blink_cycle(on_time_ms: int, off_time_ms: int):
	if on_time_ms:
		output(Action.ACTION_ON)
		sleep(on_time_ms)
	if off_time_ms:
		output(Action.ACTION_OFF)
		sleep(off_time_ms)


def main():
	hour, minute, sec = get_realtime(TIMEZONE_OFFSET)
	# check entry and exit timers first
	if sh_get_entry_timer():
		# 1. off for 1000 ms
		blink_cycle(0, 1000)
		# 2. blink once till the entry timer is timed out
		blink_cycle(250, sh_get_entry_timer())
	elif sh_get_exit_timer():
		# 1. off for 1000 ms
		blink_cycle(0, 1000)
		# 2. blink twice till the exit timer is timed out
		blink_cycle(250, 250)
		blink_cycle(250, sh_get_exit_timer())
	# otherwise check states
	else:
		state = accumulated_state()
		if state == AlarmState.DISARMED:
			blink_cycle(0, 1000)
		if state == AlarmState.ARMED:
			blink_cycle(1000, 0)
		if state == AlarmState.ALARMING:
			blink_cycle(500, 500)


main()
