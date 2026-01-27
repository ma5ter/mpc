from _builtins import *

# SCRIPT PARAMETERS
SECTION_MASK = 0  # Выбираются отслеживаемые разделы. Т.е. разделы, при взятии которых будет вкллючаться маяк и при тревоге из разделов будет включаться индикация тревоги на маяке.
MAX_TIME = 0  # Продолжительность работы. Устанавливается время работы индикации тревоги в секундах.
SHOW_DELAYS = 0  # Индикация задержки. Т.е. можно включать(=1)/выключать(=0) индикацию при задержке на вход/выход.

# CONSTS
SECTION_MAX = 24


def accumulated_state() -> AlarmState:
	# Calculates the highest alarm state among the sections specified by SECTION_MASK.
	#
	# This function iterates through all possible section indices up to SECTION_MAX.
	# For each index, it checks if the corresponding section is included in the global
	# SECTION_MASK. If a section is masked, its current alarm state is retrieved,
	# and the function keeps track of the highest (most critical) state found so far.
	#
	# Returns:
	# 	AlarmState: The highest alarm state found among the monitored sections.
	# 				Returns AlarmState.DISARMED if no monitored sections are in a higher state.
	accumulator = AlarmState.DISARMED  # Initialize accumulator with the lowest alarm state (disarmed).
	index = 0
	# Iterate through all possible section indices from 0 up to SECTION_MAX.
	while index <= SECTION_MAX:
		# Check if the current section (represented by 2^index) is part of the SECTION_MASK.
		# This bitwise AND operation determines if the section is selected for monitoring.
		if (2 ** index) & SECTION_MASK:
			# If the section is monitored, get its current alarm state.
			state = sh_get_section_state(index)
			# Compare the current section's state with the accumulated highest state.
			# The higher the AlarmState enum value, the more critical the state.
			if accumulator < state:
				# Update accumulator if a more critical state is found.
				accumulator = state
		index += 1
	# Return the highest alarm state encountered.
	return accumulator


def main():
	# Initialize the previous action to 'OFF' to ensure the lamp starts in an off state.
	previous_action = Action.ACTION_OFF
	# Initialize 'time_off'.
	# This variable will store the time until the lamp should indicate alarm, based on MAX_TIME.
	time_off = get_time()

	# Main loop for continuous operation of the alert lamp.
	while True:
		action = Action.ACTION_OFF

		# First check the panic timer
		if sh_get_panic_timer() > 0:
			state = AlarmState.ALARMING
		else:
			# Determine the aggregated state of the monitored sections.
			state = accumulated_state()

		# If any monitored section is actually alarming, use the 'ALARM' action, but limit the overall time.
		if state == AlarmState.ALARMING:
			if time_off + MAX_TIME > get_time():
				action = Action.ACTION_ALARM
		else:
			# When not alarming update maximum alarm time
			time_off = get_time()
			# If SHOW_DELAYS is set to 0, suppress the display of entry/exit delays.
			if SHOW_DELAYS:
				# Get the current alarm timer status to determine entry & exit delays
				# When no timers active this will return Action.ACTION_OFF (armed or disarmed)
				_, action = sh_get_alarm_timer()

		# If no action was set yet, check that it is armed
		if action == Action.ACTION_OFF:
			if state == AlarmState.ARMED:
				action = Action.ACTION_ARM

		# Check if the current determined action is different from the previously recorded action.
		if previous_action != action:
			# Update the 'previous_action' to the new action.
			previous_action = action
			# The lamp should reflect the current 'action'.
			output(action)


main()
