WELL_KNOWN_ENUMS = {
	"AlarmState": {
		"DISARMED": 0,
		"ARMED": 1,
		"ALARMING": 2,
	},
	"Action": {
		"ACTION_OFF": 0,
		"ACTION_DISARM": 0,
		"ACTION_ON": 1,
		"ACTION_ARM": 1,
		"ACTION_EXIT": 2,
		"ACTION_ENTRY": 3,
		"ACTION_ALARM": 4,
		"ACTION_KTS_CHECK_START": 5,
		"ACTION_KTS_CHECK_SUCCESS": 6,
		"ACTION_KTS_CHECK_FAIL": 7,
		"ACTION__ONCE": 128,
	},
}
BUILTIN_FUNCTIONS = [
	('print', ['*'], 0),
	('output', ['action'], 0),
	('get_tick', [], 1),
	('get_time', [], 1),
	('get_realtime', ['timezone_offset'], 3),
	('get_date', ['timezone_offset'], 3),
	('get_weekday', ['timezone_offset'], 1),
	('sh_get_entry_timer', [], 1),
	('sh_get_exit_timer', [], 1),
	('sh_section_state', ['section_number'], 1),
]
