from _builtins import *

TIMEZONE_OFFSET = 3 * 60 * 60


def main():
	while True:
		year, month, date = get_date(TIMEZONE_OFFSET)
		hour, minute, second = get_realtime(TIMEZONE_OFFSET)
		weekday = get_weekday(TIMEZONE_OFFSET)
		print(year, month, date, hour, minute, second, weekday)
		sleep(1000)


main()
