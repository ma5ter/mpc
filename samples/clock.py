from _builtins import *

def main():
	year, month, date = get_date()
	hour, minute, second = get_realtime()
	weekday = get_weekday()
	print(year, month, date, hour, minute, second, weekday + 1)
	sleep(1000)

main()