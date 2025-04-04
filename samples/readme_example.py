from _builtins import *

TIMEZONE_OFFSET = 5 * 60 * 60


def main():
    default_state = Action.ACTION_OFF
    while True:
        year, month, date = get_date(TIMEZONE_OFFSET)
        hour, minute, second = get_realtime(TIMEZONE_OFFSET)
        weekday = get_weekday(TIMEZONE_OFFSET)
        print(year, month, date, hour, minute, second, weekday)
        state = Action.ACTION_OFF
        if weekday >= 1:
            if weekday <= 5:
                # turn on a bound output for weekdays
                output(Action.ACTION_ON)
        if default_state != state:
            default_state = state
            output(state)
        sleep(1000)


main()
