#!/bin/bash

set -e

WD="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

if [ "$1" != '' ]; then
	if [ "$1" == '--help' ]; then
		echo -e "Usage: $0 [python_version [path_to_lybpython.a]]\n\nExamples:\n\t$0\n\t$0 3.12\n\t$0 3.12 ./libpython3.12.0.a"
		exit 0
	fi
	VER=$1
else
	VER=$(python3 -V 2>&1 | grep -oP '(?<=Python )[0-9]+\.[0-9]+')
fi
echo "Building for python $VER"

python3 "$WD/amalgamate.py" "$WD/../mpc.py" /tmp/mpc.py

cython /tmp/mpc.py --embed
rm /tmp/mpc.py
if [ "$2" != '' ]; then
	if [ -f "$2" ]; then
		echo "Statically linking against $2"
		gcc -Os -I /usr/include/python${VER} /tmp/mpc.c "$2" -lm -o mpc-static
		rm /tmp/mpc.c
	else
		echo "To build statically against libpython.a provide a full path to it"
	fi
else
	gcc -Os -I /usr/include/python${VER} /tmp/mpc.c -lpython${VER} -o mpc
	rm /tmp/mpc.c
fi
