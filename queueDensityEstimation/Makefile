all:
	gcc -c -Wall -Werror -fpic FLGR.c
	gcc -shared -o libfoo.so FLGR.o
	g++ -std=c++11 -pthread -L. -Wall -o code_no_dyn.o code_no_dyn.cpp -lfoo `pkg-config --cflags --libs opencv`
