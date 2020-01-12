test: test.o sinkhorn2.o
		gcc $(CFLAGS) -o test -std="c11" test.o sinkhorn2.o -lm -lpthread

sinkhorn2.o: sinkhorn2.c test.h
		gcc $(CFLAGS) -std="c11" -c sinkhorn2.c -lm

test.o: test.c test.h
		gcc $(CFLAGS) -std="c11" -c test.c -lm -lpthread


clean:
		rm -f *.o test
