import os

class Test:
    def __enter__(self):
        print('__enter__()')
        # return 1234

    def __exit__(self, *arg):
        print('__exit__()', arg)

if __name__ == '__main__':
    with Test() as test:
        print(test)

    with Test():
        print('sdsdsdsd')