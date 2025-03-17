import time


def process_file() -> None:
    time.sleep(1)
    for i in range(10**6):
        pass


def sort() -> None:
    time.sleep(1)
    for i in range(10**8):
        pass
    process_file()


def main() -> None:
    process_file()
    sort()


if __name__ == "__main__":
    main()
