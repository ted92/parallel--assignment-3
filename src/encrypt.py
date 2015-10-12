#!/usr/bin/env python

import argparse
import precode

def main():
    parser = argparse.ArgumentParser(description='Encrypt a message with CBC XTEA')
    parser.add_argument('-o', default='encrypted', help='Output file')
    parser.add_argument('-s', default='Secret message', help='Secret text to encrypt')
    parser.add_argument('-l', default='1000', type=int, help='Length of output in bytes')
    parser.add_argument('-p', default='a', help='Password')

    args = parser.parse_args()

    secret = precode.generate_secret(args.l, args.s)
    encrypted = precode.encrypt_bytes(secret, args.p)

    precode.save_secret(args.o, encrypted)
    print("File encrypted")

if __name__ == "__main__":
    main()
