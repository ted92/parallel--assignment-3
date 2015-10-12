#!/usr/bin/env python

import argparse
import precode

def main():
    parser = argparse.ArgumentParser(description='Encrypt a message with CBC XTEA')
    parser.add_argument('input', default='encrypted', help='Encrypted file')
    parser.add_argument('-p', help='Password (if none is given, we will brute-force)')
    parser.add_argument('known', default='secret', help='What part of the plaintext is known')
    parser.add_argument('-l', default=2, type=int, help='Max length of password before we give up')

    args = parser.parse_args()

    encrypted = precode.load_secret(args.input)
    if(args.p):
        if(precode.try_password(encrypted, args.p, args.known)):
            print("Password is correct")
        else:
            print("The password is incorrect")
    else:
        result = precode.guess_password(args.l, encrypted, args.known)
        if(type(result) != bool):
            print("Password found: " + result)
    

if __name__ == "__main__":
    main()
