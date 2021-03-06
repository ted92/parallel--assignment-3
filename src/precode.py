#!/usr/bin/env python

"""
Precode for assignment 3
INF-3201, UIT
Written by Edvard Pedersen
"""

import numpy as np
import random
import hashlib
import string
import collections
import time
import decrypt_cuda
import decrypt_cuda_v2

base_path = "."
known = "but safe"

# number of maximum key to process in parallel
NUM_KEY = 20

def main():
    """Run when invoked as stand-alone.
    Encrypt something, then time how long it takes to decrypt it by guessing the password.
    Prints out the time it took.
    """
    secret = generate_secret(1000, "this is secret, but safe")
    # numpy array which contains the message in a random order in 8 bit and the order in the other 24 bit

    encrypted = encrypt_bytes(secret, "Z")
    start_time = time.time()

    # sequential version
    result = guess_password(3, encrypted, known)
    end_time = time.time()
    print("The password is \"" + result + "\"\nTime taken (Python): " + str(end_time - start_time))

def load_secret(filename):
    """Load numpy array from disk"""
    return np.load(filename)

def save_secret(filename, array):
    """Save numpy array to disk"""
    np.save(filename, array)

def guess_password(max_length, in_data, known_part):
    """Iterative brute-force password guessing.

    Arguments:
    max_length -- the maximum length of the password (int)
    in_data -- the encrypted data (numpy array)
    known_part -- A part of the decoded data which is known (string)

    returns -- False or the correct guess
    """

    # This is a fairly silly way to do this
    # Must be changed for longer passwords to avoid running out of RAM

    # pull all the possible strings
    guesses = collections.deque(string.printable)


    while(guesses):

        # pop first NUM_KEY guesses if there are any
        if(len(guesses)<NUM_KEY):

            cur_guess = guesses.popleft()

            if(len(cur_guess) > max_length):
                return False

            # decrypt and recondtruct the message
            # decrypt_bytes in here is more useful to split each key for each threads

            # cuda version
            decrypted = decrypt_cuda.decrypt_bytes(in_data, cur_guess)

            # sequential version
            # decrypted = decrypt_bytes(in_data, cur_guess)
            reconstructed = reconstruct_secret(decrypted)

            if(try_password(reconstructed, known_part)):
                return cur_guess
            else:
                if(len(cur_guess) < max_length):
                    for char in string.printable:
                        guesses.append(cur_guess + char)

        else:
            # call another parallel class which parallelize also the keys
            cur_guesses = np.empty(NUM_KEY, dtype=np.uint32)
            # pop first NUM_KEY elements from guesses
            i = 0
            while (i < NUM_KEY):
                cur_guesses[i] = guesses.popleft()
                i = i + 1
            # return a list of decrypted messages
            decrypted_list = decrypt_cuda_v2.decrypt_bytes(in_data, cur_guesses)

            # in decrypted_list there is a long array made of all the decrypted values from 500 keys.
            # each len(in_data) position there is a new key

            #TODO: PROBLEM: HOW DO I RECOGNIZE WHICH KEY IS? according to i number
            for i in range(len(decrypted_list)/len(in_data)):
                reconstructed = reconstruct_secret(decrypted_list[(i*len(in_data)):((i+1) * len(in_data))])
                if(try_password(reconstructed, known_part)):
                    return cur_guesses[i]
                else:
                    if(len(cur_guesses[i]) < max_length):
                        for char in string.printable:
                            guesses.append(cur_guesses[i] + char)



def try_password(reconstructed, known_part):
    """Check if a password is correct by decrypting, reconstructing, and looking for the known string.

    Arguments:
        reconstructed -- message deciphred and reconstructed
        known_part -- known part of the plaintext

    returns -- True if the guess is correct, False otherwise
    """
    if(type(reconstructed) == bool):
        return False
    if(known_part in reconstructed.tostring()):
         return True
    else:
        return False

def encrypt_bytes(bytes_in, key):
    """Encrypts a numpy array with key using XTEA in CBC mode.
    
    Arguments:
    bytes_in -- numpy array to encrypt
    key -- password to use
    
    returns -- a new numpy array containing the encrypted data"""
    iv = np.array([1,2], dtype=np.uint32)
    ha = np.fromstring(hashlib.md5(key).digest(), np.uint32)

    output = np.empty_like(bytes_in)

    prev_encrypt = iv
    i = 0
    length = len(bytes_in)
    while(i < length - 1):
        output[i:i+2] = encipher(32, np.bitwise_xor(bytes_in[i:i+2], prev_encrypt), ha)
        prev_encrypt = output[i:i+2]
        i += 2

    return output

def decrypt_bytes(bytes_in, key):
    """Decrypts a numpy array with key using XTEA in CBC mode.
    
    Arguments:
    bytes_in -- numpy array with encrypted data
    key -- password to use to decrypt
    
    returns -- a new numpy array containing the decrypted data"""
    iv = np.array([1,2], dtype=np.uint32)
    ha = np.fromstring(hashlib.md5(key).digest(), np.uint32)

    output = np.empty_like(bytes_in)

    prev_decrypt = iv
    i = 0
    length = len(bytes_in)
    while(i < length - 1):
        output[i:i+2] = np.bitwise_xor(decipher(32, bytes_in[i:i+2], ha), prev_decrypt)

        prev_decrypt = bytes_in[i:i+2]
        i += 2
    return output

def generate_secret(size, secret):
    """Generates a numpy array which contains random data and the secret data.
    The array consists of 32-bit data points, where the top 24 bits contain the 
    offset of the bottom 8 bits in the reconstructed array.

    Arguments:
    size -- size of the numpy array to be generated (int)
    secret -- secret message (str)

    returns -- a new numpy array with the shuffled data"""
    if(size < len(secret)):
        raise Exception("Secret too large for secret generation")

    secret2 = np.fromstring(secret, np.int8)
    
    positions = np.arange(size)
    max_position = size - len(secret)
    
    secret_array = np.empty(size, dtype=np.uint32)
    insertion_array = np.random.randint(255,size=size)

    target_position = random.choice(range(max_position))
    
    i = 0
    for element in secret2:
        insertion_array[target_position + i] = element
        i+=1

    np.random.shuffle(positions)
    i = 0
    possible_modifiers = range(0, (2**23) - size, size)
    for position in positions:
        secret_array[i] = (position + random.choice(possible_modifiers)) << 8 | insertion_array[position]
        i+=1

    return secret_array

def reconstruct_secret(secret):
    """Un-shuffles the data in secret.

    Arguments:
    secret -- the shuffled numpy array

    returns -- a new numpy array containing the de-shuffled data"""
    result = np.empty_like(secret.astype(np.uint8))
    mask = 0xffL
    for element in secret:
        try:
            result[(element >> 8) % len(result)] = element & mask
        except IndexError:
            print("IndexError, this should never happen")
            return False
    return result

def encipher(num_rounds, input_data, key):
    """XTEA implementation in python, encryption.

    Modified version from Simon Biewald (http://varbin.github.io/xtea/)

    Arguments:
    num_rounds -- the number of iterations in the algorithm, 32 is reccomended
    input_data -- the input data to use, 32 bits of the first 2 elements are used
    key -- 128-bit key to use

    returns -- a numpy array containing the enciphered data"""
    v0 = input_data[0]
    v1 = input_data[1]
    sum = 0L
    delta = 0x9e3779b9L
    mask = 0xffffffffL
    for round in range(num_rounds):
        v0 = (v0 + (((v1<<4 ^ v1>>5) + v1) ^ (sum + key[sum & 3]))) & mask
        sum = (sum + delta) & mask
        v1 = (v1 + (((v0<<4 ^ v0>>5) + v0) ^ (sum + key[sum>>11 & 3]))) & mask
    return np.array([v0, v1], dtype=np.uint32)


# do not call this decipher but the one in decrypt_cuda
def decipher(num_rounds, input_data, key):
    """XTEA implementation in python, decryption.

    Modified version from Simon Biewald (http://varbin.github.io/xtea/)

    Arguments:
    num_rounds -- the number of iterations in the algorithm, 32 is reccomended
    input_data -- the input data to use, 32 bits of the first 2 elements are used
    key -- 128-bit key to use

    returns -- a numpy array containing the deciphered data"""
    v0 = input_data[0]
    v1 = input_data[1]
    delta = 0x9e3779b9L
    mask = 0xffffffffL
    sum = (delta*num_rounds) & mask

    for round in range(num_rounds):
        v1 = (v1 - (((v0<<4 ^ v0>>5) + v0) ^ (sum + key[sum>>11 & 3]))) & mask
        sum = (sum - delta) & mask
        v0 = (v0 - (((v1<<4 ^ v1>>5) + v1) ^ (sum + key[sum & 3]))) & mask
    return np.array([v0, v1], dtype=np.uint32)

if __name__ == "__main__":
    main()
