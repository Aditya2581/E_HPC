# Define the SHA-256 constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]


# Define the SHA-256 functions
def rotr(x, n):
    return (x >> n) | (x << (32 - n))


def shr(x, n):
    return x >> n


def ch(x, y, z):
    return (x & y) ^ (~x & z)


def maj(x, y, z):
    return (x & y) ^ (x & z) ^ (y & z)


def sigma0(x):
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)


def sigma1(x):
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)


def Gamma0(x):
    return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)


def Gamma1(x):
    return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)




def sha256(message):
    # Pre-processing
    bit_len = len(message) * 8
    message += b'\x80'
    while (len(message) + 8) % 64 != 0:
        message += b'\x00'
    message += bit_len.to_bytes(8, byteorder='big')

    # Initialize hash values
    h = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]

    # Process message in 512-bit chunks
    for i in range(0, len(message), 64):
        chunk = message[i:i+64]
        w = [0] * 64

        # Divide chunk into sixteen 32-bit big-endian words
        for j in range(16):
            w[j] = int.from_bytes(chunk[j*4:j*4+4], byteorder='big')

        # Extend the sixteen 32-bit words into sixty-four 32-bit words
        for j in range(16, 64):
            s0 = sigma0(w[j-15])
            s1 = sigma1(w[j-2])
            w[j] = (w[j-16] + s0 + w[j-7] + s1) & 0xffffffff

        # Initialize working variables
        a, b, c, d, e, f, g, h = h

        # Main loop
        for j in range(64):
            S1 = Gamma1(e)
            ch_val = ch(e, f, g)
            temp1 = h + S1 + ch_val + K[j] + w[j]
            S0 = Gamma0(a)
            maj_val = maj(a, b, c)
            temp2 = S0 + maj_val

            h = g
            g = f
            f = e
            e = (d + temp1) & 0xffffffff
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xffffffff

        # Update hash values
        h = (h + e) & 0xffffffff
        g = (g + f) & 0xffffffff
        f = (f + e) & 0xffffffff
        e = (e + d) & 0xffffffff
        d = (d + c) & 0xffffffff
        c = (c + b) & 0xffffffff
        b = (b + a) & 0xffffffff
        a = (a + h) & 0xffffffff

    # Produce the final hash value
    hash_val = (a, b, c, d, e, f, g, h)
    hash_bytes = b''.join([x.to_bytes(4, byteorder='big') for x in hash_val])
    return hash_bytes.hex()



message = b'abc'
hash_val = sha256(message)
print(hash_val)

