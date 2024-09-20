import numpy as np
import string

def generate_random_string(length):
    # 사용할 문자를 정의 (대문자, 소문자, 숫자 포함)
    characters = string.ascii_letters + string.digits + ' '
    # 무작위로 문자열 생성
    random_string = ''.join(np.random.choice(list(characters), length))
    return random_string

# 길이가 10인 랜덤 문자열 생성
random_str = generate_random_string(10)
print(random_str)

