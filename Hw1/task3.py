def longestCommonPrefix(str):
    common_prefix = ''
    for i in range(len(str[0].strip())+1):
        flag = 1
        for word in str[1:]:
            if str[0].strip()[:i] != word.strip()[:i]:
                flag = 0
        if flag: common_prefix = str[0].strip()[:i]
    return common_prefix
print(longestCommonPrefix(["1" for _ in range(5)]))
