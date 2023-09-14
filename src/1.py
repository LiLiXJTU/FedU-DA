def deleteSameNum(nums):
    if not nums:
        return 0
    n = len(nums)
    fast = slow = 1
    while fast < n:
        if nums[fast] != nums[fast - 1]:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    return fast


if __name__ == "__main__":
    num = [1, 1, 1, 4, 5,5]
    print(deleteSameNum(num))